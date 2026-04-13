"""Oracle instance dataset built from xBD json building annotations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from datasets.transforms import EvalTransform, TrainTransform
from datasets.xbd_json_parser import XBDJsonParser


class XBDOracleInstanceDataset(Dataset):
    """Per-building xBD dataset using oracle polygons and masked crops."""

    def __init__(
        self,
        data_root: str,
        split: str,
        train_split_file: str = "",
        val_split_file: str = "",
        image_size: int = 224,
        crop_context_ratio: float = 0.2,
        min_crop_size: int = 64,
        use_vertical_flip: bool = False,
        use_color_jitter: bool = True,
    ) -> None:
        super().__init__()
        self.split = split
        self.crop_context_ratio = float(crop_context_ratio)
        self.min_crop_size = int(min_crop_size)

        parser = XBDJsonParser(
            data_root=data_root,
            train_split_file=train_split_file,
            val_split_file=val_split_file,
        )
        self.records = parser.parse_split(split)
        self.parser_stats = parser.get_split_stats(split)
        self.transform = (
            TrainTransform(image_size=image_size, use_vertical_flip=use_vertical_flip, use_color_jitter=use_color_jitter)
            if split == "train"
            else EvalTransform(image_size=image_size)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        pre_image = Image.open(record["pre_image_path"]).convert("RGB")
        post_image = Image.open(record["post_image_path"]).convert("RGB")
        if pre_image.size != post_image.size:
            raise ValueError(
                f"Mismatched image sizes for sample '{record['sample_id']}': "
                f"{pre_image.size} vs {post_image.size}"
            )

        crop_box = self._build_crop_box(record["bbox_xyxy"], image_size=pre_image.size)
        pre_crop = pre_image.crop(crop_box)
        post_crop = post_image.crop(crop_box)
        mask_crop, mask_meta = self._rasterize_crop_mask(
            polygons=record["polygon"],
            crop_box=crop_box,
            crop_size=pre_crop.size,
            bbox_xyxy=record["bbox_xyxy"],
        )

        pre_tensor, post_tensor, mask_tensor = self.transform(pre_crop, post_crop, mask_crop)

        meta = {
            "sample_id": record["sample_id"],
            "image_id": record["image_id"],
            "building_id": record["building_id"],
            "crop_box_xyxy": [int(value) for value in crop_box],
            "bbox_xyxy": [float(value) for value in record["bbox_xyxy"]],
            "raw_subtype": record["raw_subtype"],
            "label": int(record["label"]),
            "polygon_source": record.get("polygon_source", "pre"),
            "pre_image_path": record["pre_image_path"],
            "post_image_path": record["post_image_path"],
            **mask_meta,
        }

        return {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(record["label"], dtype=torch.long),
            "meta": meta,
        }

    def _build_crop_box(self, bbox_xyxy: List[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        image_width, image_height = image_size
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Invalid image size.")

        x1, y1, x2, y2 = bbox_xyxy
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        desired_crop_width = max(self.min_crop_size, int(math.ceil(box_width * (1.0 + 2.0 * self.crop_context_ratio))))
        desired_crop_height = max(self.min_crop_size, int(math.ceil(box_height * (1.0 + 2.0 * self.crop_context_ratio))))

        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        left, right = self._fit_crop_axis(center=center_x, desired_size=desired_crop_width, limit=image_width)
        top, bottom = self._fit_crop_axis(center=center_y, desired_size=desired_crop_height, limit=image_height)
        return left, top, right, bottom

    def _fit_crop_axis(self, center: float, desired_size: int, limit: int) -> Tuple[int, int]:
        desired_size = max(1, min(int(desired_size), int(limit)))
        left = int(math.floor(center - desired_size / 2.0))
        right = left + desired_size

        if left < 0:
            right = min(limit, right - left)
            left = 0
        if right > limit:
            left = max(0, left - (right - limit))
            right = limit

        current_size = right - left
        if current_size < desired_size:
            deficit = desired_size - current_size
            grow_left = min(left, int(math.ceil(deficit / 2.0)))
            left -= grow_left
            deficit -= grow_left
            grow_right = min(limit - right, deficit)
            right += grow_right
            deficit -= grow_right
            if deficit > 0:
                extra_left = min(left, deficit)
                left -= extra_left
                deficit -= extra_left
                right = min(limit, right + deficit)

        if right <= left:
            left = max(0, min(left, limit - 1))
            right = min(limit, max(left + 1, right))
        return left, right

    def _rasterize_crop_mask(
        self,
        polygons: List[List[Tuple[float, float]]],
        crop_box: Tuple[int, int, int, int],
        crop_size: Tuple[int, int],
        bbox_xyxy: List[float],
    ) -> Tuple[Image.Image, Dict[str, float | int | bool]]:
        crop_width, crop_height = crop_size
        left, top, _, _ = crop_box
        mask = Image.new("L", (crop_width, crop_height), 0)
        draw = ImageDraw.Draw(mask)
        is_mask_fallback = False

        for polygon in polygons:
            shifted = [(x - left, y - top) for x, y in polygon]
            if len(shifted) >= 3:
                draw.polygon(shifted, outline=1, fill=1)

        mask_area = self._mask_area(mask)
        if mask_area == 0:
            is_mask_fallback = True
            self._draw_bbox_fallback(draw, bbox_xyxy=bbox_xyxy, crop_left=left, crop_top=top, crop_size=crop_size)
            mask_area = self._mask_area(mask)

        if mask_area == 0:
            is_mask_fallback = True
            self._draw_center_fallback(draw, crop_size=crop_size)
            mask_area = self._mask_area(mask)

        crop_area = int(crop_width * crop_height)
        mask_meta: Dict[str, float | int | bool] = {
            "mask_area": int(mask_area),
            "crop_area": crop_area,
            "mask_area_ratio": float(mask_area) / float(crop_area) if crop_area > 0 else 0.0,
            "is_mask_fallback": is_mask_fallback,
        }
        return mask, mask_meta

    @staticmethod
    def _mask_area(mask: Image.Image) -> int:
        return int((np.asarray(mask) > 0).sum())

    @staticmethod
    def _draw_bbox_fallback(
        draw: ImageDraw.ImageDraw,
        bbox_xyxy: List[float],
        crop_left: int,
        crop_top: int,
        crop_size: Tuple[int, int],
    ) -> None:
        crop_width, crop_height = crop_size
        x1, y1, x2, y2 = bbox_xyxy
        left = max(0, min(crop_width - 1, int(math.floor(x1 - crop_left))))
        top = max(0, min(crop_height - 1, int(math.floor(y1 - crop_top))))
        right = max(0, min(crop_width, int(math.ceil(x2 - crop_left))))
        bottom = max(0, min(crop_height, int(math.ceil(y2 - crop_top))))
        if right <= left:
            right = min(crop_width, left + 1)
        if bottom <= top:
            bottom = min(crop_height, top + 1)
        if right > left and bottom > top:
            draw.rectangle([left, top, right - 1, bottom - 1], outline=1, fill=1)

    @staticmethod
    def _draw_center_fallback(draw: ImageDraw.ImageDraw, crop_size: Tuple[int, int]) -> None:
        crop_width, crop_height = crop_size
        patch_half = max(1, min(crop_width, crop_height) // 16)
        center_x = crop_width // 2
        center_y = crop_height // 2
        left = max(0, center_x - patch_half)
        top = max(0, center_y - patch_half)
        right = min(crop_width, center_x + patch_half)
        bottom = min(crop_height, center_y + patch_half)
        if right <= left:
            right = min(crop_width, left + 1)
        if bottom <= top:
            bottom = min(crop_height, top + 1)
        draw.rectangle([left, top, right - 1, bottom - 1], outline=1, fill=1)


def dataset_record_distribution(records: List[Dict]) -> Dict[int, int]:
    """Summarize class counts from raw parsed records."""
    stats: Dict[int, int] = {}
    for item in records:
        label = int(item["label"])
        stats[label] = stats.get(label, 0) + 1
    return stats
