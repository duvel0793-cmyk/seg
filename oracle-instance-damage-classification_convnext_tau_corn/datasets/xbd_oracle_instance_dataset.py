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

        crop_box = self._build_crop_box(record["bbox_xyxy"], image_size=pre_image.size)
        pre_crop = pre_image.crop(crop_box)
        post_crop = post_image.crop(crop_box)
        mask_crop = self._rasterize_crop_mask(
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
            "pre_image_path": record["pre_image_path"],
            "post_image_path": record["post_image_path"],
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
        x1, y1, x2, y2 = bbox_xyxy
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        crop_width = max(self.min_crop_size, int(math.ceil(box_width * (1.0 + 2.0 * self.crop_context_ratio))))
        crop_height = max(self.min_crop_size, int(math.ceil(box_height * (1.0 + 2.0 * self.crop_context_ratio))))

        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        left = int(math.floor(center_x - crop_width / 2.0))
        top = int(math.floor(center_y - crop_height / 2.0))
        right = left + crop_width
        bottom = top + crop_height

        if right <= left:
            right = left + self.min_crop_size
        if bottom <= top:
            bottom = top + self.min_crop_size

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Invalid image size.")

        return left, top, right, bottom

    def _rasterize_crop_mask(
        self,
        polygons: List[List[Tuple[float, float]]],
        crop_box: Tuple[int, int, int, int],
        crop_size: Tuple[int, int],
        bbox_xyxy: List[float],
    ) -> Image.Image:
        crop_width, crop_height = crop_size
        left, top, _, _ = crop_box
        mask = Image.new("L", (crop_width, crop_height), 0)
        draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            shifted = [(x - left, y - top) for x, y in polygon]
            if len(shifted) >= 3:
                draw.polygon(shifted, outline=1, fill=1)

        if np.asarray(mask).max() == 0:
            x1, y1, x2, y2 = bbox_xyxy
            fallback_box = [x1 - left, y1 - top, x2 - left, y2 - top]
            draw.rectangle(fallback_box, outline=1, fill=1)

        if np.asarray(mask).max() == 0:
            center_x = crop_width // 2
            center_y = crop_height // 2
            draw.rectangle([center_x, center_y, center_x + 1, center_y + 1], outline=1, fill=1)

        return mask


def dataset_record_distribution(records: List[Dict]) -> Dict[int, int]:
    """Summarize class counts from raw parsed records."""
    stats: Dict[int, int] = {}
    for item in records:
        label = int(item["label"])
        stats[label] = stats.get(label, 0) + 1
    return stats

