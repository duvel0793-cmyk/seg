from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from datasets.polygon_utils import (
    infer_disaster_name,
    is_valid_polygon,
    load_label_png,
    parse_wkt_polygon,
    polygon_area,
    polygon_bbox,
    polygon_to_mask,
)
from datasets.transforms import build_transforms
from utils.misc import LABEL_TO_INDEX, read_json

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TilePaths:
    subset: str
    pre_image: str
    post_image: str
    post_label: str
    post_target: str | None


def _read_split_list(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _resolve_target_png(targets_dir: Path, tile_id: str) -> Path | None:
    candidates = [
        targets_dir / f"{tile_id}_post_disaster_target.png",
        targets_dir / f"{tile_id}_damage_target.png",
        targets_dir / f"{tile_id}_post_disaster_damage_target.png",
        targets_dir / f"{tile_id}_target.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_tile_paths(root_dir: str | Path, tile_id: str, allow_tier3: bool = False) -> TilePaths | None:
    root_dir = Path(root_dir)
    subsets = ["train", "hold", "test"] + (["tier3"] if allow_tier3 else [])
    for subset in subsets:
        base_dir = root_dir / subset
        pre_image = base_dir / "images" / f"{tile_id}_pre_disaster.png"
        post_image = base_dir / "images" / f"{tile_id}_post_disaster.png"
        post_label = base_dir / "labels" / f"{tile_id}_post_disaster.json"
        if pre_image.exists() and post_image.exists() and post_label.exists():
            target_png = _resolve_target_png(base_dir / "targets", tile_id)
            return TilePaths(
                subset=subset,
                pre_image=str(pre_image),
                post_image=str(post_image),
                post_label=str(post_label),
                post_target=None if target_png is None else str(target_png),
            )
    return None


def _mask_to_box_xyxy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _xyxy_to_cxcywh_normalized(box: list[float], height: int, width: int) -> list[float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / float(width)
    cy = ((y1 + y2) * 0.5) / float(height)
    bw = (x2 - x1) / float(width)
    bh = (y2 - y1) / float(height)
    return [cx, cy, bw, bh]


class XBDQueryDataset(Dataset):
    def __init__(self, *, config: dict[str, Any], split: str, is_train: bool) -> None:
        super().__init__()
        self.config = config
        self.split = str(split)
        self.is_train = bool(is_train)
        self.root_dir = str(config["data"]["root"])
        self.allow_tier3 = bool(config.get("dataset", {}).get("allow_tier3", False))
        self.min_polygon_area = float(config.get("dataset", {}).get("min_polygon_area", 16.0))
        self.min_mask_pixels = int(config.get("dataset", {}).get("min_mask_pixels", 8))
        self.image_size = int(config.get("dataset", {}).get("image_size", 512))
        self.transforms = build_transforms(config, is_train=is_train)
        split_key = f"{self.split}_list"
        if split_key not in config["data"]:
            raise KeyError(f"Missing data.{split_key} in config.")
        self.tile_ids = _read_split_list(config["data"][split_key])
        debug_subset = int(config.get("dataset", {}).get("debug_subset", 0) or 0)
        if debug_subset > 0:
            self.tile_ids = self.tile_ids[:debug_subset]
        self.samples = self._index_tiles()

    def _index_tiles(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for tile_id in self.tile_ids:
            tile_paths = _resolve_tile_paths(self.root_dir, tile_id, allow_tier3=self.allow_tier3)
            if tile_paths is None:
                continue
            payload = read_json(tile_paths.post_label)
            metadata = payload.get("metadata", {})
            width = int(metadata.get("width", metadata.get("original_width", 1024)))
            height = int(metadata.get("height", metadata.get("original_height", 1024)))
            instances: list[dict[str, Any]] = []
            for building_idx, feature in enumerate(payload.get("features", {}).get("xy", [])):
                properties = feature.get("properties", {})
                if properties.get("feature_type") != "building":
                    continue
                subtype = properties.get("subtype")
                if subtype not in LABEL_TO_INDEX:
                    continue
                try:
                    polygon = parse_wkt_polygon(feature.get("wkt", ""))
                except Exception:
                    continue
                if not is_valid_polygon(polygon, min_area=self.min_polygon_area):
                    continue
                instances.append(
                    {
                        "building_idx": int(building_idx),
                        "label": int(LABEL_TO_INDEX[subtype]),
                        "subtype": str(subtype),
                        "polygon_xy": [(float(x), float(y)) for x, y in polygon],
                        "bbox_xyxy": [float(v) for v in polygon_bbox(polygon)],
                        "polygon_area": float(polygon_area(polygon)),
                    }
                )
            samples.append(
                {
                    "tile_id": tile_id,
                    "paths": tile_paths,
                    "width": width,
                    "height": height,
                    "instances": instances,
                    "disaster_name": infer_disaster_name(tile_id, payload),
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        paths: TilePaths = sample["paths"]
        pre_image = Image.open(paths.pre_image).convert("RGB")
        post_image = Image.open(paths.post_image).convert("RGB")
        orig_height = int(sample["height"])
        orig_width = int(sample["width"])

        instance_masks: list[np.ndarray] = []
        instance_labels: list[int] = []
        instance_ids: list[int] = []
        instance_polygons: list[list[tuple[float, float]]] = []
        for instance in sample["instances"]:
            mask = polygon_to_mask(instance["polygon_xy"], orig_height, orig_width)
            if int(mask.sum()) < self.min_mask_pixels:
                continue
            instance_masks.append(mask.astype(np.uint8))
            instance_labels.append(int(instance["label"]))
            instance_ids.append(int(instance["building_idx"]))
            instance_polygons.append(instance["polygon_xy"])

        gt_damage_map = None
        if paths.post_target is not None:
            gt_damage_map = load_label_png(paths.post_target).astype(np.uint8)
        elif instance_masks:
            gt_damage_map = np.zeros((orig_height, orig_width), dtype=np.uint8)
            for mask, label in zip(instance_masks, instance_labels):
                gt_damage_map[mask > 0] = np.maximum(gt_damage_map[mask > 0], np.uint8(label + 1))

        transformed = self.transforms(
            pre_image=pre_image,
            post_image=post_image,
            instance_masks=instance_masks,
            damage_map=gt_damage_map,
        )

        resized_masks = transformed["instance_masks"]
        kept_masks: list[np.ndarray] = []
        kept_labels: list[int] = []
        kept_ids: list[int] = []
        kept_boxes: list[list[float]] = []
        kept_boxes_norm: list[list[float]] = []
        for mask, label, building_idx in zip(resized_masks, instance_labels, instance_ids):
            if int(mask.sum()) < self.min_mask_pixels:
                continue
            box = _mask_to_box_xyxy(mask)
            if box is None:
                continue
            kept_masks.append(mask.astype(np.uint8))
            kept_labels.append(int(label))
            kept_ids.append(int(building_idx))
            kept_boxes.append(box)
            kept_boxes_norm.append(_xyxy_to_cxcywh_normalized(box, self.image_size, self.image_size))

        if kept_masks:
            masks_tensor = torch.from_numpy(np.stack(kept_masks, axis=0)).float()
            boxes_tensor = torch.tensor(kept_boxes, dtype=torch.float32)
            boxes_norm_tensor = torch.tensor(kept_boxes_norm, dtype=torch.float32)
            labels_tensor = torch.tensor(kept_labels, dtype=torch.long)
            ids_tensor = torch.tensor(kept_ids, dtype=torch.long)
        else:
            masks_tensor = torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            boxes_norm_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
            ids_tensor = torch.zeros((0,), dtype=torch.long)

        damage_map_tensor = None
        if transformed["damage_map"] is not None:
            damage_map_tensor = torch.from_numpy(transformed["damage_map"].astype(np.int64))

        target = {
            "labels": labels_tensor,
            "boxes": boxes_tensor,
            "boxes_norm": boxes_norm_tensor,
            "masks": masks_tensor,
            "image_id": str(sample["tile_id"]),
            "instance_ids": ids_tensor,
            "orig_size": torch.tensor([orig_height, orig_width], dtype=torch.long),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        return {
            "pre_image": transformed["pre_image"],
            "post_image": transformed["post_image"],
            "gt_damage_map": damage_map_tensor,
            "target": target,
            "image_id": str(sample["tile_id"]),
            "original_size": (orig_height, orig_width),
            "scaled_size": (self.image_size, self.image_size),
            "meta": {
                "tile_id": str(sample["tile_id"]),
                "pre_image_path": paths.pre_image,
                "post_image_path": paths.post_image,
                "post_label_path": paths.post_label,
                "post_target_path": paths.post_target,
                "disaster_name": sample["disaster_name"],
                "source_subset": paths.subset,
                "num_instances": int(labels_tensor.numel()),
                "geometry": transformed["geometry"],
            },
        }
