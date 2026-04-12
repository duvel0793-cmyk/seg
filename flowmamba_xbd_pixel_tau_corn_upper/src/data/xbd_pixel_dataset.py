"""xBD pixel-level dataset for post-disaster damage supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifest import build_manifest
from .transforms import XBDTransformPipeline
from .xbd_polygon import load_xbd_polygons, serialize_polygons


class XBDPixelDataset(Dataset):
    """Load pre/post images, post target PNG, and polygon annotations."""

    def __init__(
        self,
        data_root: str,
        mode: str,
        list_path: str,
        preferred_split: str,
        crop_size: Tuple[int, int],
        image_size: Tuple[int, int],
        ignore_index: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        augment_cfg: Dict[str, bool],
        validation_cfg: Dict[str, object],
        min_polygon_area: float = 4.0,
        strict: bool = False,
    ) -> None:
        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.data_root = Path(data_root).expanduser().resolve()
        self.mode = mode
        self.ignore_index = int(ignore_index)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.min_polygon_area = float(min_polygon_area)

        self.manifest, self.manifest_stats = build_manifest(
            data_root=self.data_root,
            preferred_split=preferred_split,
            list_path=list_path,
        )
        if strict and self.manifest_stats["num_missing"] > 0:
            raise FileNotFoundError(
                f"Found {self.manifest_stats['num_missing']} invalid xBD samples while strict mode is enabled."
            )
        if not self.manifest:
            raise RuntimeError(
                f"No valid samples found for mode={mode}, split={preferred_split}, list_path={list_path}"
            )

        self.transform = XBDTransformPipeline(
            mode=mode,
            crop_size=tuple(crop_size),
            image_size=tuple(image_size),
            augment_cfg=augment_cfg,
            min_polygon_area=self.min_polygon_area,
            validation_cfg=validation_cfg,
        )

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        return np.asarray(Image.open(path))

    @staticmethod
    def _validate_target(post_target: np.ndarray, image_id: str) -> None:
        valid_values = set(np.unique(post_target).tolist())
        allowed = {0, 1, 2, 3, 4}
        if not valid_values.issubset(allowed):
            raise ValueError(f"Unexpected target values for {image_id}: {sorted(valid_values)}")

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.manifest[index]
        pre_image = self._load_rgb(item["pre_image"])
        post_image = self._load_rgb(item["post_image"])
        post_target = self._load_mask(item["post_target"]).astype(np.uint8)
        self._validate_target(post_target, item["image_id"])

        polygons = load_xbd_polygons(item["post_json"], min_area=self.min_polygon_area)
        sample = {
            "pre_image": pre_image,
            "post_image": post_image,
            "post_target": post_target,
            "polygons": polygons,
            "image_id": item["image_id"],
            "meta": {
                "split": item["split"],
                "original_size_hw": [int(post_target.shape[0]), int(post_target.shape[1])],
                "paths": {
                    "pre_image": item["pre_image"],
                    "post_image": item["post_image"],
                    "post_target": item["post_target"],
                    "post_json": item["post_json"],
                },
            },
        }

        sample = self.transform(sample)
        post_target = sample["post_target"].astype(np.int64)
        loc_target = (post_target > 0).astype(np.int64)
        damage_rank_target = np.full_like(post_target, self.ignore_index, dtype=np.int64)
        building_mask = post_target > 0
        damage_rank_target[building_mask] = post_target[building_mask] - 1

        pre_image = sample["pre_image"].astype(np.float32) / 255.0
        post_image = sample["post_image"].astype(np.float32) / 255.0
        pre_image = (pre_image - self.mean) / self.std
        post_image = (post_image - self.mean) / self.std

        return {
            "pre_image": torch.from_numpy(pre_image.transpose(2, 0, 1)).float(),
            "post_image": torch.from_numpy(post_image.transpose(2, 0, 1)).float(),
            "post_target": torch.from_numpy(post_target).long(),
            "loc_target": torch.from_numpy(loc_target).long(),
            "damage_rank_target": torch.from_numpy(damage_rank_target).long(),
            "polygons": serialize_polygons(sample["polygons"]),
            "image_id": sample["image_id"],
            "meta": sample["meta"],
        }
