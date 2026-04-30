"""Manifest-driven instance dataset for xBD building damage classification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely.geometry import Polygon
from torch.utils.data import Dataset

from data.crop_policy import build_tight_context_crops, polygon_to_bbox
from data.manifest import (
    load_manifest_dataframe,
    parse_json_field,
    resolve_path,
    validate_manifest_dataframe,
)
from data.mask_ops import bbox_to_mask, polygon_to_mask
from data.transforms import apply_sync_transform, build_sync_transform, image_to_tensor, mask_to_tensor


class XBDInstanceDataset(Dataset):
    """Instance-level xBD dataset that emits tight/context paired crops."""

    def __init__(self, data_cfg: Dict[str, Any], split: str) -> None:
        self.data_cfg = dict(data_cfg)
        self.split = split
        self.manifest_path = Path(self.data_cfg["manifest_path"])
        self.image_root = self.data_cfg.get("image_root", "")
        self.tight_size = int(self.data_cfg.get("tight_size", 256))
        self.context_size = int(self.data_cfg.get("context_size", 256))
        self.tight_padding_ratio = float(self.data_cfg.get("tight_padding_ratio", 0.15))
        self.context_padding_ratio = float(self.data_cfg.get("context_padding_ratio", 1.0))
        self.use_mask = bool(self.data_cfg.get("use_mask", True))
        self.mask_mode = str(self.data_cfg.get("mask_mode", "mask_pooling"))
        self.overfit_n_samples = self.data_cfg.get("overfit_n_samples")

        df = load_manifest_dataframe(self.manifest_path)
        validate_manifest_dataframe(df)
        df = df[df["split"].astype(str) == split].reset_index(drop=True)
        if self.overfit_n_samples:
            df = df.iloc[: int(self.overfit_n_samples)].reset_index(drop=True)
        self.samples = self._filter_valid_rows(df)
        is_train = split == self.data_cfg.get("split_train", "train")
        self.transform = build_sync_transform(self.data_cfg, is_train=is_train)

    def _filter_valid_rows(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            polygon = parse_json_field(row.get("polygon"))
            bbox = parse_json_field(row.get("bbox"))
            if polygon is None and bbox is None:
                continue
            if int(row["label"]) not in {0, 1, 2, 3}:
                continue
            samples.append(row)
        if not samples:
            raise ValueError(
                f"No valid samples found for split='{self.split}' in manifest: {self.manifest_path}. "
                "Check the split column, labels, and geometry fields."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path_str: str) -> np.ndarray:
        path = resolve_path(path_str, self.image_root)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return np.asarray(Image.open(path).convert("RGB"))

    def _resolve_geometry(self, row: Dict[str, Any], image_shape: tuple[int, int]) -> tuple[Optional[list[list[float]]], list[float], np.ndarray, float]:
        polygon = parse_json_field(row.get("polygon"))
        bbox = parse_json_field(row.get("bbox"))

        if polygon is not None and len(polygon) >= 3:
            bbox = polygon_to_bbox(polygon)
            mask = polygon_to_mask(image_shape, polygon)
            try:
                area = float(Polygon(polygon).area)
            except Exception:
                area = float(mask.sum())
            return polygon, bbox, mask, area

        if bbox is not None and len(bbox) == 4:
            mask = bbox_to_mask(image_shape, bbox)
            area = float(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            return None, [float(v) for v in bbox], mask, area

        raise ValueError(
            f"Sample building_id={row.get('building_id')} has neither valid polygon nor bbox."
        )

    def _prepare_masks(self, crops: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        mask_tight = crops["mask_tight"].astype(np.uint8)
        mask_context = crops["mask_context"].astype(np.uint8)
        if not self.use_mask or self.mask_mode == "none":
            mask_tight = np.zeros_like(mask_tight, dtype=np.uint8)
            mask_context = np.zeros_like(mask_context, dtype=np.uint8)
        return mask_tight, mask_context

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.samples[index]
        pre_image = self._load_image(str(row["pre_image"]))
        post_image = self._load_image(str(row["post_image"]))
        if pre_image.shape[:2] != post_image.shape[:2]:
            raise ValueError(
                f"Pre/Post image size mismatch for building_id={row.get('building_id')}: "
                f"{pre_image.shape[:2]} vs {post_image.shape[:2]}"
            )

        polygon, bbox, base_mask, area = self._resolve_geometry(row, pre_image.shape[:2])
        crops = build_tight_context_crops(
            pre_image=pre_image,
            post_image=post_image,
            mask=base_mask,
            bbox=bbox,
            tight_size=self.tight_size,
            context_size=self.context_size,
            tight_padding_ratio=self.tight_padding_ratio,
            context_padding_ratio=self.context_padding_ratio,
        )

        transformed = apply_sync_transform(self.transform, crops)
        mask_tight, mask_context = self._prepare_masks(transformed)

        pre_tight = transformed["pre_tight"]
        post_tight = transformed["post_tight"]
        pre_context = transformed["pre_context"]
        post_context = transformed["post_context"]

        if self.mask_mode == "rgbm":
            pre_tight = np.concatenate([pre_tight, mask_tight[..., None].astype(np.float32)], axis=2)
            post_tight = np.concatenate([post_tight, mask_tight[..., None].astype(np.float32)], axis=2)
            pre_context = np.concatenate([pre_context, mask_context[..., None].astype(np.float32)], axis=2)
            post_context = np.concatenate([post_context, mask_context[..., None].astype(np.float32)], axis=2)

        label = int(row["label"])
        binary_label = 0 if label in (0, 1) else 1

        return {
            "pre_tight": image_to_tensor(pre_tight),
            "post_tight": image_to_tensor(post_tight),
            "pre_context": image_to_tensor(pre_context),
            "post_context": image_to_tensor(post_context),
            "mask_tight": mask_to_tensor(mask_tight),
            "mask_context": mask_to_tensor(mask_context),
            "label": torch.tensor(label, dtype=torch.long),
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "building_id": str(row["building_id"]),
            "disaster_id": str(row["disaster_id"]),
            "area": float(area if math.isfinite(area) else 0.0),
        }
