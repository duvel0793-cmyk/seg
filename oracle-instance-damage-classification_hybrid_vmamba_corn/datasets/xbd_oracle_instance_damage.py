from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.transforms import build_transforms
from utils.cache import make_cache_path, load_pickle, save_pickle
from utils.geometry import (
    clip_bbox_to_image,
    expand_bbox,
    is_small_target,
    is_valid_polygon,
    out_of_bounds_fraction,
    parse_wkt_polygon,
    polygon_area,
    polygon_bbox,
    polygon_to_mask,
)
from utils.io import read_json

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class TilePaths:
    source_subset: str
    pre_image: str
    post_image: str
    post_label: str


class BaseInstanceSource:
    name = "base"

    def build_tile_samples(
        self,
        tile_id: str,
        split_name: str,
        tile_paths: TilePaths,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class GTJsonInstanceSource(BaseInstanceSource):
    name = "gt_json"

    def build_tile_samples(
        self,
        tile_id: str,
        split_name: str,
        tile_paths: TilePaths,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        data_cfg = config["data"]
        payload = read_json(tile_paths.post_label)
        metadata = payload.get("metadata", {})
        image_width = int(metadata.get("width", metadata.get("original_width", 1024)))
        image_height = int(metadata.get("height", metadata.get("original_height", 1024)))

        features = payload.get("features", {}).get("xy", [])
        samples: list[dict[str, Any]] = []
        for building_idx, feature in enumerate(features):
            properties = feature.get("properties", {})
            subtype = properties.get("subtype")
            if properties.get("feature_type") != "building":
                continue
            if subtype not in LABEL_TO_INDEX:
                continue

            try:
                polygon_xy = parse_wkt_polygon(feature.get("wkt", ""))
            except Exception:
                continue
            if not is_valid_polygon(polygon_xy, min_area=float(data_cfg["min_polygon_area"])):
                continue
            if is_small_target(polygon_xy, min_area=float(data_cfg["min_polygon_area"])):
                continue

            tight_bbox = polygon_bbox(polygon_xy)
            crop_bbox_float = expand_bbox(tight_bbox, float(data_cfg["context_ratio"]))
            if out_of_bounds_fraction(crop_bbox_float, image_width, image_height) > float(data_cfg["max_out_of_bound_ratio"]):
                continue

            crop_bbox = clip_bbox_to_image(crop_bbox_float, image_width, image_height)
            crop_w = crop_bbox[2] - crop_bbox[0]
            crop_h = crop_bbox[3] - crop_bbox[1]
            if crop_w <= 1 or crop_h <= 1:
                continue

            mask = polygon_to_mask(polygon_xy, crop_h, crop_w, offset=(crop_bbox[0], crop_bbox[1]))
            mask_pixels = int(mask.sum())
            if mask_pixels < int(data_cfg["min_mask_pixels"]):
                continue

            sample = {
                "tile_id": tile_id,
                "building_idx": building_idx,
                "uid": properties.get("uid"),
                "label": LABEL_TO_INDEX[subtype],
                "original_subtype": subtype,
                "polygon_xy": [(float(x), float(y)) for x, y in polygon_xy],
                "polygon_area": float(polygon_area(polygon_xy)),
                "bbox_xyxy": [float(v) for v in tight_bbox],
                "crop_bbox_xyxy": [int(v) for v in crop_bbox],
                "mask_pixels": mask_pixels,
                "split": split_name,
                "source_subset": tile_paths.source_subset,
                "pre_image": tile_paths.pre_image,
                "post_image": tile_paths.post_image,
                "post_label": tile_paths.post_label,
            }
            samples.append(sample)
        return samples


INSTANCE_SOURCE_REGISTRY = {
    GTJsonInstanceSource.name: GTJsonInstanceSource,
}


def _read_split_list(list_path: str | Path) -> list[str]:
    with Path(list_path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_tile_paths(root_dir: str | Path, tile_id: str, allow_tier3: bool = False) -> TilePaths | None:
    root_dir = Path(root_dir)
    subsets = ["train", "hold", "test"]
    if allow_tier3:
        subsets.append("tier3")

    for subset in subsets:
        base = root_dir / subset
        pre_image = base / "images" / f"{tile_id}_pre_disaster.png"
        post_image = base / "images" / f"{tile_id}_post_disaster.png"
        post_label = base / "labels" / f"{tile_id}_post_disaster.json"
        if pre_image.exists() and post_image.exists() and post_label.exists():
            if pre_image.stat().st_size <= 0 or post_image.stat().st_size <= 0 or post_label.stat().st_size <= 0:
                return None
            return TilePaths(
                source_subset=subset,
                pre_image=str(pre_image),
                post_image=str(post_image),
                post_label=str(post_label),
            )
    return None


def _build_cache_payload(config: dict[str, Any], split_name: str, list_path: str | Path) -> dict[str, Any]:
    list_path = Path(list_path)
    return {
        "dataset_version": 1,
        "split_name": split_name,
        "list_path": str(list_path.resolve()),
        "list_mtime": list_path.stat().st_mtime,
        "root_dir": str(Path(config["data"]["root_dir"]).resolve()),
        "instance_source": config["data"]["instance_source"],
        "allow_tier3": bool(config["data"]["allow_tier3"]),
        "image_size": int(config["data"]["image_size"]),
        "context_ratio": float(config["data"]["context_ratio"]),
        "min_polygon_area": float(config["data"]["min_polygon_area"]),
        "min_mask_pixels": int(config["data"]["min_mask_pixels"]),
        "max_out_of_bound_ratio": float(config["data"]["max_out_of_bound_ratio"]),
    }


def oracle_instance_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pre_image": torch.stack([item["pre_image"] for item in batch], dim=0),
        "post_image": torch.stack([item["post_image"] for item in batch], dim=0),
        "instance_mask": torch.stack([item["instance_mask"] for item in batch], dim=0),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "sample_index": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "meta": [item["meta"] for item in batch],
        "augmentation_stats": [item["augmentation_stats"] for item in batch],
    }


class XBDOracleInstanceDamageDataset(Dataset):
    def __init__(
        self,
        config: dict[str, Any],
        split_name: str,
        list_path: str | Path,
        is_train: bool = False,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.split_name = split_name
        self.list_path = str(list_path)
        self.is_train = is_train
        self.transform = build_transforms(self.config, is_train=is_train)

        instance_source_name = self.config["data"]["instance_source"]
        if instance_source_name not in INSTANCE_SOURCE_REGISTRY:
            raise ValueError(
                f"Unsupported instance_source='{instance_source_name}'. Current runnable backend: gt_json."
            )
        self.instance_source = INSTANCE_SOURCE_REGISTRY[instance_source_name]()

        payload = _build_cache_payload(self.config, split_name, list_path)
        cache_path = make_cache_path(self.config["data"]["cache_dir"], f"xbd_oracle_{split_name}", payload)
        if cache_path.exists():
            cached = load_pickle(cache_path)
            self.samples = cached["samples"]
            self.cache_path = str(cache_path)
        else:
            self.samples = self._build_samples()
            save_pickle(cache_path, {"samples": self.samples, "payload": payload})
            self.cache_path = str(cache_path)

        self.class_counts = [0 for _ in CLASS_NAMES]
        for sample in self.samples:
            self.class_counts[int(sample["label"])] += 1

    def _build_samples(self) -> list[dict[str, Any]]:
        tile_ids = _read_split_list(self.list_path)
        samples: list[dict[str, Any]] = []
        root_dir = self.config["data"]["root_dir"]
        allow_tier3 = bool(self.config["data"]["allow_tier3"])

        for tile_id in tqdm(tile_ids, desc=f"Indexing {self.split_name}", leave=False):
            tile_paths = _resolve_tile_paths(root_dir, tile_id, allow_tier3=allow_tier3)
            if tile_paths is None:
                continue
            tile_samples = self.instance_source.build_tile_samples(
                tile_id=tile_id,
                split_name=self.split_name,
                tile_paths=tile_paths,
                config=self.config,
            )
            samples.extend(tile_samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre_image = Image.open(sample["pre_image"]).convert("RGB")
        post_image = Image.open(sample["post_image"]).convert("RGB")

        x1, y1, x2, y2 = sample["crop_bbox_xyxy"]
        pre_crop = pre_image.crop((x1, y1, x2, y2))
        post_crop = post_image.crop((x1, y1, x2, y2))

        crop_w = x2 - x1
        crop_h = y2 - y1
        mask_np = polygon_to_mask(sample["polygon_xy"], crop_h, crop_w, offset=(x1, y1))
        mask = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

        if self.is_train:
            transformed = self.transform(pre_crop, post_crop, mask, return_stats=True)  # type: ignore[misc]
            pre_tensor, post_tensor, mask_tensor, augmentation_stats = transformed
        else:
            pre_tensor, post_tensor, mask_tensor = self.transform(pre_crop, post_crop, mask)
            augmentation_stats = {}
        if mask_tensor.sum().item() <= 0:
            mask_tensor = torch.ones_like(mask_tensor)

        meta = {
            "tile_id": sample["tile_id"],
            "building_idx": int(sample["building_idx"]),
            "uid": sample.get("uid"),
            "original_subtype": sample["original_subtype"],
            "polygon_area": float(sample["polygon_area"]),
            "bbox_xyxy": list(sample["bbox_xyxy"]),
            "crop_bbox_xyxy": list(sample["crop_bbox_xyxy"]),
            "source_subset": sample["source_subset"],
            "split": sample["split"],
            "cache_path": self.cache_path,
        }

        sample_dict = {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "instance_mask": mask_tensor,
            "label": int(sample["label"]),
            "sample_index": int(index),
            "meta": meta,
            "augmentation_stats": augmentation_stats,
        }
        return sample_dict
