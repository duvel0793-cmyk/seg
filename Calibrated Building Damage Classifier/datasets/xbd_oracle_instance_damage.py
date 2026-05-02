from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.transforms import SCALE_NAMES, build_transforms
from utils.cache import make_cache_path, load_pickle, save_pickle
from utils.geometry import (
    clip_bbox_to_image,
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


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 3:
        bbox = polygon_bbox(points)
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        cross = (x0 * y1) - (x1 * y0)
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(signed_area) <= 1e-6:
        bbox = polygon_bbox(points)
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
    signed_area *= 0.5
    return (cx / (6.0 * signed_area), cy / (6.0 * signed_area))


def _scale_bbox(
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max((x2 - x1) * 0.5 * scale, 0.5)
    half_h = max((y2 - y1) * 0.5 * scale, 0.5)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _square_bbox_from_center(
    center_xy: tuple[float, float],
    side_length: float,
) -> tuple[float, float, float, float]:
    side = max(float(side_length), 1.0)
    half = side * 0.5
    cx, cy = center_xy
    return (cx - half, cy - half, cx + half, cy + half)


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
        data_cfg = config["dataset"]
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

            tight_bbox_raw = polygon_bbox(polygon_xy)
            tight_bbox_float = _scale_bbox(tight_bbox_raw, scale=1.0 + float(data_cfg["tight_padding"]))
            context_bbox_float = _scale_bbox(tight_bbox_raw, scale=float(data_cfg["context_scale"]))
            centroid_xy = _polygon_centroid(polygon_xy)
            neighborhood_bbox_float = _square_bbox_from_center(
                centroid_xy,
                float(data_cfg["neighborhood_crop_size"]),
            )

            if out_of_bounds_fraction(context_bbox_float, image_width, image_height) > float(data_cfg["max_out_of_bound_ratio"]):
                continue
            if out_of_bounds_fraction(neighborhood_bbox_float, image_width, image_height) > float(data_cfg["max_out_of_bound_ratio"]):
                continue

            tight_bbox = clip_bbox_to_image(tight_bbox_float, image_width, image_height)
            context_bbox = clip_bbox_to_image(context_bbox_float, image_width, image_height)
            neighborhood_bbox = clip_bbox_to_image(neighborhood_bbox_float, image_width, image_height)

            tight_w = tight_bbox[2] - tight_bbox[0]
            tight_h = tight_bbox[3] - tight_bbox[1]
            if tight_w <= 1 or tight_h <= 1:
                continue

            tight_mask = polygon_to_mask(polygon_xy, tight_h, tight_w, offset=(tight_bbox[0], tight_bbox[1]))
            mask_pixels = int(tight_mask.sum())
            if mask_pixels < int(data_cfg["min_mask_pixels"]):
                continue

            samples.append(
                {
                    "tile_id": tile_id,
                    "building_idx": int(building_idx),
                    "uid": properties.get("uid"),
                    "label": LABEL_TO_INDEX[subtype],
                    "original_subtype": subtype,
                    "polygon_xy": [(float(x), float(y)) for x, y in polygon_xy],
                    "polygon_area": float(polygon_area(polygon_xy)),
                    "bbox_xyxy": [float(v) for v in tight_bbox_raw],
                    "tight_bbox_xyxy": [int(v) for v in tight_bbox],
                    "context_bbox_xyxy": [int(v) for v in context_bbox],
                    "neighborhood_bbox_xyxy": [int(v) for v in neighborhood_bbox],
                    "mask_pixels": mask_pixels,
                    "centroid_xy": [float(centroid_xy[0]), float(centroid_xy[1])],
                    "split": split_name,
                    "source_subset": tile_paths.source_subset,
                    "pre_image": tile_paths.pre_image,
                    "post_image": tile_paths.post_image,
                    "post_label": tile_paths.post_label,
                }
            )
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
    dataset_cfg = config["dataset"]
    return {
        "dataset_version": 5,
        "split_name": split_name,
        "list_path": str(list_path.resolve()),
        "list_mtime": list_path.stat().st_mtime,
        "root_dir": str(Path(dataset_cfg["root_dir"]).resolve()),
        "instance_source": dataset_cfg["instance_source"],
        "allow_tier3": bool(dataset_cfg["allow_tier3"]),
        "use_multi_context": bool(dataset_cfg["use_multi_context"]),
        "image_size_tight": int(dataset_cfg["image_size_tight"]),
        "image_size_context": int(dataset_cfg["image_size_context"]),
        "image_size_neighborhood": int(dataset_cfg["image_size_neighborhood"]),
        "tight_padding": float(dataset_cfg["tight_padding"]),
        "context_scale": float(dataset_cfg["context_scale"]),
        "neighborhood_crop_size": int(dataset_cfg["neighborhood_crop_size"]),
        "min_polygon_area": float(dataset_cfg["min_polygon_area"]),
        "min_mask_pixels": int(dataset_cfg["min_mask_pixels"]),
        "max_out_of_bound_ratio": float(dataset_cfg["max_out_of_bound_ratio"]),
    }


def oracle_instance_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "sample_index": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "meta": [item["meta"] for item in batch],
    }
    for scale_name in SCALE_NAMES:
        collated[f"pre_{scale_name}"] = torch.stack([item[f"pre_{scale_name}"] for item in batch], dim=0)
        collated[f"post_{scale_name}"] = torch.stack([item[f"post_{scale_name}"] for item in batch], dim=0)
        collated[f"mask_{scale_name}"] = torch.stack([item[f"mask_{scale_name}"] for item in batch], dim=0)
    if "augmentation_stats" in batch[0]:
        collated["augmentation_stats"] = [item["augmentation_stats"] for item in batch]
    return collated


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
        self.split_name = str(split_name)
        self.list_path = str(list_path)
        self.is_train = bool(is_train)
        self.transform = build_transforms(self.config, is_train=is_train)

        instance_source_name = self.config["dataset"]["instance_source"]
        if instance_source_name not in INSTANCE_SOURCE_REGISTRY:
            raise ValueError(f"Unsupported instance_source='{instance_source_name}'.")
        self.instance_source = INSTANCE_SOURCE_REGISTRY[instance_source_name]()

        payload = _build_cache_payload(self.config, self.split_name, list_path)
        cache_path = make_cache_path(self.config["dataset"]["cache_dir"], f"xbd_oracle_{self.split_name}", payload)
        if cache_path.exists():
            cached = load_pickle(cache_path)
            self.samples = cached["samples"]
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
        root_dir = self.config["dataset"]["root_dir"]
        allow_tier3 = bool(self.config["dataset"]["allow_tier3"])

        for tile_id in tqdm(tile_ids, desc=f"Indexing {self.split_name}", leave=False):
            tile_paths = _resolve_tile_paths(root_dir, tile_id, allow_tier3=allow_tier3)
            if tile_paths is None:
                continue
            samples.extend(
                self.instance_source.build_tile_samples(
                    tile_id=tile_id,
                    split_name=self.split_name,
                    tile_paths=tile_paths,
                    config=self.config,
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _build_crop_mask(
        self,
        polygon_xy: list[tuple[float, float]],
        crop_bbox_xyxy: list[int],
    ) -> Image.Image:
        x1, y1, x2, y2 = crop_bbox_xyxy
        crop_w = x2 - x1
        crop_h = y2 - y1
        mask_np = polygon_to_mask(polygon_xy, crop_h, crop_w, offset=(x1, y1))
        return Image.fromarray((mask_np * 255).astype("uint8"), mode="L")

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre_image = Image.open(sample["pre_image"]).convert("RGB")
        post_image = Image.open(sample["post_image"]).convert("RGB")

        raw_crops: dict[str, dict[str, Image.Image]] = {}
        for scale_name in SCALE_NAMES:
            bbox_key = f"{scale_name}_bbox_xyxy"
            x1, y1, x2, y2 = sample[bbox_key]
            raw_crops[scale_name] = {
                "pre": pre_image.crop((x1, y1, x2, y2)),
                "post": post_image.crop((x1, y1, x2, y2)),
                "mask": self._build_crop_mask(sample["polygon_xy"], sample[bbox_key]),
            }

        prepared_crops, augmentation_stats = self.transform.prepare_sample(raw_crops)  # type: ignore[attr-defined]
        tensor_dict = self.transform.to_tensor_dict(prepared_crops)

        meta = {
            "tile_id": sample["tile_id"],
            "building_idx": int(sample["building_idx"]),
            "uid": sample.get("uid"),
            "original_subtype": sample["original_subtype"],
            "polygon_xy": [(float(x), float(y)) for x, y in sample["polygon_xy"]],
            "polygon_area": float(sample["polygon_area"]),
            "bbox_xyxy": list(sample["bbox_xyxy"]),
            "tight_bbox_xyxy": list(sample["tight_bbox_xyxy"]),
            "context_bbox_xyxy": list(sample["context_bbox_xyxy"]),
            "neighborhood_bbox_xyxy": list(sample["neighborhood_bbox_xyxy"]),
            "centroid_xy": list(sample["centroid_xy"]),
            "source_subset": sample["source_subset"],
            "split": sample["split"],
            "cache_path": self.cache_path,
        }

        return {
            **tensor_dict,
            "label": int(sample["label"]),
            "sample_index": int(index),
            "meta": meta,
            "augmentation_stats": augmentation_stats,
        }
