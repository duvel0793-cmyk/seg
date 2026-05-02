from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.transforms import build_transforms
from utils.misc import (
    CLASS_NAMES,
    LABEL_TO_INDEX,
    SCALE_NAMES,
    clip_bbox_to_image,
    get_enabled_scale_names,
    infer_disaster_name,
    is_valid_polygon,
    load_label_png,
    out_of_bounds_fraction,
    parse_wkt_polygon,
    polygon_area,
    polygon_bbox,
    polygon_to_mask,
    read_json,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TilePaths:
    subset: str
    pre_image: str
    post_image: str
    post_label: str
    post_target: str | None


def _resolve_target_png(targets_dir: Path, tile_id: str) -> Path | None:
    candidates = [
        targets_dir / f"{tile_id}_post_disaster_target.png",
        targets_dir / f"{tile_id}_damage_target.png",
        targets_dir / f"{tile_id}_post_disaster_damage_target.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_tile_paths(root_dir: str | Path, tile_id: str, allow_tier3: bool = False) -> TilePaths | None:
    subsets = ["train", "hold", "test"] + (["tier3"] if allow_tier3 else [])
    root_dir = Path(root_dir)
    for subset in subsets:
        base = root_dir / subset
        pre = base / "images" / f"{tile_id}_pre_disaster.png"
        post = base / "images" / f"{tile_id}_post_disaster.png"
        label = base / "labels" / f"{tile_id}_post_disaster.json"
        if pre.exists() and post.exists() and label.exists():
            post_target = _resolve_target_png(base / "targets", tile_id)
            return TilePaths(
                subset=subset,
                pre_image=str(pre),
                post_image=str(post),
                post_label=str(label),
                post_target=None if post_target is None else str(post_target),
            )
    return None


def _read_split_list(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _scale_bbox(bbox: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max((x2 - x1) * 0.5 * float(factor), 0.5)
    half_h = max((y2 - y1) * 0.5 * float(factor), 0.5)
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


class XBDInstanceDataset(Dataset):
    def __init__(self, *, config: dict[str, Any], split: str, list_path: str | Path, is_train: bool) -> None:
        super().__init__()
        self.config = config
        self.split = str(split)
        self.list_path = str(list_path)
        self.is_train = bool(is_train)
        self.scale_names = get_enabled_scale_names(config)
        self.transform = build_transforms(config, is_train=is_train)
        self.samples = self._build_samples()
        debug_max_samples = int(config["dataset"].get("debug_max_samples", 0) or 0)
        if debug_max_samples > 0:
            self.samples = self.samples[:debug_max_samples]
        self.class_counts = [0 for _ in CLASS_NAMES]
        for sample in self.samples:
            self.class_counts[int(sample["label"])] += 1

    def _build_samples(self) -> list[dict[str, Any]]:
        root = self.config["data"]["root"]
        allow_tier3 = bool(self.config["dataset"].get("allow_tier3", False))
        min_polygon_area = float(self.config["dataset"].get("min_polygon_area", 16.0))
        min_mask_pixels = int(self.config["dataset"].get("min_mask_pixels", 16))
        max_oob = float(self.config["dataset"].get("max_out_of_bound_ratio", 0.4))
        scale_cfg = self.config["dataset"]["crop_scales"]
        samples: list[dict[str, Any]] = []
        tile_ids = _read_split_list(self.list_path)
        debug_subset = int(self.config["dataset"].get("debug_subset", 0) or 0)
        if debug_subset > 0:
            tile_ids = tile_ids[:debug_subset]
        for tile_id in tqdm(tile_ids, desc=f"Indexing {self.split}", leave=False):
            tile_paths = _resolve_tile_paths(root, tile_id, allow_tier3=allow_tier3)
            if tile_paths is None:
                continue
            payload = read_json(tile_paths.post_label)
            metadata = payload.get("metadata", {})
            image_width = int(metadata.get("width", metadata.get("original_width", 1024)))
            image_height = int(metadata.get("height", metadata.get("original_height", 1024)))
            features = payload.get("features", {}).get("xy", [])
            disaster_name = infer_disaster_name(tile_id, payload)
            for building_idx, feature in enumerate(features):
                props = feature.get("properties", {})
                if props.get("feature_type") != "building":
                    continue
                subtype = props.get("subtype")
                if subtype not in LABEL_TO_INDEX:
                    continue
                try:
                    polygon = parse_wkt_polygon(feature.get("wkt", ""))
                except Exception:
                    continue
                if not is_valid_polygon(polygon, min_area=min_polygon_area):
                    continue
                tight_bbox_raw = polygon_bbox(polygon)
                scale_bboxes: dict[str, list[int] | None] = {scale_name: None for scale_name in SCALE_NAMES}
                skip_sample = False
                for scale_name in self.scale_names:
                    scaled_bbox_float = _scale_bbox(tight_bbox_raw, float(scale_cfg[scale_name]["context_factor"]))
                    if scale_name != "tight" and out_of_bounds_fraction(scaled_bbox_float, image_width, image_height) > max_oob:
                        skip_sample = True
                        break
                    scale_bboxes[scale_name] = [int(v) for v in clip_bbox_to_image(scaled_bbox_float, image_width, image_height)]
                if skip_sample:
                    continue
                tight_bbox = tuple(scale_bboxes["tight"] or clip_bbox_to_image(tight_bbox_raw, image_width, image_height))
                tight_mask = polygon_to_mask(
                    polygon,
                    tight_bbox[3] - tight_bbox[1],
                    tight_bbox[2] - tight_bbox[0],
                    offset=(tight_bbox[0], tight_bbox[1]),
                )
                if int(tight_mask.sum()) < min_mask_pixels:
                    continue
                samples.append(
                    {
                        "tile_id": tile_id,
                        "building_idx": int(building_idx),
                        "label": int(LABEL_TO_INDEX[subtype]),
                        "original_subtype": subtype,
                        "polygon_xy": [(float(x), float(y)) for x, y in polygon],
                        "bbox_xyxy": [float(v) for v in tight_bbox_raw],
                        "tight_bbox_xyxy": scale_bboxes["tight"],
                        "context_bbox_xyxy": scale_bboxes["context"],
                        "neighborhood_bbox_xyxy": scale_bboxes["neighborhood"],
                        "polygon_area": float(polygon_area(polygon)),
                        "pre_image": tile_paths.pre_image,
                        "post_image": tile_paths.post_image,
                        "post_label": tile_paths.post_label,
                        "post_target": tile_paths.post_target,
                        "source_subset": tile_paths.subset,
                        "disaster_name": disaster_name,
                        "image_size": [image_height, image_width],
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_mask(self, polygon: list[tuple[float, float]], bbox: list[int]) -> Image.Image:
        x1, y1, x2, y2 = bbox
        mask = polygon_to_mask(polygon, y2 - y1, x2 - x1, offset=(x1, y1))
        return Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre_image = Image.open(sample["pre_image"]).convert("RGB")
        post_image = Image.open(sample["post_image"]).convert("RGB")
        post_target = load_label_png(sample["post_target"]) if sample["post_target"] and bool(self.config["data"].get("use_pixel_targets", True)) else None

        crops: dict[str, dict[str, Image.Image]] = {}
        for scale_name in self.scale_names:
            bbox = sample[f"{scale_name}_bbox_xyxy"]
            x1, y1, x2, y2 = bbox
            entry = {
                "pre": pre_image.crop((x1, y1, x2, y2)),
                "post": post_image.crop((x1, y1, x2, y2)),
                "mask": self._crop_mask(sample["polygon_xy"], bbox),
            }
            if post_target is not None:
                entry["post_target"] = Image.fromarray(post_target[y1:y2, x1:x2], mode="L")
            crops[scale_name] = entry

        prepared, aug_stats = self.transform.prepare_sample(crops)
        tensors = self.transform.to_tensor_dict(prepared, ignore_index=int(self.config["data"].get("ignore_index", 255)))
        meta = {
            "disaster_name": sample["disaster_name"],
            "image_id": sample["tile_id"],
            "tile_id": sample["tile_id"],
            "pre_image_path": sample["pre_image"],
            "post_image_path": sample["post_image"],
            "label_path": sample["post_label"],
            "target_polygon": sample["polygon_xy"],
            "bbox": sample["bbox_xyxy"],
            "tight_bbox_xyxy": sample["tight_bbox_xyxy"],
            "context_bbox_xyxy": sample["context_bbox_xyxy"],
            "neighborhood_bbox_xyxy": sample["neighborhood_bbox_xyxy"],
            "enabled_scales": list(self.scale_names),
            "original_image_size": sample["image_size"],
            "building_idx": sample["building_idx"],
            "source_subset": sample["source_subset"],
        }
        return {
            **tensors,
            "label": int(sample["label"]),
            "sample_index": int(index),
            "meta": meta,
            "augmentation_stats": aug_stats,
        }
