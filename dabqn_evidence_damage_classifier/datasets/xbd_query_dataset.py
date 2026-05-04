from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any
import warnings

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from datasets.polygon_utils import (
    clip_polygon_to_box,
    infer_disaster_name,
    is_valid_polygon,
    load_label_png,
    parse_wkt_polygon,
    polygon_area,
    polygon_bbox,
    polygon_to_mask,
    translate_polygon,
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


@dataclass
class PatchSelection:
    patch_box: tuple[int, int, int, int]
    patch_size: int
    dense_retry_count: int
    adaptive_shrink_count: int
    use_limit_fallback: bool
    requested_empty: bool
    estimated_instance_count: int


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


def _boxes_intersect(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def _clamp_patch_origin(origin: float, full_size: int, patch_size: int) -> int:
    if full_size <= patch_size:
        return 0
    max_origin = full_size - patch_size
    return int(min(max(round(origin), 0), max_origin))


def _compute_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    stride = max(int(stride), 1)
    last_origin = length - window
    positions = list(range(0, last_origin + 1, stride))
    if positions[-1] != last_origin:
        positions.append(last_origin)
    return positions


class XBDQueryDataset(Dataset):
    def __init__(self, *, config: dict[str, Any], split: str, is_train: bool) -> None:
        super().__init__()
        self.config = config
        self.split = str(split)
        self.is_train = bool(is_train)
        self.root_dir = str(config["data"]["root"])
        data_cfg = config.get("data", {})
        dataset_cfg = config.get("dataset", {})
        eval_cfg = config.get("evaluation", config.get("eval", {}))

        def cfg_value(name: str, default: Any) -> Any:
            if name in data_cfg:
                return data_cfg[name]
            if name in dataset_cfg:
                return dataset_cfg[name]
            return default

        self.allow_tier3 = bool(cfg_value("allow_tier3", False))
        self.min_polygon_area = float(cfg_value("min_polygon_area", 16.0))
        self.min_mask_pixels = int(cfg_value("min_mask_pixels", 8))
        self.min_area_after_clip = float(cfg_value("min_area_after_clip", 4.0))
        self.image_size = int(cfg_value("image_size", 512))
        self.patch_size = int(cfg_value("patch_size", self.image_size))
        raw_adaptive_sizes = list(cfg_value("adaptive_patch_sizes", [self.patch_size, 384, 256]))
        normalized_sizes = [int(size) for size in raw_adaptive_sizes if int(size) > 0]
        if self.patch_size not in normalized_sizes:
            normalized_sizes.insert(0, self.patch_size)
        self.adaptive_patch_sizes: list[int] = []
        for size in normalized_sizes:
            if size not in self.adaptive_patch_sizes:
                self.adaptive_patch_sizes.append(size)
        self.train_mode = str(cfg_value("train_mode", "random_patch"))
        self.max_instances_per_patch = int(cfg_value("max_instances_per_patch", 120))
        self.min_instances_per_patch = int(cfg_value("min_instances_per_patch", 1))
        self.allow_empty_patch = bool(cfg_value("allow_empty_patch", True))
        self.empty_patch_ratio = float(cfg_value("empty_patch_ratio", 0.1))
        self.dense_patch_retry = int(cfg_value("dense_patch_retry", 30))
        self.dense_shrink_enabled = bool(cfg_value("dense_shrink_enabled", True))
        self.dense_fallback_policy = str(cfg_value("dense_fallback_policy", "resample_shrink_then_center_keep"))
        self.fallback_keep_policy = str(cfg_value("fallback_keep_policy", "center_priority"))
        self.grid_stride = int(cfg_value("grid_stride", self.patch_size))
        self.val_mode = str(eval_cfg.get("val_mode", "sliding_window"))
        self.val_patch_size = int(eval_cfg.get("patch_size", self.patch_size))
        self.val_stride = int(eval_cfg.get("stride", min(max(self.val_patch_size - 128, 1), self.val_patch_size)))
        self.num_queries = int(config.get("model", {}).get("num_queries", 150))
        self.transforms = build_transforms(config, is_train=is_train)
        self._fallback_keep_warning_count = 0

        split_key = f"{self.split}_list"
        if split_key not in config["data"]:
            raise KeyError(f"Missing data.{split_key} in config.")
        self.tile_ids = _read_split_list(config["data"][split_key])
        debug_subset = int(cfg_value("debug_subset", 0) or 0)
        if debug_subset > 0:
            self.tile_ids = self.tile_ids[:debug_subset]
        self.samples = self._index_tiles()
        self.grid_index: list[tuple[int, tuple[int, int, int, int]]] = []
        if self.is_train and self.train_mode == "grid_patch":
            self.grid_index = self._build_grid_index(stride=self.grid_stride)

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
                bbox = polygon_bbox(polygon)
                instances.append(
                    {
                        "building_idx": int(building_idx),
                        "label": int(LABEL_TO_INDEX[subtype]),
                        "subtype": str(subtype),
                        "polygon_xy": [(float(x), float(y)) for x, y in polygon],
                        "bbox_xyxy": [float(v) for v in bbox],
                        "polygon_area": float(polygon_area(polygon)),
                        "center_xy": (float((bbox[0] + bbox[2]) * 0.5), float((bbox[1] + bbox[3]) * 0.5)),
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

    def _build_grid_index(self, *, stride: int) -> list[tuple[int, tuple[int, int, int, int]]]:
        grid_index: list[tuple[int, tuple[int, int, int, int]]] = []
        for sample_index, sample in enumerate(self.samples):
            for patch_box in self.get_sliding_window_boxes(sample_index, patch_size=self.patch_size, stride=stride):
                grid_index.append((sample_index, patch_box))
        return grid_index

    def __len__(self) -> int:
        if self.is_train and self.train_mode == "grid_patch":
            return len(self.grid_index)
        return len(self.samples)

    def _sample_to_region_box(
        self,
        *,
        width: int,
        height: int,
        center_x: float | None = None,
        center_y: float | None = None,
        patch_size: int,
        jitter_radius: float = 0.0,
    ) -> tuple[int, int, int, int]:
        if center_x is None or center_y is None:
            max_x = max(width - patch_size, 0)
            max_y = max(height - patch_size, 0)
            origin_x = 0 if max_x == 0 else random.randint(0, max_x)
            origin_y = 0 if max_y == 0 else random.randint(0, max_y)
        else:
            origin_x = _clamp_patch_origin(center_x - (patch_size * 0.5) + random.uniform(-jitter_radius, jitter_radius), width, patch_size)
            origin_y = _clamp_patch_origin(center_y - (patch_size * 0.5) + random.uniform(-jitter_radius, jitter_radius), height, patch_size)
        return (origin_x, origin_y, origin_x + patch_size, origin_y + patch_size)

    def _count_intersecting_instances(self, sample: dict[str, Any], patch_box: tuple[int, int, int, int]) -> int:
        patch_tuple = tuple(float(v) for v in patch_box)
        count = 0
        for instance in sample["instances"]:
            if _boxes_intersect(tuple(instance["bbox_xyxy"]), patch_tuple):
                count += 1
        return count

    def _candidate_quality(self, count: int) -> tuple[float, float, float]:
        overflow = max(count - self.max_instances_per_patch, 0)
        underflow = max(self.min_instances_per_patch - count, 0)
        valid = 0.0 if overflow == 0 and underflow == 0 else 1.0
        desired = float(min(self.max_instances_per_patch, max(self.min_instances_per_patch, int(self.max_instances_per_patch * 0.5))))
        return (valid, float(overflow + underflow), abs(float(count) - desired))

    def _iter_patch_sizes(self) -> list[int]:
        if not self.dense_shrink_enabled:
            return [self.patch_size]
        return list(self.adaptive_patch_sizes)

    def _select_empty_patch(self, sample_index: int) -> PatchSelection:
        sample = self.samples[sample_index]
        width = int(sample["width"])
        height = int(sample["height"])
        attempts = 0
        best: tuple[tuple[float, float, float], PatchSelection] | None = None
        for size_index, patch_size in enumerate(self._iter_patch_sizes()):
            for _ in range(max(self.dense_patch_retry, 1)):
                attempts += 1
                patch_box = self._sample_to_region_box(width=width, height=height, patch_size=patch_size)
                count = self._count_intersecting_instances(sample, patch_box)
                selection = PatchSelection(
                    patch_box=patch_box,
                    patch_size=patch_size,
                    dense_retry_count=attempts - 1,
                    adaptive_shrink_count=size_index,
                    use_limit_fallback=False,
                    requested_empty=True,
                    estimated_instance_count=count,
                )
                if count == 0:
                    return selection
                quality = (0.0, float(count), float(size_index))
                if best is None or quality < best[0]:
                    best = (quality, selection)
        if best is None:
            patch_box = self._sample_to_region_box(width=width, height=height, patch_size=self.patch_size)
            count = self._count_intersecting_instances(sample, patch_box)
            return PatchSelection(
                patch_box=patch_box,
                patch_size=self.patch_size,
                dense_retry_count=max(attempts - 1, 0),
                adaptive_shrink_count=0,
                use_limit_fallback=False,
                requested_empty=True,
                estimated_instance_count=count,
            )
        return best[1]

    def _select_positive_patch(self, sample_index: int) -> PatchSelection:
        sample = self.samples[sample_index]
        width = int(sample["width"])
        height = int(sample["height"])
        instances = sample["instances"]
        if not instances:
            patch_box = self._sample_to_region_box(width=width, height=height, patch_size=self.patch_size)
            return PatchSelection(
                patch_box=patch_box,
                patch_size=self.patch_size,
                dense_retry_count=0,
                adaptive_shrink_count=0,
                use_limit_fallback=False,
                requested_empty=False,
                estimated_instance_count=0,
            )

        attempts = 0
        best: tuple[tuple[float, float, float], PatchSelection] | None = None
        for size_index, patch_size in enumerate(self._iter_patch_sizes()):
            for retry in range(max(self.dense_patch_retry, 1)):
                attempts += 1
                reference = random.choice(instances)
                center_x, center_y = reference["center_xy"]
                shrink_ratio = max(0.1, 1.0 - (float(retry) / max(float(self.dense_patch_retry), 1.0)))
                jitter = 0.5 * patch_size * shrink_ratio
                patch_box = self._sample_to_region_box(
                    width=width,
                    height=height,
                    center_x=center_x,
                    center_y=center_y,
                    patch_size=patch_size,
                    jitter_radius=jitter,
                )
                count = self._count_intersecting_instances(sample, patch_box)
                selection = PatchSelection(
                    patch_box=patch_box,
                    patch_size=patch_size,
                    dense_retry_count=attempts - 1,
                    adaptive_shrink_count=size_index,
                    use_limit_fallback=False,
                    requested_empty=False,
                    estimated_instance_count=count,
                )
                if count >= self.min_instances_per_patch and count <= self.max_instances_per_patch:
                    return selection
                quality = self._candidate_quality(count) + (float(size_index),)
                if best is None or quality < best[0]:
                    best = (quality, selection)

                if retry >= (self.dense_patch_retry // 2):
                    random_patch_box = self._sample_to_region_box(width=width, height=height, patch_size=patch_size)
                    random_count = self._count_intersecting_instances(sample, random_patch_box)
                    random_selection = PatchSelection(
                        patch_box=random_patch_box,
                        patch_size=patch_size,
                        dense_retry_count=attempts - 1,
                        adaptive_shrink_count=size_index,
                        use_limit_fallback=False,
                        requested_empty=False,
                        estimated_instance_count=random_count,
                    )
                    if random_count >= self.min_instances_per_patch and random_count <= self.max_instances_per_patch:
                        return random_selection
                    random_quality = self._candidate_quality(random_count) + (float(size_index),)
                    if best is None or random_quality < best[0]:
                        best = (random_quality, random_selection)

        if best is None:
            patch_box = self._sample_to_region_box(width=width, height=height, patch_size=self.patch_size)
            count = self._count_intersecting_instances(sample, patch_box)
            return PatchSelection(
                patch_box=patch_box,
                patch_size=self.patch_size,
                dense_retry_count=max(attempts - 1, 0),
                adaptive_shrink_count=0,
                use_limit_fallback=count > self.max_instances_per_patch,
                requested_empty=False,
                estimated_instance_count=count,
            )

        chosen = best[1]
        return PatchSelection(
            patch_box=chosen.patch_box,
            patch_size=chosen.patch_size,
            dense_retry_count=chosen.dense_retry_count,
            adaptive_shrink_count=chosen.adaptive_shrink_count,
            use_limit_fallback=chosen.estimated_instance_count > self.max_instances_per_patch,
            requested_empty=False,
            estimated_instance_count=chosen.estimated_instance_count,
        )

    def _select_random_patch(self, sample_index: int) -> PatchSelection:
        request_empty = self.allow_empty_patch and random.random() < self.empty_patch_ratio
        if request_empty:
            return self._select_empty_patch(sample_index)
        return self._select_positive_patch(sample_index)

    def _collect_region_instances(self, sample_index: int, patch_box: tuple[int, int, int, int]) -> list[dict[str, Any]]:
        sample = self.samples[sample_index]
        x1, y1, x2, y2 = [int(value) for value in patch_box]
        patch_width = int(x2 - x1)
        patch_height = int(y2 - y1)
        patch_tuple = tuple(float(v) for v in patch_box)

        patch_instances: list[dict[str, Any]] = []
        for instance in sample["instances"]:
            if not _boxes_intersect(tuple(instance["bbox_xyxy"]), patch_tuple):
                continue
            clipped_polygon = clip_polygon_to_box(instance["polygon_xy"], patch_tuple)
            if len(clipped_polygon) < 3:
                continue
            if float(polygon_area(clipped_polygon)) < self.min_area_after_clip:
                continue
            if not is_valid_polygon(clipped_polygon, min_area=self.min_area_after_clip):
                continue
            local_polygon = translate_polygon(clipped_polygon, x1, y1)
            mask = polygon_to_mask(local_polygon, patch_height, patch_width)
            mask_pixels = int(mask.sum())
            if mask_pixels < self.min_mask_pixels:
                continue
            local_box = _mask_to_box_xyxy(mask)
            if local_box is None:
                continue
            patch_instances.append(
                {
                    "building_idx": int(instance["building_idx"]),
                    "label": int(instance["label"]),
                    "subtype": str(instance["subtype"]),
                    "polygon_xy": local_polygon,
                    "bbox_xyxy": local_box,
                    "polygon_area": float(polygon_area(local_polygon)),
                    "mask": mask.astype(np.uint8),
                    "mask_pixels": mask_pixels,
                }
            )
        return patch_instances

    def _crop_damage_map(
        self,
        sample: dict[str, Any],
        patch_box: tuple[int, int, int, int],
        patch_instances: list[dict[str, Any]],
    ) -> np.ndarray | None:
        paths: TilePaths = sample["paths"]
        x1, y1, x2, y2 = [int(value) for value in patch_box]
        patch_width = int(x2 - x1)
        patch_height = int(y2 - y1)

        if paths.post_target is not None:
            damage_image = Image.open(paths.post_target)
            damage_crop = damage_image.crop((x1, y1, x2, y2))
            return np.asarray(damage_crop, dtype=np.uint8)

        if not patch_instances:
            return np.zeros((patch_height, patch_width), dtype=np.uint8)

        damage_map = np.zeros((patch_height, patch_width), dtype=np.uint8)
        for instance in patch_instances:
            damage_map[instance["mask"] > 0] = np.maximum(
                damage_map[instance["mask"] > 0],
                np.uint8(int(instance["label"]) + 1),
            )
        return damage_map

    def _apply_fallback_keep(
        self,
        patch_instances: list[dict[str, Any]],
        *,
        patch_width: int,
        patch_height: int,
    ) -> list[dict[str, Any]]:
        if self.fallback_keep_policy != "center_priority":
            raise ValueError(f"Unsupported fallback_keep_policy='{self.fallback_keep_policy}'.")

        patch_center_x = float(patch_width) * 0.5
        patch_center_y = float(patch_height) * 0.5
        prioritized: list[tuple[float, float, int, dict[str, Any]]] = []
        for instance in patch_instances:
            area = float(instance["polygon_area"])
            if area < self.min_area_after_clip:
                continue
            box = instance["bbox_xyxy"]
            center_x = float((box[0] + box[2]) * 0.5)
            center_y = float((box[1] + box[3]) * 0.5)
            distance = ((center_x - patch_center_x) ** 2) + ((center_y - patch_center_y) ** 2)
            prioritized.append((distance, -float(instance["mask_pixels"]), int(instance["building_idx"]), instance))

        if len(prioritized) < self.max_instances_per_patch:
            seen_ids = {item[3]["building_idx"] for item in prioritized}
            for instance in patch_instances:
                if int(instance["building_idx"]) in seen_ids:
                    continue
                box = instance["bbox_xyxy"]
                center_x = float((box[0] + box[2]) * 0.5)
                center_y = float((box[1] + box[3]) * 0.5)
                distance = ((center_x - patch_center_x) ** 2) + ((center_y - patch_center_y) ** 2)
                prioritized.append((distance, -float(instance["mask_pixels"]), int(instance["building_idx"]), instance))

        prioritized.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in prioritized[: self.max_instances_per_patch]]

    def _tensorize_region(
        self,
        *,
        sample_index: int,
        patch_box: tuple[int, int, int, int],
        patch_source: str,
        dense_retry_count: int,
        adaptive_shrink_count: int,
        allow_limit_fallback: bool,
        force_image_id: str | None = None,
        requested_empty: bool = False,
        estimated_instance_count: int | None = None,
    ) -> dict[str, Any]:
        sample = self.samples[sample_index]
        paths: TilePaths = sample["paths"]
        x1, y1, x2, y2 = [int(value) for value in patch_box]
        patch_width = int(x2 - x1)
        patch_height = int(y2 - y1)

        pre_image = Image.open(paths.pre_image).convert("RGB").crop((x1, y1, x2, y2))
        post_image = Image.open(paths.post_image).convert("RGB").crop((x1, y1, x2, y2))
        patch_instances = self._collect_region_instances(sample_index, patch_box)
        num_instances_before_limit = len(patch_instances)
        gt_over_patch_limit = num_instances_before_limit > self.max_instances_per_patch
        fallback_keep_applied = False

        if allow_limit_fallback and gt_over_patch_limit:
            patch_instances = self._apply_fallback_keep(patch_instances, patch_width=patch_width, patch_height=patch_height)
            fallback_keep_applied = True
            self._fallback_keep_warning_count += 1
            warnings.warn(
                f"Patch fallback keep on {sample['tile_id']} patch={patch_box}: "
                f"{num_instances_before_limit} instances -> {len(patch_instances)}",
                stacklevel=2,
            )

        damage_map = self._crop_damage_map(sample, patch_box, patch_instances)
        transformed = self.transforms(
            pre_image=pre_image,
            post_image=post_image,
            instance_masks=[instance["mask"] for instance in patch_instances],
            damage_map=damage_map,
        )

        resized_masks = transformed["instance_masks"]
        kept_masks: list[np.ndarray] = []
        kept_labels: list[int] = []
        kept_ids: list[int] = []
        kept_boxes: list[list[float]] = []
        kept_boxes_norm: list[list[float]] = []
        for mask, instance in zip(resized_masks, patch_instances):
            if int(mask.sum()) < self.min_mask_pixels:
                continue
            box = _mask_to_box_xyxy(mask)
            if box is None:
                continue
            kept_masks.append(mask.astype(np.uint8))
            kept_labels.append(int(instance["label"]))
            kept_ids.append(int(instance["building_idx"]))
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

        patch_id = force_image_id or (
            str(sample["tile_id"])
            if patch_source == "full_image"
            else f"{sample['tile_id']}@{x1}_{y1}_{x2}_{y2}"
        )
        target = {
            "labels": labels_tensor,
            "boxes": boxes_tensor,
            "boxes_norm": boxes_norm_tensor,
            "masks": masks_tensor,
            "image_id": patch_id,
            "instance_ids": ids_tensor,
            "orig_size": torch.tensor([patch_height, patch_width], dtype=torch.long),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        num_instances_after_limit = int(labels_tensor.numel())
        return {
            "pre_image": transformed["pre_image"],
            "post_image": transformed["post_image"],
            "gt_damage_map": damage_map_tensor,
            "target": target,
            "image_id": patch_id,
            "original_size": (patch_height, patch_width),
            "scaled_size": (self.image_size, self.image_size),
            "meta": {
                "sample_index": int(sample_index),
                "tile_id": str(sample["tile_id"]),
                "full_image_id": str(sample["tile_id"]),
                "pre_image_path": paths.pre_image,
                "post_image_path": paths.post_image,
                "post_label_path": paths.post_label,
                "post_target_path": paths.post_target,
                "disaster_name": sample["disaster_name"],
                "source_subset": paths.subset,
                "patch_source": patch_source,
                "patch_box": (x1, y1, x2, y2),
                "patch_size": int(patch_width),
                "sampled_patch_size": int(patch_width),
                "num_instances": num_instances_after_limit,
                "num_instances_before_limit": int(num_instances_before_limit),
                "num_instances_after_limit": int(num_instances_after_limit),
                "estimated_instance_count": int(estimated_instance_count if estimated_instance_count is not None else num_instances_before_limit),
                "gt_over_query": bool(num_instances_after_limit > self.num_queries),
                "gt_over_max_instances_per_patch": bool(gt_over_patch_limit),
                "fallback_keep_applied": bool(fallback_keep_applied),
                "dense_retry_count": int(dense_retry_count),
                "adaptive_shrink_count": int(adaptive_shrink_count),
                "adaptive_shrink_applied": bool(adaptive_shrink_count > 0),
                "requested_empty_patch": bool(requested_empty),
                "is_empty_patch": bool(num_instances_after_limit == 0),
                "is_positive_patch": bool(num_instances_after_limit > 0),
                "geometry": transformed["geometry"],
            },
        }

    def _build_full_image_item(self, sample_index: int) -> dict[str, Any]:
        sample = self.samples[sample_index]
        patch_box = (0, 0, int(sample["width"]), int(sample["height"]))
        return self._tensorize_region(
            sample_index=sample_index,
            patch_box=patch_box,
            patch_source="full_image",
            dense_retry_count=0,
            adaptive_shrink_count=0,
            allow_limit_fallback=False,
            force_image_id=str(sample["tile_id"]),
            requested_empty=False,
        )

    def _build_random_patch_item(self, sample_index: int) -> dict[str, Any]:
        selection = self._select_random_patch(sample_index)
        return self._tensorize_region(
            sample_index=sample_index,
            patch_box=selection.patch_box,
            patch_source="random_patch",
            dense_retry_count=selection.dense_retry_count,
            adaptive_shrink_count=selection.adaptive_shrink_count,
            allow_limit_fallback=selection.use_limit_fallback,
            requested_empty=selection.requested_empty,
            estimated_instance_count=selection.estimated_instance_count,
        )

    def _build_grid_patch_item(self, index: int) -> dict[str, Any]:
        sample_index, patch_box = self.grid_index[index]
        return self._tensorize_region(
            sample_index=sample_index,
            patch_box=patch_box,
            patch_source="grid_patch",
            dense_retry_count=0,
            adaptive_shrink_count=0,
            allow_limit_fallback=True,
            requested_empty=False,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.is_train:
            if self.train_mode == "full_image":
                return self._build_full_image_item(index)
            if self.train_mode == "grid_patch":
                return self._build_grid_patch_item(index)
            if self.train_mode != "random_patch":
                raise ValueError(f"Unsupported train_mode='{self.train_mode}'.")
            return self._build_random_patch_item(index)
        return self._build_full_image_item(index)

    def get_sliding_window_boxes(self, sample_index: int, *, patch_size: int | None = None, stride: int | None = None) -> list[tuple[int, int, int, int]]:
        sample = self.samples[sample_index]
        patch_size = int(patch_size or self.val_patch_size)
        stride = int(stride or self.val_stride)
        x_positions = _compute_positions(int(sample["width"]), patch_size, stride)
        y_positions = _compute_positions(int(sample["height"]), patch_size, stride)
        windows: list[tuple[int, int, int, int]] = []
        for y in y_positions:
            for x in x_positions:
                windows.append((int(x), int(y), int(x + patch_size), int(y + patch_size)))
        return windows

    def build_inference_patch_item(self, sample_index: int, patch_box: tuple[int, int, int, int]) -> dict[str, Any]:
        sample = self.samples[sample_index]
        return self._tensorize_region(
            sample_index=sample_index,
            patch_box=patch_box,
            patch_source="sliding_window",
            dense_retry_count=0,
            adaptive_shrink_count=0,
            allow_limit_fallback=False,
            force_image_id=f"{sample['tile_id']}@{patch_box[0]}_{patch_box[1]}_{patch_box[2]}_{patch_box[3]}",
            requested_empty=False,
        )

    def build_full_image_ground_truth(self, sample_index: int) -> dict[str, Any]:
        sample = self.samples[sample_index]
        width = int(sample["width"])
        height = int(sample["height"])
        patch_instances = self._collect_region_instances(sample_index, (0, 0, width, height))
        gt_instances: list[dict[str, Any]] = []
        for instance in patch_instances:
            gt_instances.append(
                {
                    "mask": instance["mask"].astype(bool),
                    "label": int(instance["label"]),
                    "box_xyxy": [float(value) for value in instance["bbox_xyxy"]],
                    "instance_id": int(instance["building_idx"]),
                }
            )
        gt_damage_map = self._crop_damage_map(sample, (0, 0, width, height), patch_instances)
        return {
            "image_id": str(sample["tile_id"]),
            "gt_instances": gt_instances,
            "gt_damage_map": gt_damage_map,
            "height": height,
            "width": width,
        }
