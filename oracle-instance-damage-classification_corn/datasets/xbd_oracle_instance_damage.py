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

from datasets.input_utils import (
    build_boundary_tensor,
    build_cache_key_summary,
    build_crop_box,
    crop_image,
    maybe_build_abs_diff,
    resolve_diff_input_config,
    resolve_geometry_prior_config,
    resolve_input_mode_config,
)
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
            legacy_crop_bbox_float = expand_bbox(tight_bbox, float(data_cfg["context_ratio"]))
            if (
                out_of_bounds_fraction(legacy_crop_bbox_float, image_width, image_height)
                > float(data_cfg["max_out_of_bound_ratio"])
            ):
                continue

            legacy_crop_bbox = clip_bbox_to_image(legacy_crop_bbox_float, image_width, image_height)
            crop_w = legacy_crop_bbox[2] - legacy_crop_bbox[0]
            crop_h = legacy_crop_bbox[3] - legacy_crop_bbox[1]
            if crop_w <= 1 or crop_h <= 1:
                continue

            mask = polygon_to_mask(
                polygon_xy,
                crop_h,
                crop_w,
                offset=(legacy_crop_bbox[0], legacy_crop_bbox[1]),
            )
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
                "tight_bbox_xyxy": [float(v) for v in tight_bbox],
                "legacy_crop_bbox_xyxy": [int(v) for v in legacy_crop_bbox],
                "mask_pixels": mask_pixels,
                "split": split_name,
                "source_subset": tile_paths.source_subset,
                "pre_image": tile_paths.pre_image,
                "post_image": tile_paths.post_image,
                "post_label": tile_paths.post_label,
                "source_name": "xbd",
                "sample_id": f"xbd:{tile_id}:{building_idx}",
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
        "dataset_version": 2,
        "split_name": split_name,
        "list_path": str(list_path.resolve()),
        "list_mtime": list_path.stat().st_mtime,
        "root_dir": str(Path(config["data"]["root_dir"]).resolve()),
        "instance_source": config["data"]["instance_source"],
        "allow_tier3": bool(config["data"]["allow_tier3"]),
        "context_ratio": float(config["data"]["context_ratio"]),
        "min_polygon_area": float(config["data"]["min_polygon_area"]),
        "min_mask_pixels": int(config["data"]["min_mask_pixels"]),
        "max_out_of_bound_ratio": float(config["data"]["max_out_of_bound_ratio"]),
        "input_signature": build_cache_key_summary(config),
    }


def oracle_instance_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif key in {"label", "sample_index"}:
            collated[key] = torch.tensor(values, dtype=torch.long)
        else:
            collated[key] = values
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
        self.split_name = split_name
        self.list_path = str(list_path)
        self.is_train = is_train

        self.input_mode_cfg = resolve_input_mode_config(self.config)
        self.geometry_cfg = resolve_geometry_prior_config(self.config)
        self.diff_cfg = resolve_diff_input_config(self.config)
        self.use_dual_scale = bool(self.input_mode_cfg["use_dual_scale"])
        self.use_context_branch = bool(self.config.get("model", {}).get("use_context_branch", False))
        self.return_context = bool(self.use_dual_scale or self.use_context_branch)
        self.pixel_aux_cfg = self.config.get("pixel_auxiliary", {})
        self.pixel_supervision_cfg = self.pixel_aux_cfg.get("supervision", {})
        self.pixel_aux_enabled = bool(self.pixel_aux_cfg.get("enabled", False))
        self.pixel_train_only = bool(self.pixel_aux_cfg.get("train_only", True))
        self.enable_pixel_targets = bool(self.pixel_aux_enabled and (self.is_train or not self.pixel_train_only))
        self.prefer_true_pixel_target = bool(self.pixel_supervision_cfg.get("prefer_true_pixel_target", True))
        self.allow_rasterized_fallback = bool(self.pixel_supervision_cfg.get("allow_rasterized_fallback", True))
        self.xbd_target_subdir = str(self.pixel_supervision_cfg.get("xbd_target_subdir", "targets"))
        self.xbd_target_suffix = str(self.pixel_supervision_cfg.get("xbd_target_suffix", "_post_disaster_target.png"))
        self.xbd_target_value_mapping = {
            int(raw_value): int(mapped_value)
            for raw_value, mapped_value in self.pixel_supervision_cfg.get(
                "xbd_target_value_to_damage_id",
                {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            ).items()
        }

        local_transform_cfg = copy.deepcopy(self.config)
        local_transform_cfg["data"]["image_size"] = int(self.input_mode_cfg["local_size"])
        self.local_transform = build_transforms(local_transform_cfg, is_train=is_train)

        context_transform_cfg = copy.deepcopy(self.config)
        context_transform_cfg["data"]["image_size"] = int(self.input_mode_cfg["context_size"])
        self.context_transform = build_transforms(context_transform_cfg, is_train=is_train)

        self.dual_view_enabled = bool(self.is_train and self.config.get("training", {}).get("dual_view_enabled", False))
        self.dual_view_stronger_context_prob = float(
            self.config.get("training", {}).get("dual_view_stronger_context_prob", 0.20)
        )

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
        self.source_counts = {"xbd": len(self.samples)}

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

    def _resolve_crop_box(self, sample: dict[str, Any], *, mode: str) -> tuple[int, int, int, int]:
        if mode == "local" and not self.use_dual_scale:
            return tuple(int(v) for v in sample["legacy_crop_bbox_xyxy"])
        if mode == "context" and not self.return_context:
            return tuple(int(v) for v in sample["legacy_crop_bbox_xyxy"])
        tight_bbox = tuple(float(v) for v in sample.get("tight_bbox_xyxy", sample["bbox_xyxy"]))
        return build_crop_box(
            tight_bbox,
            mode=mode,
            local_margin_ratio=float(self.input_mode_cfg["local_margin_ratio"]),
            context_scale=float(self.input_mode_cfg["context_scale"]),
            min_crop_size=int(self.input_mode_cfg["min_crop_size"]),
        )

    def _make_mask_image(self, sample: dict[str, Any], crop_box_xyxy: tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = crop_box_xyxy
        crop_w = max(int(x2 - x1), 1)
        crop_h = max(int(y2 - y1), 1)
        mask_np = polygon_to_mask(sample["polygon_xy"], crop_h, crop_w, offset=(x1, y1))
        return Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

    def _transform_scale(
        self,
        pre_crop: Image.Image,
        post_crop: Image.Image,
        mask_image: Image.Image,
        *,
        transform: Any,
        extra_masks: list[Image.Image] | None = None,
    ) -> dict[str, Any]:
        if self.dual_view_enabled and hasattr(transform, "make_dual_views"):
            if extra_masks is not None and hasattr(transform, "make_dual_views_with_extra_masks"):
                view1, view2, view1_extra_masks, view2_extra_masks = transform.make_dual_views_with_extra_masks(
                    pre_crop,
                    post_crop,
                    mask_image,
                    extra_masks=extra_masks,
                    extra_context_prob=self.dual_view_stronger_context_prob,
                )
            else:
                view1, view2 = transform.make_dual_views(
                    pre_crop,
                    post_crop,
                    mask_image,
                    extra_context_prob=self.dual_view_stronger_context_prob,
                )
                view1_extra_masks = None
                view2_extra_masks = None
            pre_tensor, post_tensor, mask_tensor, augmentation_stats = view1
            view2_pre_tensor, view2_post_tensor, view2_mask_tensor, view2_augmentation_stats = view2
        else:
            view1_extra_masks = None
            view2_extra_masks = None
            if self.is_train:
                if extra_masks is not None and hasattr(transform, "apply_with_extra_masks"):
                    pre_tensor, post_tensor, mask_tensor, view1_extra_masks, augmentation_stats = transform.apply_with_extra_masks(
                        pre_crop,
                        post_crop,
                        mask_image,
                        extra_masks=extra_masks,
                        return_stats=True,
                    )
                else:
                    transformed = transform(pre_crop, post_crop, mask_image, return_stats=True)
                    pre_tensor, post_tensor, mask_tensor, augmentation_stats = transformed
            else:
                if extra_masks is not None and hasattr(transform, "apply_with_extra_masks"):
                    pre_tensor, post_tensor, mask_tensor, view1_extra_masks = transform.apply_with_extra_masks(
                        pre_crop,
                        post_crop,
                        mask_image,
                        extra_masks=extra_masks,
                    )
                else:
                    pre_tensor, post_tensor, mask_tensor = transform(pre_crop, post_crop, mask_image)
                augmentation_stats = {}
            view2_pre_tensor = None
            view2_post_tensor = None
            view2_mask_tensor = None
            view2_augmentation_stats = None

        if mask_tensor.sum().item() <= 0:
            mask_tensor = torch.ones_like(mask_tensor)
        boundary_tensor = build_boundary_tensor(mask_tensor, self.config)

        view2_boundary_tensor = None
        if view2_mask_tensor is not None:
            if view2_mask_tensor.sum().item() <= 0:
                view2_mask_tensor = torch.ones_like(view2_mask_tensor)
            view2_boundary_tensor = build_boundary_tensor(view2_mask_tensor, self.config)

        return {
            "pre": pre_tensor,
            "post": post_tensor,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
            "abs_diff": maybe_build_abs_diff(pre_tensor, post_tensor, self.config),
            "augmentation_stats": augmentation_stats,
            "view2_pre": view2_pre_tensor,
            "view2_post": view2_post_tensor,
            "view2_mask": view2_mask_tensor,
            "view2_boundary": view2_boundary_tensor,
            "view2_abs_diff": (
                None
                if view2_pre_tensor is None or view2_post_tensor is None
                else maybe_build_abs_diff(view2_pre_tensor, view2_post_tensor, self.config)
            ),
            "view2_augmentation_stats": view2_augmentation_stats,
            "extra_masks": view1_extra_masks,
            "view2_extra_masks": view2_extra_masks,
        }

    def _resolve_xbd_pixel_target_path(self, sample: dict[str, Any]) -> Path | None:
        subset = str(sample.get("source_subset", "train"))
        target_path = Path(self.config["data"]["root_dir"]) / subset / self.xbd_target_subdir / (
            f"{sample['tile_id']}{self.xbd_target_suffix}"
        )
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path
        return None

    def _map_damage_values(self, target_array: np.ndarray) -> np.ndarray:
        mapped = np.zeros_like(target_array, dtype=np.uint8)
        for raw_value, mapped_value in self.xbd_target_value_mapping.items():
            mapped[target_array == int(raw_value)] = int(mapped_value)
        return mapped

    def _build_local_pixel_supervision(
        self,
        sample: dict[str, Any],
        local_crop_box: tuple[int, int, int, int],
        local_mask_image: Image.Image,
    ) -> dict[str, Any]:
        default = {
            "local_building_target_image": None,
            "local_damage_target_image": None,
            "has_pixel_supervision": False,
            "pixel_source_name": "none",
        }
        if not self.enable_pixel_targets:
            return default

        target_path = self._resolve_xbd_pixel_target_path(sample)
        if self.prefer_true_pixel_target and target_path is not None:
            target_array = np.asarray(Image.open(target_path), dtype=np.uint8)
            mapped_target = self._map_damage_values(target_array)
            mapped_target_image = Image.fromarray(mapped_target, mode="L")
            local_damage_target_image = crop_image(mapped_target_image, local_crop_box)
            local_damage_target_array = np.asarray(local_damage_target_image, dtype=np.uint8)
            local_building_target_image = Image.fromarray(
                ((local_damage_target_array > 0).astype(np.uint8)),
                mode="L",
            )
            return {
                "local_building_target_image": local_building_target_image,
                "local_damage_target_image": local_damage_target_image,
                "has_pixel_supervision": True,
                "pixel_source_name": "xbd",
            }

        if self.allow_rasterized_fallback:
            local_mask_array = (np.asarray(local_mask_image, dtype=np.uint8) > 0).astype(np.uint8)
            weak_damage_array = local_mask_array * int(sample["label"] + 1)
            return {
                "local_building_target_image": Image.fromarray(local_mask_array, mode="L"),
                "local_damage_target_image": Image.fromarray(weak_damage_array.astype(np.uint8), mode="L"),
                "has_pixel_supervision": True,
                "pixel_source_name": "xbd",
            }

        return default

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre_image = Image.open(sample["pre_image"]).convert("RGB")
        post_image = Image.open(sample["post_image"]).convert("RGB")

        local_crop_box = self._resolve_crop_box(sample, mode="local")
        local_pre_crop = crop_image(pre_image, local_crop_box)
        local_post_crop = crop_image(post_image, local_crop_box)
        local_mask_image = self._make_mask_image(sample, local_crop_box)
        local_pixel_targets = self._build_local_pixel_supervision(sample, local_crop_box, local_mask_image)
        local_extra_masks = None
        if (
            local_pixel_targets["local_building_target_image"] is not None
            and local_pixel_targets["local_damage_target_image"] is not None
        ):
            local_extra_masks = [
                local_pixel_targets["local_building_target_image"],
                local_pixel_targets["local_damage_target_image"],
            ]
        local_outputs = self._transform_scale(
            local_pre_crop,
            local_post_crop,
            local_mask_image,
            transform=self.local_transform,
            extra_masks=local_extra_masks,
        )

        context_outputs = None
        context_crop_box = None
        if self.return_context:
            context_crop_box = self._resolve_crop_box(sample, mode="context")
            context_pre_crop = crop_image(pre_image, context_crop_box)
            context_post_crop = crop_image(post_image, context_crop_box)
            context_mask_image = self._make_mask_image(sample, context_crop_box)
            context_outputs = self._transform_scale(
                context_pre_crop,
                context_post_crop,
                context_mask_image,
                transform=self.context_transform,
            )

        meta = {
            "tile_id": sample["tile_id"],
            "building_idx": int(sample["building_idx"]),
            "uid": sample.get("uid"),
            "original_subtype": sample["original_subtype"],
            "polygon_area": float(sample["polygon_area"]),
            "bbox_xyxy": list(sample["bbox_xyxy"]),
            "legacy_crop_bbox_xyxy": list(sample["legacy_crop_bbox_xyxy"]),
            "local_crop_bbox_xyxy": list(local_crop_box),
            "context_crop_bbox_xyxy": None if context_crop_box is None else list(context_crop_box),
            "source_subset": sample["source_subset"],
            "split": sample["split"],
            "cache_path": self.cache_path,
            "source_name": "xbd",
            "sample_id": sample["sample_id"],
            "sar_available": False,
            "pixel_source_name": local_pixel_targets["pixel_source_name"],
            "has_pixel_supervision": bool(local_pixel_targets["has_pixel_supervision"]),
        }

        extra_masks = local_outputs.get("extra_masks")
        if extra_masks is not None and len(extra_masks) >= 2:
            local_building_target = (extra_masks[0] > 0).float().unsqueeze(0)
            local_damage_target = extra_masks[1].long()
        else:
            local_building_target = torch.zeros_like(local_outputs["mask"])
            local_damage_target = torch.zeros(
                local_outputs["mask"].shape[-2:],
                dtype=torch.long,
            )

        sample_dict: dict[str, Any] = {
            "pre_image": local_outputs["pre"],
            "post_image": local_outputs["post"],
            "instance_mask": local_outputs["mask"],
            "instance_boundary": local_outputs["boundary"],
            "local_pre": local_outputs["pre"],
            "local_post": local_outputs["post"],
            "local_mask": local_outputs["mask"],
            "local_boundary": local_outputs["boundary"],
            "label": int(sample["label"]),
            "sample_index": int(index),
            "meta": meta,
            "augmentation_stats": local_outputs["augmentation_stats"],
            "source_name": "xbd",
            "sample_id": sample["sample_id"],
            "sar_pre": None,
            "sar_post": None,
            "local_building_target": local_building_target,
            "local_damage_target": local_damage_target,
            "has_pixel_supervision": bool(local_pixel_targets["has_pixel_supervision"]),
            "pixel_source_name": local_pixel_targets["pixel_source_name"],
        }
        if local_outputs["abs_diff"] is not None:
            sample_dict["local_abs_diff"] = local_outputs["abs_diff"]

        if context_outputs is not None:
            sample_dict["context_pre"] = context_outputs["pre"]
            sample_dict["context_post"] = context_outputs["post"]
            sample_dict["context_mask"] = context_outputs["mask"]
            sample_dict["context_boundary"] = context_outputs["boundary"]
            sample_dict["context_augmentation_stats"] = context_outputs["augmentation_stats"]
            if context_outputs["abs_diff"] is not None:
                sample_dict["context_abs_diff"] = context_outputs["abs_diff"]

        if local_outputs["view2_pre"] is not None and local_outputs["view2_post"] is not None:
            sample_dict["view2_pre_image"] = local_outputs["view2_pre"]
            sample_dict["view2_post_image"] = local_outputs["view2_post"]
            sample_dict["view2_instance_mask"] = local_outputs["view2_mask"]
            sample_dict["view2_instance_boundary"] = local_outputs["view2_boundary"]
            sample_dict["view2_local_pre"] = local_outputs["view2_pre"]
            sample_dict["view2_local_post"] = local_outputs["view2_post"]
            sample_dict["view2_local_mask"] = local_outputs["view2_mask"]
            sample_dict["view2_local_boundary"] = local_outputs["view2_boundary"]
            sample_dict["view2_augmentation_stats"] = local_outputs["view2_augmentation_stats"]
            if local_outputs["view2_abs_diff"] is not None:
                sample_dict["view2_local_abs_diff"] = local_outputs["view2_abs_diff"]

        if context_outputs is not None and context_outputs["view2_pre"] is not None and context_outputs["view2_post"] is not None:
            sample_dict["view2_context_pre"] = context_outputs["view2_pre"]
            sample_dict["view2_context_post"] = context_outputs["view2_post"]
            sample_dict["view2_context_mask"] = context_outputs["view2_mask"]
            sample_dict["view2_context_boundary"] = context_outputs["view2_boundary"]
            sample_dict["view2_context_augmentation_stats"] = context_outputs["view2_augmentation_stats"]
            if context_outputs["view2_abs_diff"] is not None:
                sample_dict["view2_context_abs_diff"] = context_outputs["view2_abs_diff"]
        return sample_dict
