from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from scipy import ndimage
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.input_utils import (
    build_boundary_tensor,
    build_cache_key_summary,
    build_crop_box,
    crop_image,
    ensure_rgb_image,
    maybe_build_abs_diff,
    resolve_diff_input_config,
    resolve_input_mode_config,
)
from datasets.transforms import build_transforms
from datasets.xbd_oracle_instance_damage import CLASS_NAMES, LABEL_TO_INDEX
from utils.cache import make_cache_path, load_pickle, save_pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _read_split_file(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _discover_bright_tiles(root_dir: str | Path) -> dict[str, dict[str, str]]:
    root_dir = Path(root_dir)
    pre_paths = {
        path.name.replace("_pre_disaster.tif", ""): str(path)
        for path in (root_dir / "pre-event").glob("*_pre_disaster.tif")
        if path.is_file() and path.stat().st_size > 0
    }
    post_paths = {
        path.name.replace("_post_disaster.tif", ""): str(path)
        for path in (root_dir / "post-event").glob("*_post_disaster.tif")
        if path.is_file() and path.stat().st_size > 0
    }
    target_paths = {
        path.name.replace("_building_damage.tif", ""): str(path)
        for path in (root_dir / "target").glob("*_building_damage.tif")
        if path.is_file() and path.stat().st_size > 0
    }

    paired_ids = sorted(set(pre_paths) & set(post_paths) & set(target_paths))
    return {
        tile_id: {
            "pre_image": pre_paths[tile_id],
            "post_image": post_paths[tile_id],
            "target_mask": target_paths[tile_id],
        }
        for tile_id in paired_ids
    }


def _resolve_bright_split_ids(root_dir: str | Path, split_name: str, config: dict[str, Any]) -> list[str]:
    bright_cfg = config["data"]["bright"]
    split_key = {
        "train": "train_list",
        "val": "val_list",
        "hold": "val_list",
        "test": "test_list",
    }.get(split_name, "test_list")
    split_file = bright_cfg.get(split_key)
    if split_file:
        return _read_split_file(split_file)

    tile_ids = sorted(_discover_bright_tiles(root_dir).keys())
    rng = random.Random(int(bright_cfg.get("split_seed", config.get("seed", 42))))
    shuffled = tile_ids[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_ratio = float(bright_cfg.get("train_ratio", 0.8))
    val_ratio = float(bright_cfg.get("val_ratio", 0.1))
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]

    if split_name == "train":
        return train_ids
    if split_name in {"val", "hold"}:
        return val_ids
    return test_ids


def _build_cache_payload(config: dict[str, Any], split_name: str) -> dict[str, Any]:
    bright_root = Path(config["data"]["bright_root"])
    bright_cfg = config["data"]["bright"]
    return {
        "dataset_version": 1,
        "split_name": split_name,
        "bright_root": str(bright_root.resolve()),
        "bright_root_mtime": bright_root.stat().st_mtime,
        "target_mode": str(bright_cfg.get("target_mode", "binary_damage_mask")),
        "positive_class_name": bright_cfg.get("positive_class_name"),
        "label_mapping": copy.deepcopy(bright_cfg.get("label_mapping")),
        "train_list": bright_cfg.get("train_list"),
        "val_list": bright_cfg.get("val_list"),
        "test_list": bright_cfg.get("test_list"),
        "train_ratio": float(bright_cfg.get("train_ratio", 0.8)),
        "val_ratio": float(bright_cfg.get("val_ratio", 0.1)),
        "split_seed": int(bright_cfg.get("split_seed", config.get("seed", 42))),
        "min_mask_pixels": int(config["data"]["min_mask_pixels"]),
        "input_signature": build_cache_key_summary(config),
    }


class BrightOpticalInstanceDamageDataset(Dataset):
    def __init__(
        self,
        config: dict[str, Any],
        split_name: str,
        is_train: bool = False,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.split_name = split_name
        self.is_train = is_train

        bright_cfg = self.config["data"]["bright"]
        self.bright_root = Path(self.config["data"]["bright_root"])
        self.target_mode = str(bright_cfg.get("target_mode", "binary_damage_mask"))
        raw_label_mapping = bright_cfg.get("label_mapping")
        self.label_mapping: dict[int, int] = {}
        self.label_mapping_names: dict[int, str] = {}
        if raw_label_mapping:
            for raw_value, class_name in raw_label_mapping.items():
                class_name_str = str(class_name)
                if class_name_str not in LABEL_TO_INDEX:
                    raise ValueError(
                        f"Unsupported BRIGHT label mapping target '{class_name_str}'. Choose one of {CLASS_NAMES}."
                    )
                raw_value_int = int(raw_value)
                self.label_mapping[raw_value_int] = LABEL_TO_INDEX[class_name_str]
                self.label_mapping_names[raw_value_int] = class_name_str
        else:
            positive_class_name = bright_cfg.get("positive_class_name")
            if positive_class_name is None:
                raise ValueError(
                    "BRIGHT local target requires either data.bright.label_mapping or data.bright.positive_class_name."
                )
            if str(positive_class_name) not in LABEL_TO_INDEX:
                raise ValueError(
                    f"Unsupported data.bright.positive_class_name='{positive_class_name}'. Choose one of {CLASS_NAMES}."
                )
            self.label_mapping[1] = LABEL_TO_INDEX[str(positive_class_name)]
            self.label_mapping_names[1] = str(positive_class_name)
        self.post_convert_mode = str(bright_cfg.get("post_convert_mode", "grayscale_to_rgb"))
        self.component_connectivity = int(bright_cfg.get("component_connectivity", 2))
        self.input_mode_cfg = resolve_input_mode_config(self.config)
        self.diff_cfg = resolve_diff_input_config(self.config)
        self.use_dual_scale = bool(self.input_mode_cfg["use_dual_scale"])
        self.use_context_branch = bool(self.config.get("model", {}).get("use_context_branch", False))
        self.return_context = bool(self.use_dual_scale or self.use_context_branch)

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

        payload = _build_cache_payload(self.config, split_name)
        cache_path = make_cache_path(self.config["data"]["cache_dir"], f"bright_optical_{split_name}", payload)
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
        self.source_counts = {"bright": len(self.samples)}

    def _component_structure(self) -> np.ndarray:
        if self.component_connectivity == 1:
            return np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        return np.ones((3, 3), dtype=np.uint8)

    def _build_samples(self) -> list[dict[str, Any]]:
        tile_paths = _discover_bright_tiles(self.bright_root)
        split_ids = _resolve_bright_split_ids(self.bright_root, self.split_name, self.config)
        min_mask_pixels = int(self.config["data"]["min_mask_pixels"])
        structure = self._component_structure()

        samples: list[dict[str, Any]] = []
        for tile_id in tqdm(split_ids, desc=f"Indexing bright-{self.split_name}", leave=False):
            tile = tile_paths.get(tile_id)
            if tile is None:
                continue
            target_array = np.asarray(Image.open(tile["target_mask"]), dtype=np.uint8)
            positive_values = sorted(int(value) for value in np.unique(target_array) if int(value) > 0)
            unmapped_values = [value for value in positive_values if value not in self.label_mapping]
            if unmapped_values:
                raise ValueError(
                    f"Found BRIGHT target values {unmapped_values} in {tile_id} without data.bright.label_mapping entries."
                )
            running_index = 0
            for raw_value in positive_values:
                class_mask = (target_array == int(raw_value)).astype(np.uint8)
                component_map, num_components = ndimage.label(class_mask, structure=structure)
                if num_components <= 0:
                    continue
                component_slices = ndimage.find_objects(component_map)
                for component_index, component_slice in enumerate(component_slices, start=1):
                    if component_slice is None:
                        continue
                    component_crop = component_map[component_slice] == component_index
                    mask_pixels = int(component_crop.sum())
                    if mask_pixels < min_mask_pixels:
                        continue
                    ys, xs = np.nonzero(component_crop)
                    if ys.size == 0 or xs.size == 0:
                        continue
                    y_slice, x_slice = component_slice
                    tight_bbox = (
                        int(x_slice.start + xs.min()),
                        int(y_slice.start + ys.min()),
                        int(x_slice.start + xs.max() + 1),
                        int(y_slice.start + ys.max() + 1),
                    )
                    anchor_xy = (int(x_slice.start + xs[0]), int(y_slice.start + ys[0]))
                    class_name = self.label_mapping_names[int(raw_value)]
                    samples.append(
                        {
                            "tile_id": tile_id,
                            "building_idx": running_index,
                            "label": int(self.label_mapping[int(raw_value)]),
                            "original_subtype": class_name,
                            "raw_target_value": int(raw_value),
                            "bbox_xyxy": list(tight_bbox),
                            "tight_bbox_xyxy": list(tight_bbox),
                            "anchor_xy": [int(anchor_xy[0]), int(anchor_xy[1])],
                            "mask_pixels": mask_pixels,
                            "split": self.split_name,
                            "source_subset": self.split_name,
                            "pre_image": tile["pre_image"],
                            "post_image": tile["post_image"],
                            "target_mask": tile["target_mask"],
                            "source_name": "bright",
                            "sample_id": f"bright:{tile_id}:{running_index}",
                        }
                    )
                    running_index += 1
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_crop_box(self, sample: dict[str, Any], *, mode: str) -> tuple[int, int, int, int]:
        if mode == "context" and not self.return_context:
            mode = "local"
        return build_crop_box(
            tuple(float(v) for v in sample["tight_bbox_xyxy"]),
            mode=mode,
            local_margin_ratio=float(self.input_mode_cfg["local_margin_ratio"]),
            context_scale=float(self.input_mode_cfg["context_scale"]),
            min_crop_size=int(self.input_mode_cfg["min_crop_size"]),
        )

    def _extract_component_mask(
        self,
        target_binary_image: Image.Image,
        crop_box_xyxy: tuple[int, int, int, int],
        anchor_xy: tuple[int, int],
    ) -> Image.Image:
        crop = crop_image(target_binary_image, crop_box_xyxy)
        binary_crop = (np.asarray(crop, dtype=np.uint8) > 0).astype(np.uint8)
        rel_x = int(anchor_xy[0] - crop_box_xyxy[0])
        rel_y = int(anchor_xy[1] - crop_box_xyxy[1])
        rel_x = min(max(rel_x, 0), max(binary_crop.shape[1] - 1, 0))
        rel_y = min(max(rel_y, 0), max(binary_crop.shape[0] - 1, 0))

        component_map, num_components = ndimage.label(binary_crop, structure=self._component_structure())
        if num_components <= 0:
            return Image.fromarray(np.zeros_like(binary_crop, dtype=np.uint8), mode="L")

        component_label = int(component_map[rel_y, rel_x])
        if component_label <= 0:
            positive_labels = np.unique(component_map[binary_crop > 0])
            positive_labels = positive_labels[positive_labels > 0]
            if positive_labels.size == 1:
                component_label = int(positive_labels[0])
            else:
                return Image.fromarray((binary_crop * 255).astype(np.uint8), mode="L")

        component_mask = (component_map == component_label).astype(np.uint8)
        return Image.fromarray((component_mask * 255).astype(np.uint8), mode="L")

    def _transform_scale(
        self,
        pre_crop: Image.Image,
        post_crop: Image.Image,
        mask_image: Image.Image,
        *,
        transform: Any,
    ) -> dict[str, Any]:
        if self.dual_view_enabled and hasattr(transform, "make_dual_views"):
            view1, view2 = transform.make_dual_views(
                pre_crop,
                post_crop,
                mask_image,
                extra_context_prob=self.dual_view_stronger_context_prob,
            )
            pre_tensor, post_tensor, mask_tensor, augmentation_stats = view1
            view2_pre_tensor, view2_post_tensor, view2_mask_tensor, view2_augmentation_stats = view2
        else:
            if self.is_train:
                pre_tensor, post_tensor, mask_tensor, augmentation_stats = transform(
                    pre_crop,
                    post_crop,
                    mask_image,
                    return_stats=True,
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
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre_image = Image.open(sample["pre_image"]).convert("RGB")
        post_image = ensure_rgb_image(Image.open(sample["post_image"]), convert_mode=self.post_convert_mode)
        target_array = np.asarray(Image.open(sample["target_mask"]), dtype=np.uint8)
        target_binary_image = Image.fromarray(
            (((target_array == int(sample["raw_target_value"])).astype(np.uint8)) * 255),
            mode="L",
        )
        anchor_xy = (int(sample["anchor_xy"][0]), int(sample["anchor_xy"][1]))

        local_crop_box = self._resolve_crop_box(sample, mode="local")
        local_outputs = self._transform_scale(
            crop_image(pre_image, local_crop_box),
            crop_image(post_image, local_crop_box),
            self._extract_component_mask(target_binary_image, local_crop_box, anchor_xy),
            transform=self.local_transform,
        )

        context_outputs = None
        context_crop_box = None
        if self.return_context:
            context_crop_box = self._resolve_crop_box(sample, mode="context")
            context_outputs = self._transform_scale(
                crop_image(pre_image, context_crop_box),
                crop_image(post_image, context_crop_box),
                self._extract_component_mask(target_binary_image, context_crop_box, anchor_xy),
                transform=self.context_transform,
            )

        meta = {
            "tile_id": sample["tile_id"],
            "building_idx": int(sample["building_idx"]),
            "original_subtype": sample["original_subtype"],
            "bbox_xyxy": list(sample["bbox_xyxy"]),
            "local_crop_bbox_xyxy": list(local_crop_box),
            "context_crop_bbox_xyxy": None if context_crop_box is None else list(context_crop_box),
            "source_subset": sample["source_subset"],
            "split": sample["split"],
            "cache_path": self.cache_path,
            "source_name": "bright",
            "sample_id": sample["sample_id"],
            "post_convert_mode": self.post_convert_mode,
            "label_mapping": {
                "target_mode": self.target_mode,
                "raw_value_to_class_name": self.label_mapping_names,
            },
            "sar_available": False,
        }

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
            "source_name": "bright",
            "sample_id": sample["sample_id"],
            "sar_pre": None,
            "sar_post": None,
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
