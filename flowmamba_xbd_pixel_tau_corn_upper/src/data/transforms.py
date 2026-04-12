"""Pair-image and polygon-aware transforms for xBD."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance

from .xbd_polygon import (
    crop_polygons,
    flip_polygons_horizontal,
    flip_polygons_vertical,
    rotate_polygons_90,
    scale_polygons,
)


@dataclass
class XBDTransformPipeline:
    mode: str
    crop_size: Tuple[int, int]
    image_size: Tuple[int, int]
    augment_cfg: Dict[str, object]
    min_polygon_area: float
    validation_cfg: Dict[str, object]

    def _pad_to_crop(self, image: np.ndarray, crop_size: Tuple[int, int], fill_value: int = 0) -> np.ndarray:
        target_h, target_w = crop_size
        height, width = image.shape[:2]
        pad_h = max(target_h - height, 0)
        pad_w = max(target_w - width, 0)
        if pad_h == 0 and pad_w == 0:
            return image
        if image.ndim == 3:
            out = np.full((height + pad_h, width + pad_w, image.shape[2]), fill_value, dtype=image.dtype)
            out[:height, :width] = image
            return out
        out = np.full((height + pad_h, width + pad_w), fill_value, dtype=image.dtype)
        out[:height, :width] = image
        return out

    def _sample_crop_size(self) -> Tuple[int, int]:
        crop_h, crop_w = self.crop_size
        random_resized_crop = self.augment_cfg.get("random_resized_crop", False)
        if self.mode != "train" or not random_resized_crop:
            return crop_h, crop_w

        scale = self.augment_cfg.get("random_resized_crop_scale", [0.8, 1.0])
        if isinstance(scale, (list, tuple)) and len(scale) == 2:
            scale_min, scale_max = float(scale[0]), float(scale[1])
        else:
            scale_min, scale_max = 0.8, 1.0
        sampled = random.uniform(min(scale_min, scale_max), max(scale_min, scale_max))
        active_h = max(int(round(crop_h * sampled)), 128)
        active_w = max(int(round(crop_w * sampled)), 128)
        return active_h, active_w

    def _random_or_center_crop(
        self,
        post_target: np.ndarray,
        crop_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        crop_h, crop_w = crop_size
        height, width = post_target.shape[:2]
        if self.mode != "train":
            top = max((height - crop_h) // 2, 0)
            left = max((width - crop_w) // 2, 0)
            return left, top, crop_w, crop_h

        attempts = int(self.augment_cfg.get("crop_retry", 1))
        min_building_pixels = int(self.augment_cfg.get("min_building_pixels", 0))
        min_building_ratio = float(self.augment_cfg.get("min_building_ratio", 0.0))

        best_box = (0, 0, crop_w, crop_h)
        best_score = -1
        for _ in range(max(attempts, 1)):
            top = random.randint(0, max(height - crop_h, 0))
            left = random.randint(0, max(width - crop_w, 0))
            crop = post_target[top : top + crop_h, left : left + crop_w]
            building_pixels = int((crop > 0).sum())
            valid_ratio = building_pixels / max(crop_h * crop_w, 1)
            if building_pixels > best_score:
                best_score = building_pixels
                best_box = (left, top, crop_w, crop_h)
            if building_pixels >= min_building_pixels and valid_ratio >= min_building_ratio:
                return left, top, crop_w, crop_h
        return best_box

    @staticmethod
    def _resize_rgb(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        width, height = size[1], size[0]
        return np.asarray(Image.fromarray(image).resize((width, height), resample=Image.BILINEAR))

    @staticmethod
    def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        width, height = size[1], size[0]
        return np.asarray(Image.fromarray(mask).resize((width, height), resample=Image.NEAREST))

    @staticmethod
    def _apply_color_jitter(pre_image: np.ndarray, post_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        pre = Image.fromarray(pre_image)
        post = Image.fromarray(post_image)
        for enhancer_cls, factor in ((ImageEnhance.Brightness, brightness), (ImageEnhance.Contrast, contrast)):
            pre = enhancer_cls(pre).enhance(factor)
            post = enhancer_cls(post).enhance(factor)
        return np.asarray(pre), np.asarray(post)

    def _apply_train_augmentations(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        post_target: np.ndarray,
        polygons: List[Dict[str, object]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]], Dict[str, object]]:
        meta = {"applied_hflip": False, "applied_vflip": False, "applied_rot90_k": 0}
        current_h, current_w = post_target.shape[:2]

        if self.augment_cfg.get("random_hflip", False) and random.random() < 0.5:
            pre_image = np.flip(pre_image, axis=1).copy()
            post_image = np.flip(post_image, axis=1).copy()
            post_target = np.flip(post_target, axis=1).copy()
            polygons = flip_polygons_horizontal(polygons, width=current_w)
            meta["applied_hflip"] = True

        if self.augment_cfg.get("random_vflip", False) and random.random() < 0.5:
            pre_image = np.flip(pre_image, axis=0).copy()
            post_image = np.flip(post_image, axis=0).copy()
            post_target = np.flip(post_target, axis=0).copy()
            polygons = flip_polygons_vertical(polygons, height=current_h)
            meta["applied_vflip"] = True

        if self.augment_cfg.get("random_rotate_90", False):
            rot_k = random.randint(0, 3)
            if rot_k:
                pre_image = np.rot90(pre_image, k=rot_k, axes=(0, 1)).copy()
                post_image = np.rot90(post_image, k=rot_k, axes=(0, 1)).copy()
                post_target = np.rot90(post_target, k=rot_k, axes=(0, 1)).copy()
                polygons = rotate_polygons_90(polygons, width=current_w, height=current_h, k=rot_k)
                meta["applied_rot90_k"] = rot_k

        if self.augment_cfg.get("color_jitter", False):
            pre_image, post_image = self._apply_color_jitter(pre_image, post_image)

        return pre_image, post_image, post_target, polygons, meta

    def __call__(self, sample: Dict[str, object]) -> Dict[str, object]:
        pre_image = sample["pre_image"]
        post_image = sample["post_image"]
        post_target = sample["post_target"]
        polygons: List[Dict[str, object]] = sample["polygons"]

        val_mode = str(self.validation_cfg.get("mode", "center_crop")).lower()
        if self.mode != "train" and val_mode in {"full_image", "tiled"}:
            sample["meta"] = {
                **sample["meta"],
                "crop_box": [0, 0, int(post_target.shape[1]), int(post_target.shape[0])],
                "scale_xy": [1.0, 1.0],
                "padded_size_hw": [int(post_target.shape[0]), int(post_target.shape[1])],
                "output_size_hw": [int(post_target.shape[0]), int(post_target.shape[1])],
                "applied_hflip": False,
                "applied_vflip": False,
                "applied_rot90_k": 0,
                "validation_mode": val_mode,
            }
            return sample

        if self.mode != "train" and val_mode == "full_image_resize":
            orig_h, orig_w = post_target.shape[:2]
            scale_y = self.image_size[0] / float(orig_h)
            scale_x = self.image_size[1] / float(orig_w)
            pre_image = self._resize_rgb(pre_image, self.image_size)
            post_image = self._resize_rgb(post_image, self.image_size)
            post_target = self._resize_mask(post_target, self.image_size)
            polygons = scale_polygons(polygons, scale_x=scale_x, scale_y=scale_y)
            sample["pre_image"] = pre_image
            sample["post_image"] = post_image
            sample["post_target"] = post_target
            sample["polygons"] = polygons
            sample["meta"] = {
                **sample["meta"],
                "crop_box": [0, 0, int(orig_w), int(orig_h)],
                "scale_xy": [scale_x, scale_y],
                "padded_size_hw": [int(orig_h), int(orig_w)],
                "output_size_hw": [int(post_target.shape[0]), int(post_target.shape[1])],
                "applied_hflip": False,
                "applied_vflip": False,
                "applied_rot90_k": 0,
                "validation_mode": val_mode,
            }
            return sample

        active_crop_size = self._sample_crop_size()
        pre_image = self._pad_to_crop(pre_image, active_crop_size, fill_value=0)
        post_image = self._pad_to_crop(post_image, active_crop_size, fill_value=0)
        post_target = self._pad_to_crop(post_target, active_crop_size, fill_value=0)

        padded_h, padded_w = post_target.shape[:2]
        crop_left, crop_top, crop_w, crop_h = self._random_or_center_crop(post_target, active_crop_size)

        pre_image = pre_image[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
        post_image = post_image[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
        post_target = post_target[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
        polygons = crop_polygons(polygons, crop_left, crop_top, crop_w, crop_h, min_area=self.min_polygon_area)

        aug_meta = {"applied_hflip": False, "applied_vflip": False, "applied_rot90_k": 0}
        if self.mode == "train":
            pre_image, post_image, post_target, polygons, aug_meta = self._apply_train_augmentations(
                pre_image=pre_image,
                post_image=post_image,
                post_target=post_target,
                polygons=polygons,
            )

        scale_x = 1.0
        scale_y = 1.0
        if tuple(self.image_size) != tuple(post_target.shape[:2]):
            scale_y = self.image_size[0] / float(post_target.shape[0])
            scale_x = self.image_size[1] / float(post_target.shape[1])
            pre_image = self._resize_rgb(pre_image, self.image_size)
            post_image = self._resize_rgb(post_image, self.image_size)
            post_target = self._resize_mask(post_target, self.image_size)
            polygons = scale_polygons(polygons, scale_x=scale_x, scale_y=scale_y)

        sample["pre_image"] = pre_image
        sample["post_image"] = post_image
        sample["post_target"] = post_target
        sample["polygons"] = polygons
        sample["meta"] = {
            **sample["meta"],
            "crop_box": [crop_left, crop_top, crop_w, crop_h],
            "scale_xy": [scale_x, scale_y],
            "padded_size_hw": [padded_h, padded_w],
            "output_size_hw": [int(post_target.shape[0]), int(post_target.shape[1])],
            **aug_meta,
            "validation_mode": val_mode if self.mode != "train" else "crop",
        }
        return sample

