from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils.misc import SCALE_NAMES, get_enabled_scale_names


class MultiScaleTransform:
    def __init__(
        self,
        *,
        scale_names: list[str],
        image_sizes: dict[str, int],
        normalize_mean: list[float],
        normalize_std: list[float],
        is_train: bool,
        aug_cfg: dict[str, Any],
    ) -> None:
        self.scale_names = list(scale_names)
        self.image_sizes = {name: int(image_sizes[name]) for name in self.scale_names}
        self.mean = list(normalize_mean)
        self.std = list(normalize_std)
        self.is_train = bool(is_train)
        self.aug_cfg = dict(aug_cfg)

    def _sample_geometry(self) -> dict[str, Any]:
        if not self.is_train:
            return {"hflip": False, "vflip": False, "rotate": 0, "use_crop": False}
        use_crop = random.random() < float(self.aug_cfg.get("random_resized_crop_prob", 0.0))
        crop_scale = random.uniform(*self.aug_cfg.get("random_resized_crop_scale", [0.9, 1.0])) if use_crop else 1.0
        crop_ratio = random.uniform(*self.aug_cfg.get("random_resized_crop_ratio", [0.95, 1.05])) if use_crop else 1.0
        crop_h = min(1.0, math.sqrt(crop_scale / max(crop_ratio, 1e-6)))
        crop_w = min(1.0, math.sqrt(crop_scale * max(crop_ratio, 1e-6)))
        return {
            "hflip": random.random() < float(self.aug_cfg.get("hflip_prob", 0.0)),
            "vflip": random.random() < float(self.aug_cfg.get("vflip_prob", 0.0)),
            "rotate": random.choice([0, 90, 180, 270]) if random.random() < float(self.aug_cfg.get("rotate90_prob", 0.0)) else 0,
            "use_crop": use_crop,
            "crop_top": 0.0 if crop_h >= 1.0 else random.uniform(0.0, 1.0 - crop_h),
            "crop_left": 0.0 if crop_w >= 1.0 else random.uniform(0.0, 1.0 - crop_w),
            "crop_h": crop_h,
            "crop_w": crop_w,
        }

    def _apply_geometry(self, image: Image.Image, geometry: dict[str, Any], interpolation: InterpolationMode) -> Image.Image:
        if geometry["hflip"]:
            image = TF.hflip(image)
        if geometry["vflip"]:
            image = TF.vflip(image)
        if int(geometry["rotate"]) != 0:
            image = TF.rotate(image, int(geometry["rotate"]), interpolation=interpolation)
        if geometry["use_crop"]:
            width, height = image.size
            crop_h = max(1, min(height, int(round(height * float(geometry["crop_h"])))))
            crop_w = max(1, min(width, int(round(width * float(geometry["crop_w"])))))
            top = int(round(float(geometry["crop_top"]) * max(height - crop_h, 0)))
            left = int(round(float(geometry["crop_left"]) * max(width - crop_w, 0)))
            image = TF.crop(image, top, left, crop_h, crop_w)
        return image

    def _resize(self, image: Image.Image, scale_name: str, interpolation: InterpolationMode) -> Image.Image:
        size = [self.image_sizes[scale_name], self.image_sizes[scale_name]]
        return TF.resize(image, size, interpolation=interpolation)

    def _maybe_color_jitter(self, pre: Image.Image, post: Image.Image) -> tuple[Image.Image, Image.Image]:
        if (not self.is_train) or random.random() >= float(self.aug_cfg.get("color_jitter_prob", 0.0)):
            return pre, post
        brightness = float(self.aug_cfg.get("brightness", 0.0))
        contrast = float(self.aug_cfg.get("contrast", 0.0))
        saturation = float(self.aug_cfg.get("saturation", 0.0))
        hue = float(self.aug_cfg.get("hue", 0.0))
        b = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        c = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        s = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
        h = random.uniform(-hue, hue)
        ops = [
            lambda img: TF.adjust_brightness(img, b),
            lambda img: TF.adjust_contrast(img, c),
            lambda img: TF.adjust_saturation(img, s),
            lambda img: TF.adjust_hue(img, h),
        ]
        for op in ops:
            pre = op(pre)
            post = op(post)
        return pre, post

    def prepare_sample(self, crops: dict[str, dict[str, Image.Image]]) -> tuple[dict[str, dict[str, Image.Image]], dict[str, float]]:
        geometry = self._sample_geometry()
        prepared: dict[str, dict[str, Image.Image]] = {}
        for scale_name in self.scale_names:
            sample = crops[scale_name]
            pre = self._apply_geometry(sample["pre"], geometry, InterpolationMode.BILINEAR)
            post = self._apply_geometry(sample["post"], geometry, InterpolationMode.BILINEAR)
            mask = self._apply_geometry(sample["mask"], geometry, InterpolationMode.NEAREST)
            pre, post = self._maybe_color_jitter(pre, post)
            pre = self._resize(pre, scale_name, InterpolationMode.BILINEAR)
            post = self._resize(post, scale_name, InterpolationMode.BILINEAR)
            mask = self._resize(mask, scale_name, InterpolationMode.NEAREST)
            output = {"pre": pre, "post": post, "mask": mask}
            if "post_target" in sample:
                output["post_target"] = self._resize(
                    self._apply_geometry(sample["post_target"], geometry, InterpolationMode.NEAREST),
                    scale_name,
                    InterpolationMode.NEAREST,
                )
            prepared[scale_name] = output
        stats = {
            "shared_hflip": float(geometry["hflip"]),
            "shared_vflip": float(geometry["vflip"]),
            "shared_rotate90": float(int(geometry["rotate"]) != 0),
            "shared_random_resized_crop": float(geometry["use_crop"]),
        }
        return prepared, stats

    def to_tensor_dict(self, prepared: dict[str, dict[str, Image.Image]], ignore_index: int) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for scale_name in self.scale_names:
            sample = prepared[scale_name]
            pre = TF.normalize(TF.to_tensor(sample["pre"]), mean=self.mean, std=self.std)
            post = TF.normalize(TF.to_tensor(sample["post"]), mean=self.mean, std=self.std)
            mask = (TF.to_tensor(sample["mask"]) > 0.5).float()
            tensors[f"pre_{scale_name}"] = pre
            tensors[f"post_{scale_name}"] = post
            tensors[f"mask_{scale_name}"] = mask
            if "post_target" in sample:
                arr = np.asarray(sample["post_target"], dtype=np.uint8)
                arr = arr.copy()
                arr[arr == 0] = ignore_index
                positive = arr != ignore_index
                arr[positive] = arr[positive] - 1
                mask_arr = np.asarray(sample["mask"], dtype=np.uint8) > 0
                arr[~mask_arr] = ignore_index
                tensors[f"post_target_{scale_name}"] = torch.from_numpy(arr).long()
        return tensors


def build_transforms(config: dict[str, Any], is_train: bool) -> MultiScaleTransform:
    crop_cfg = config["dataset"]["crop_scales"]
    scale_names = get_enabled_scale_names(config)
    image_sizes = {name: int(crop_cfg[name]["output_size"]) for name in scale_names}
    aug_cfg = config.get("augmentation", {})
    return MultiScaleTransform(
        scale_names=scale_names,
        image_sizes=image_sizes,
        normalize_mean=list(aug_cfg.get("normalize_mean", [0.485, 0.456, 0.406])),
        normalize_std=list(aug_cfg.get("normalize_std", [0.229, 0.224, 0.225])),
        is_train=is_train,
        aug_cfg=aug_cfg,
    )
