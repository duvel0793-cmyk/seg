from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _sample_unit_interval(center: float, amount: float, minimum: float = 0.0) -> float:
    low = max(minimum, center - amount)
    high = center + amount
    return random.uniform(low, high)


class TrainPairTransform:
    def __init__(
        self,
        image_size: int = 224,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        random_resized_crop_prob: float = 0.5,
        random_resized_crop_scale: tuple[float, float] = (0.9, 1.0),
        random_resized_crop_ratio: tuple[float, float] = (0.95, 1.05),
        min_mask_retention: float = 0.75,
        color_jitter_prob: float = 0.6,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.1,
        hue: float = 0.02,
        normalize_mean: list[float] | tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: list[float] | tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90_prob = rotate90_prob
        self.random_resized_crop_prob = random_resized_crop_prob
        self.random_resized_crop_scale = random_resized_crop_scale
        self.random_resized_crop_ratio = random_resized_crop_ratio
        self.min_mask_retention = min_mask_retention
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.normalize_mean = list(normalize_mean)
        self.normalize_std = list(normalize_std)

    def _resize(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
        size = [self.image_size, self.image_size]
        pre = TF.resize(pre, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        post = TF.resize(post, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return pre, post, mask

    def _safe_random_resized_crop(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        mask_arr = np.asarray(mask, dtype=np.uint8)
        original_mask_pixels = float(mask_arr.sum())
        if original_mask_pixels <= 0:
            return self._resize(pre, post, mask)

        for _ in range(10):
            top, left, height, width = RandomResizedCrop.get_params(
                pre,
                scale=self.random_resized_crop_scale,
                ratio=self.random_resized_crop_ratio,
            )
            cropped_mask = TF.crop(mask, top, left, height, width)
            retained = float(np.asarray(cropped_mask, dtype=np.uint8).sum()) / max(original_mask_pixels, 1.0)
            if retained >= self.min_mask_retention:
                size = [self.image_size, self.image_size]
                pre_out = TF.resized_crop(
                    pre, top, left, height, width, size=size, interpolation=InterpolationMode.BILINEAR, antialias=True
                )
                post_out = TF.resized_crop(
                    post, top, left, height, width, size=size, interpolation=InterpolationMode.BILINEAR, antialias=True
                )
                mask_out = TF.resized_crop(
                    mask, top, left, height, width, size=size, interpolation=InterpolationMode.NEAREST
                )
                return pre_out, post_out, mask_out

        return self._resize(pre, post, mask)

    def _apply_shared_color_jitter(self, pre: Image.Image, post: Image.Image) -> tuple[Image.Image, Image.Image]:
        brightness_factor = _sample_unit_interval(1.0, self.brightness)
        contrast_factor = _sample_unit_interval(1.0, self.contrast)
        saturation_factor = _sample_unit_interval(1.0, self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        for fn in (
            lambda img: TF.adjust_brightness(img, brightness_factor),
            lambda img: TF.adjust_contrast(img, contrast_factor),
            lambda img: TF.adjust_saturation(img, saturation_factor),
            lambda img: TF.adjust_hue(img, hue_factor),
        ):
            pre = fn(pre)
            post = fn(post)
        return pre, post

    def _to_tensor(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_t = TF.to_tensor(pre)
        post_t = TF.to_tensor(post)
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        pre_t = TF.normalize(pre_t, mean=self.normalize_mean, std=self.normalize_std)
        post_t = TF.normalize(post_t, mean=self.normalize_mean, std=self.normalize_std)
        return pre_t, post_t, mask_t

    def __call__(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.hflip_prob:
            pre = TF.hflip(pre)
            post = TF.hflip(post)
            mask = TF.hflip(mask)

        if random.random() < self.vflip_prob:
            pre = TF.vflip(pre)
            post = TF.vflip(post)
            mask = TF.vflip(mask)

        if random.random() < self.rotate90_prob:
            angle = random.choice([0, 90, 180, 270])
            pre = TF.rotate(pre, angle=angle, interpolation=InterpolationMode.BILINEAR)
            post = TF.rotate(post, angle=angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < self.random_resized_crop_prob:
            pre, post, mask = self._safe_random_resized_crop(pre, post, mask)
        else:
            pre, post, mask = self._resize(pre, post, mask)

        if random.random() < self.color_jitter_prob:
            pre, post = self._apply_shared_color_jitter(pre, post)

        return self._to_tensor(pre, post, mask)


class EvalPairTransform:
    def __init__(
        self,
        image_size: int = 224,
        normalize_mean: list[float] | tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: list[float] | tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.image_size = image_size
        self.normalize_mean = list(normalize_mean)
        self.normalize_std = list(normalize_std)

    def __call__(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = [self.image_size, self.image_size]
        pre = TF.resize(pre, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        post = TF.resize(post, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        pre_t = TF.normalize(TF.to_tensor(pre), mean=self.normalize_mean, std=self.normalize_std)
        post_t = TF.normalize(TF.to_tensor(post), mean=self.normalize_mean, std=self.normalize_std)
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        return pre_t, post_t, mask_t


def build_transforms(config: dict, is_train: bool) -> TrainPairTransform | EvalPairTransform:
    image_size = int(config["data"]["image_size"])
    aug_cfg = config["augmentation"]
    if is_train:
        return TrainPairTransform(
            image_size=image_size,
            hflip_prob=float(aug_cfg["hflip_prob"]),
            vflip_prob=float(aug_cfg["vflip_prob"]),
            rotate90_prob=float(aug_cfg["rotate90_prob"]),
            random_resized_crop_prob=float(aug_cfg["random_resized_crop_prob"]),
            random_resized_crop_scale=tuple(aug_cfg["random_resized_crop_scale"]),
            random_resized_crop_ratio=tuple(aug_cfg["random_resized_crop_ratio"]),
            min_mask_retention=float(aug_cfg["min_mask_retention"]),
            color_jitter_prob=float(aug_cfg["color_jitter_prob"]),
            brightness=float(aug_cfg["brightness"]),
            contrast=float(aug_cfg["contrast"]),
            saturation=float(aug_cfg["saturation"]),
            hue=float(aug_cfg["hue"]),
            normalize_mean=aug_cfg["normalize_mean"],
            normalize_std=aug_cfg["normalize_std"],
        )
    return EvalPairTransform(
        image_size=image_size,
        normalize_mean=aug_cfg["normalize_mean"],
        normalize_std=aug_cfg["normalize_std"],
    )
