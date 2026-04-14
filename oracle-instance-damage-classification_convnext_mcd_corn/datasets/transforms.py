from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainPairTransform:
    def __init__(
        self,
        image_size: int = 224,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        random_resized_crop_prob: float = 0.35,
        random_resized_crop_scale: tuple[float, float] = (0.9, 1.0),
        random_resized_crop_ratio: tuple[float, float] = (0.95, 1.05),
        min_mask_retention: float = 0.7,
        color_jitter_prob: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.1,
        hue: float = 0.02,
        normalize_mean: tuple[float, float, float] = tuple(IMAGENET_MEAN),
        normalize_std: tuple[float, float, float] = tuple(IMAGENET_STD),
    ) -> None:
        self.image_size = int(image_size)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate90_prob = float(rotate90_prob)
        self.random_resized_crop_prob = float(random_resized_crop_prob)
        self.random_resized_crop_scale = tuple(random_resized_crop_scale)
        self.random_resized_crop_ratio = tuple(random_resized_crop_ratio)
        self.min_mask_retention = float(min_mask_retention)
        self.color_jitter_prob = float(color_jitter_prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
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
        mask_np = np.asarray(mask, dtype=np.uint8)
        original_pixels = float(mask_np.sum())
        if original_pixels <= 0:
            return self._resize(pre, post, mask)

        for _ in range(10):
            top, left, height, width = RandomResizedCrop.get_params(
                pre,
                scale=self.random_resized_crop_scale,
                ratio=self.random_resized_crop_ratio,
            )
            cropped_mask = TF.crop(mask, top, left, height, width)
            retained = float(np.asarray(cropped_mask, dtype=np.uint8).sum()) / max(original_pixels, 1.0)
            if retained >= self.min_mask_retention:
                size = [self.image_size, self.image_size]
                pre = TF.resized_crop(pre, top, left, height, width, size=size, interpolation=InterpolationMode.BILINEAR, antialias=True)
                post = TF.resized_crop(post, top, left, height, width, size=size, interpolation=InterpolationMode.BILINEAR, antialias=True)
                mask = TF.resized_crop(mask, top, left, height, width, size=size, interpolation=InterpolationMode.NEAREST)
                return pre, post, mask
        return self._resize(pre, post, mask)

    def _apply_shared_color_jitter(self, pre: Image.Image, post: Image.Image) -> tuple[Image.Image, Image.Image]:
        brightness = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
        contrast = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
        saturation = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
        hue = random.uniform(-self.hue, self.hue)
        for fn in (
            lambda img: TF.adjust_brightness(img, brightness),
            lambda img: TF.adjust_contrast(img, contrast),
            lambda img: TF.adjust_saturation(img, saturation),
            lambda img: TF.adjust_hue(img, hue),
        ):
            pre = fn(pre)
            post = fn(post)
        return pre, post

    def _to_tensors(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_tensor = TF.to_tensor(pre)
        post_tensor = TF.to_tensor(post)
        pre_tensor = TF.normalize(pre_tensor, self.normalize_mean, self.normalize_std)
        post_tensor = TF.normalize(post_tensor, self.normalize_mean, self.normalize_std)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return pre_tensor, post_tensor, mask_tensor

    def __call__(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.random_resized_crop_prob:
            pre, post, mask = self._safe_random_resized_crop(pre, post, mask)
        else:
            pre, post, mask = self._resize(pre, post, mask)

        if random.random() < self.hflip_prob:
            pre = TF.hflip(pre)
            post = TF.hflip(post)
            mask = TF.hflip(mask)
        if random.random() < self.vflip_prob:
            pre = TF.vflip(pre)
            post = TF.vflip(post)
            mask = TF.vflip(mask)
        if random.random() < self.rotate90_prob:
            k = random.randint(1, 3)
            angle = 90 * k
            pre = TF.rotate(pre, angle, interpolation=InterpolationMode.BILINEAR)
            post = TF.rotate(post, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        if random.random() < self.color_jitter_prob:
            pre, post = self._apply_shared_color_jitter(pre, post)
        return self._to_tensors(pre, post, mask)


class EvalPairTransform:
    def __init__(
        self,
        image_size: int = 224,
        normalize_mean: tuple[float, float, float] = tuple(IMAGENET_MEAN),
        normalize_std: tuple[float, float, float] = tuple(IMAGENET_STD),
    ) -> None:
        self.image_size = int(image_size)
        self.normalize_mean = list(normalize_mean)
        self.normalize_std = list(normalize_std)

    def __call__(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = [self.image_size, self.image_size]
        pre = TF.resize(pre, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        post = TF.resize(post, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        pre_tensor = TF.normalize(TF.to_tensor(pre), self.normalize_mean, self.normalize_std)
        post_tensor = TF.normalize(TF.to_tensor(post), self.normalize_mean, self.normalize_std)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return pre_tensor, post_tensor, mask_tensor


def build_transforms(config: dict[str, Any], is_train: bool) -> TrainPairTransform | EvalPairTransform:
    image_size = int(config["data"]["image_size"])
    aug_cfg = config.get("augmentation", {})
    normalize_mean = tuple(aug_cfg.get("normalize_mean", IMAGENET_MEAN))
    normalize_std = tuple(aug_cfg.get("normalize_std", IMAGENET_STD))

    if is_train:
        return TrainPairTransform(
            image_size=image_size,
            hflip_prob=float(aug_cfg.get("hflip_prob", 0.5)),
            vflip_prob=float(aug_cfg.get("vflip_prob", 0.5)),
            rotate90_prob=float(aug_cfg.get("rotate90_prob", 0.5)),
            random_resized_crop_prob=float(aug_cfg.get("random_resized_crop_prob", 0.35)),
            random_resized_crop_scale=tuple(aug_cfg.get("random_resized_crop_scale", [0.9, 1.0])),
            random_resized_crop_ratio=tuple(aug_cfg.get("random_resized_crop_ratio", [0.95, 1.05])),
            min_mask_retention=float(aug_cfg.get("min_mask_retention", 0.7)),
            color_jitter_prob=float(aug_cfg.get("color_jitter_prob", 0.5)),
            brightness=float(aug_cfg.get("brightness", 0.15)),
            contrast=float(aug_cfg.get("contrast", 0.15)),
            saturation=float(aug_cfg.get("saturation", 0.1)),
            hue=float(aug_cfg.get("hue", 0.02)),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
    return EvalPairTransform(image_size=image_size, normalize_mean=normalize_mean, normalize_std=normalize_std)

