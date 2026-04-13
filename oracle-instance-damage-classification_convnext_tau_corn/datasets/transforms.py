"""Synchronized transforms for pre/post images and binary masks."""

from __future__ import annotations

import random
from typing import Tuple

import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _apply_shared_color_jitter(pre_image: Image.Image, post_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    brightness = random.uniform(0.9, 1.1)
    contrast = random.uniform(0.9, 1.1)
    saturation = random.uniform(0.9, 1.1)

    for enhancer_cls, factor in (
        (ImageEnhance.Brightness, brightness),
        (ImageEnhance.Contrast, contrast),
        (ImageEnhance.Color, saturation),
    ):
        pre_image = enhancer_cls(pre_image).enhance(factor)
        post_image = enhancer_cls(post_image).enhance(factor)

    return pre_image, post_image


def _to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(mask)
    return (tensor > 0.5).float()


class TrainTransform:
    """Random but synchronized augmentation for image pairs and masks."""

    def __init__(self, image_size: int, use_vertical_flip: bool = False, use_color_jitter: bool = True) -> None:
        self.image_size = int(image_size)
        self.use_vertical_flip = use_vertical_flip
        self.use_color_jitter = use_color_jitter

    def __call__(self, pre_image: Image.Image, post_image: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            pre_image = TF.hflip(pre_image)
            post_image = TF.hflip(post_image)
            mask = TF.hflip(mask)

        if self.use_vertical_flip and random.random() < 0.5:
            pre_image = TF.vflip(pre_image)
            post_image = TF.vflip(post_image)
            mask = TF.vflip(mask)

        if self.use_color_jitter:
            pre_image, post_image = _apply_shared_color_jitter(pre_image, post_image)

        pre_image = TF.resize(pre_image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        post_image = TF.resize(post_image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        return _to_tensor(pre_image), _to_tensor(post_image), _mask_to_tensor(mask)


class EvalTransform:
    """Deterministic resize and tensor conversion."""

    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def __call__(self, pre_image: Image.Image, post_image: Image.Image, mask: Image.Image):
        pre_image = TF.resize(pre_image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        post_image = TF.resize(post_image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        return _to_tensor(pre_image), _to_tensor(post_image), _mask_to_tensor(mask)

