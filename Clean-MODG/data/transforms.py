"""Synchronized augmentations for paired pre/post crops and masks."""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch

try:
    import albumentations as A  # type: ignore
except ImportError:  # pragma: no cover
    A = None


def _imagenet_stats(channels: int = 3) -> tuple[list[float], list[float]]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if channels == 4:
        return mean + [0.0], std + [1.0]
    return mean, std


class SimpleSyncTransform:
    def __init__(self, data_cfg: Dict[str, Any], is_train: bool) -> None:
        self.augment = bool(data_cfg.get("augment", False)) and is_train
        self.normalize = data_cfg.get("normalize", "imagenet") == "imagenet"
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _apply_spatial(self, array: np.ndarray, horizontal: bool, vertical: bool, rot_k: int) -> np.ndarray:
        if horizontal:
            array = np.flip(array, axis=1).copy()
        if vertical:
            array = np.flip(array, axis=0).copy()
        if rot_k > 0:
            array = np.rot90(array, k=rot_k).copy()
        return array

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        return image

    def __call__(self, **kwargs: np.ndarray) -> Dict[str, np.ndarray]:
        horizontal = self.augment and random.random() < 0.5
        vertical = self.augment and random.random() < 0.15
        rot_k = random.randint(0, 3) if self.augment and random.random() < 0.5 else 0
        outputs: Dict[str, np.ndarray] = {}
        for key, value in kwargs.items():
            transformed = self._apply_spatial(value, horizontal, vertical, rot_k)
            if key.startswith("mask"):
                outputs[key] = (transformed > 0).astype(np.uint8)
            else:
                outputs[key] = self._normalize(transformed)
        return outputs


def build_sync_transform(data_cfg: Dict[str, Any], is_train: bool):
    if A is None:
        return SimpleSyncTransform(data_cfg=data_cfg, is_train=is_train)
    augment = bool(data_cfg.get("augment", False)) and is_train
    ops: list[A.BasicTransform] = []
    if augment:
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.15),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=0,
                    interpolation=1,
                    mask_interpolation=0,
                    p=0.4,
                ),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            ]
        )
    mean, std = _imagenet_stats(channels=3)
    if data_cfg.get("normalize", "imagenet") == "imagenet":
        ops.append(A.Normalize(mean=mean, std=std))
    additional_targets = {
        "post_tight": "image",
        "pre_context": "image",
        "post_context": "image",
        "mask_tight": "mask",
        "mask_context": "mask",
    }
    return A.Compose(ops, additional_targets=additional_targets)


def apply_sync_transform(transform, crops: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    transformed = transform(
        image=crops["pre_tight"],
        post_tight=crops["post_tight"],
        pre_context=crops["pre_context"],
        post_context=crops["post_context"],
        mask_tight=crops["mask_tight"],
        mask_context=crops["mask_context"],
    )
    return {
        "pre_tight": transformed["image"],
        "post_tight": transformed["post_tight"],
        "pre_context": transformed["pre_context"],
        "post_context": transformed["post_context"],
        "mask_tight": transformed["mask_tight"],
        "mask_context": transformed["mask_context"],
    }


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return tensor


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, ...]
    else:
        mask = np.transpose(mask, (2, 0, 1))
    return torch.from_numpy(mask.astype(np.float32))
