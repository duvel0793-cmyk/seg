from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


@dataclass
class TransformConfig:
    image_size: int
    normalize_mean: list[float]
    normalize_std: list[float]
    hflip_prob: float
    vflip_prob: float
    rotate90_prob: float
    color_jitter_prob: float
    brightness: float
    contrast: float
    saturation: float


def _pil_to_tensor_rgb(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={array.shape}.")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _normalize(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def _apply_color_jitter(image: Image.Image, cfg: TransformConfig) -> Image.Image:
    if random.random() >= cfg.color_jitter_prob:
        return image
    if cfg.brightness > 0.0:
        factor = random.uniform(max(0.0, 1.0 - cfg.brightness), 1.0 + cfg.brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if cfg.contrast > 0.0:
        factor = random.uniform(max(0.0, 1.0 - cfg.contrast), 1.0 + cfg.contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if cfg.saturation > 0.0:
        factor = random.uniform(max(0.0, 1.0 - cfg.saturation), 1.0 + cfg.saturation)
        image = ImageEnhance.Color(image).enhance(factor)
    return image


class SharedResizeAugment:
    def __init__(self, *, config: dict[str, Any], is_train: bool) -> None:
        aug_cfg = config.get("augmentation", {})
        dataset_cfg = config.get("dataset", {})
        self.cfg = TransformConfig(
            image_size=int(dataset_cfg.get("image_size", 512)),
            normalize_mean=list(aug_cfg.get("normalize_mean", [0.485, 0.456, 0.406])),
            normalize_std=list(aug_cfg.get("normalize_std", [0.229, 0.224, 0.225])),
            hflip_prob=float(aug_cfg.get("hflip_prob", 0.5 if is_train else 0.0)),
            vflip_prob=float(aug_cfg.get("vflip_prob", 0.0 if not is_train else 0.1)),
            rotate90_prob=float(aug_cfg.get("rotate90_prob", 0.5 if is_train else 0.0)),
            color_jitter_prob=float(aug_cfg.get("color_jitter_prob", 0.3 if is_train else 0.0)),
            brightness=float(aug_cfg.get("brightness", 0.12)),
            contrast=float(aug_cfg.get("contrast", 0.12)),
            saturation=float(aug_cfg.get("saturation", 0.08)),
        )
        self.is_train = bool(is_train)

    def _sample_geometry(self) -> dict[str, Any]:
        if not self.is_train:
            return {"hflip": False, "vflip": False, "rotate_k": 0}
        return {
            "hflip": random.random() < self.cfg.hflip_prob,
            "vflip": random.random() < self.cfg.vflip_prob,
            "rotate_k": random.choice([0, 1, 2, 3]) if random.random() < self.cfg.rotate90_prob else 0,
        }

    def _apply_geometry_pil(self, image: Image.Image, geometry: dict[str, Any], *, is_mask: bool) -> Image.Image:
        if geometry["hflip"]:
            image = ImageOps.mirror(image)
        if geometry["vflip"]:
            image = ImageOps.flip(image)
        rotate_k = int(geometry["rotate_k"])
        if rotate_k:
            image = image.transpose(
                {
                    1: Image.Transpose.ROTATE_90,
                    2: Image.Transpose.ROTATE_180,
                    3: Image.Transpose.ROTATE_270,
                }[rotate_k]
            )
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        return image.resize((self.cfg.image_size, self.cfg.image_size), resample=resample)

    def __call__(
        self,
        *,
        pre_image: Image.Image,
        post_image: Image.Image,
        instance_masks: list[np.ndarray],
        damage_map: np.ndarray | None,
    ) -> dict[str, Any]:
        geometry = self._sample_geometry()
        pre_aug = self._apply_geometry_pil(pre_image, geometry, is_mask=False)
        post_aug = self._apply_geometry_pil(post_image, geometry, is_mask=False)
        pre_aug = _apply_color_jitter(pre_aug, self.cfg)
        post_aug = _apply_color_jitter(post_aug, self.cfg)

        transformed_masks: list[np.ndarray] = []
        for mask in instance_masks:
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            mask_aug = self._apply_geometry_pil(mask_image, geometry, is_mask=True)
            transformed_masks.append((np.asarray(mask_aug, dtype=np.uint8) > 127).astype(np.uint8))

        transformed_damage_map = None
        if damage_map is not None:
            damage_pil = Image.fromarray(damage_map.astype(np.uint8), mode="L")
            damage_aug = self._apply_geometry_pil(damage_pil, geometry, is_mask=True)
            transformed_damage_map = np.asarray(damage_aug, dtype=np.uint8)

        return {
            "pre_image": _normalize(_pil_to_tensor_rgb(pre_aug), self.cfg.normalize_mean, self.cfg.normalize_std),
            "post_image": _normalize(_pil_to_tensor_rgb(post_aug), self.cfg.normalize_mean, self.cfg.normalize_std),
            "instance_masks": transformed_masks,
            "damage_map": transformed_damage_map,
            "geometry": geometry,
        }


def build_transforms(config: dict[str, Any], is_train: bool) -> SharedResizeAugment:
    return SharedResizeAugment(config=config, is_train=is_train)
