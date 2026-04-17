from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageFilter
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
        context_dropout_prob: float = 0.0,
        context_blur_prob: float = 0.0,
        context_grayscale_prob: float = 0.0,
        context_noise_prob: float = 0.0,
        context_mix_prob: float = 0.0,
        context_edge_soften_pixels: int = 4,
        context_dilate_pixels: int = 3,
        context_apply_to_pre_and_post_independently: bool = False,
        context_preserve_instance_strictly: bool = True,
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
        self.context_dropout_prob = float(context_dropout_prob)
        self.context_blur_prob = float(context_blur_prob)
        self.context_grayscale_prob = float(context_grayscale_prob)
        self.context_noise_prob = float(context_noise_prob)
        self.context_mix_prob = float(context_mix_prob)
        self.context_edge_soften_pixels = int(max(context_edge_soften_pixels, 0))
        self.context_dilate_pixels = int(max(context_dilate_pixels, 0))
        self.context_apply_to_pre_and_post_independently = bool(context_apply_to_pre_and_post_independently)
        self.context_preserve_instance_strictly = bool(context_preserve_instance_strictly)
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

    def _build_context_alpha(self, mask: Image.Image) -> np.ndarray | None:
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        if int(mask_array.sum()) <= 0:
            return None

        protected_mask = Image.fromarray(mask_array * 255, mode="L")
        if self.context_dilate_pixels > 0:
            kernel_size = max(1, (self.context_dilate_pixels * 2) + 1)
            protected_mask = protected_mask.filter(ImageFilter.MaxFilter(size=kernel_size))

        context_alpha = 255 - np.asarray(protected_mask, dtype=np.uint8)
        alpha_image = Image.fromarray(context_alpha, mode="L")
        if self.context_edge_soften_pixels > 0:
            alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(radius=float(self.context_edge_soften_pixels)))

        alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)
        if self.context_preserve_instance_strictly:
            alpha[mask_array > 0] = 0.0
        if float(alpha.mean()) <= 1e-6:
            return None
        return alpha

    def _blend_context(
        self,
        source_image: Image.Image,
        transformed_image: Image.Image,
        alpha: np.ndarray,
        mask: Image.Image,
    ) -> Image.Image:
        source_array = np.asarray(source_image, dtype=np.float32)
        transformed_array = np.asarray(transformed_image, dtype=np.float32)
        blend_alpha = alpha[..., None]
        blended = (1.0 - blend_alpha) * source_array + (blend_alpha * transformed_array)
        if self.context_preserve_instance_strictly:
            mask_array = (np.asarray(mask, dtype=np.uint8) > 0)
            blended[mask_array] = source_array[mask_array]
        return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")

    def _make_context_dropout_image(self, image: Image.Image, stronger: bool) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32)
        fill_mode = random.choice(["mean", "black", "noise"])
        if fill_mode == "mean":
            fill_value = image_array.reshape(-1, image_array.shape[-1]).mean(axis=0, keepdims=True)
            fill_array = np.broadcast_to(fill_value.reshape(1, 1, 3), image_array.shape)
        elif fill_mode == "black":
            fill_array = np.zeros_like(image_array)
        else:
            noise_std = 40.0 if stronger else 28.0
            fill_array = np.random.normal(loc=127.5, scale=noise_std, size=image_array.shape).astype(np.float32)
        return Image.fromarray(np.clip(fill_array, 0.0, 255.0).astype(np.uint8), mode="RGB")

    def _make_context_noise_image(self, image: Image.Image, stronger: bool) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32)
        noise_std = 14.0 if stronger else 8.0
        noisy = image_array + np.random.normal(loc=0.0, scale=noise_std, size=image_array.shape).astype(np.float32)
        return Image.fromarray(np.clip(noisy, 0.0, 255.0).astype(np.uint8), mode="RGB")

    def _resolve_context_prob(self, base_probability: float, extra_context_prob: float) -> float:
        return float(min(max(base_probability + extra_context_prob, 0.0), 1.0))

    def _apply_context_ops_to_image(
        self,
        image: Image.Image,
        counterpart: Image.Image,
        mask: Image.Image,
        alpha: np.ndarray | None,
        operation_flags: dict[str, bool],
        *,
        stronger: bool,
    ) -> Image.Image:
        if alpha is None:
            return image

        current_image = image
        if operation_flags["mix"]:
            mix_ratio = random.uniform(0.30, 0.60 if stronger else 0.50)
            mixed = Image.blend(current_image, counterpart, alpha=mix_ratio)
            current_image = self._blend_context(current_image, mixed, alpha, mask)
        if operation_flags["grayscale"]:
            saturation_factor = 0.10 if stronger else 0.20
            desaturated = TF.adjust_saturation(current_image, saturation_factor=saturation_factor)
            current_image = self._blend_context(current_image, desaturated, alpha, mask)
        if operation_flags["blur"]:
            blur_radius = 2.5 if stronger else 1.5
            blurred = current_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            current_image = self._blend_context(current_image, blurred, alpha, mask)
        if operation_flags["dropout"]:
            dropout_image = self._make_context_dropout_image(current_image, stronger=stronger)
            current_image = self._blend_context(current_image, dropout_image, alpha, mask)
        if operation_flags["noise"]:
            noise_image = self._make_context_noise_image(current_image, stronger=stronger)
            current_image = self._blend_context(current_image, noise_image, alpha, mask)
        return current_image

    def _sample_context_operation_flags(
        self,
        *,
        extra_context_prob: float,
    ) -> dict[str, bool]:
        return {
            "dropout": random.random() < self._resolve_context_prob(self.context_dropout_prob, extra_context_prob),
            "blur": random.random() < self._resolve_context_prob(self.context_blur_prob, extra_context_prob),
            "grayscale": random.random() < self._resolve_context_prob(self.context_grayscale_prob, extra_context_prob),
            "noise": random.random() < self._resolve_context_prob(self.context_noise_prob, extra_context_prob),
            "mix": random.random() < self._resolve_context_prob(self.context_mix_prob, extra_context_prob),
        }

    def _apply_context_debias(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
        *,
        extra_context_prob: float = 0.0,
    ) -> tuple[Image.Image, Image.Image, dict[str, float]]:
        alpha = self._build_context_alpha(mask)
        stats = {
            "context_applied": 0.0,
            "context_dropout": 0.0,
            "context_blur": 0.0,
            "context_grayscale": 0.0,
            "context_noise": 0.0,
            "context_mix": 0.0,
            "context_alpha_mean": 0.0 if alpha is None else float(alpha.mean()),
            "context_alpha_max": 0.0 if alpha is None else float(alpha.max()),
            "context_extra_prob": float(extra_context_prob),
        }
        if alpha is None:
            return pre, post, stats

        stronger = float(extra_context_prob) > 0.0
        if self.context_apply_to_pre_and_post_independently:
            pre_flags = self._sample_context_operation_flags(extra_context_prob=extra_context_prob)
            post_flags = self._sample_context_operation_flags(extra_context_prob=extra_context_prob)
        else:
            shared_flags = self._sample_context_operation_flags(extra_context_prob=extra_context_prob)
            pre_flags = shared_flags
            post_flags = shared_flags

        pre = self._apply_context_ops_to_image(pre, post, mask, alpha, pre_flags, stronger=stronger)
        post = self._apply_context_ops_to_image(post, pre, mask, alpha, post_flags, stronger=stronger)

        for key in ["dropout", "blur", "grayscale", "noise", "mix"]:
            stats[f"context_{key}"] = float(pre_flags[key] or post_flags[key])
        stats["context_applied"] = float(any(value > 0.0 for key, value in stats.items() if key.startswith("context_") and key not in {"context_alpha_mean", "context_alpha_max", "context_extra_prob"}))
        return pre, post, stats

    def _to_tensor(self, pre: Image.Image, post: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_t = TF.to_tensor(pre)
        post_t = TF.to_tensor(post)
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        pre_t = TF.normalize(pre_t, mean=self.normalize_mean, std=self.normalize_std)
        post_t = TF.normalize(post_t, mean=self.normalize_mean, std=self.normalize_std)
        return pre_t, post_t, mask_t

    def prepare_augmented_pair(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
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

        return pre, post, mask

    def transform_prepared_pair(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
        *,
        extra_context_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        pre, post, context_stats = self._apply_context_debias(
            pre,
            post,
            mask,
            extra_context_prob=extra_context_prob,
        )
        pre_t, post_t, mask_t = self._to_tensor(pre, post, mask)
        return pre_t, post_t, mask_t, context_stats

    def make_dual_views(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
        *,
        extra_context_prob: float = 0.0,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]],
    ]:
        prepared_pre, prepared_post, prepared_mask = self.prepare_augmented_pair(pre, post, mask)
        view1 = self.transform_prepared_pair(prepared_pre, prepared_post, prepared_mask, extra_context_prob=0.0)
        # The second view reuses the same geometric crop and mask, then exposes less context.
        view2 = self.transform_prepared_pair(
            prepared_pre.copy(),
            prepared_post.copy(),
            prepared_mask.copy(),
            extra_context_prob=extra_context_prob,
        )
        return view1, view2

    def __call__(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
        *,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        prepared_pre, prepared_post, prepared_mask = self.prepare_augmented_pair(pre, post, mask)
        pre_t, post_t, mask_t, context_stats = self.transform_prepared_pair(prepared_pre, prepared_post, prepared_mask)
        if return_stats:
            return pre_t, post_t, mask_t, context_stats
        return pre_t, post_t, mask_t


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
            context_dropout_prob=float(aug_cfg.get("context_dropout_prob", 0.0)),
            context_blur_prob=float(aug_cfg.get("context_blur_prob", 0.0)),
            context_grayscale_prob=float(aug_cfg.get("context_grayscale_prob", 0.0)),
            context_noise_prob=float(aug_cfg.get("context_noise_prob", 0.0)),
            context_mix_prob=float(aug_cfg.get("context_mix_prob", 0.0)),
            context_edge_soften_pixels=int(aug_cfg.get("context_edge_soften_pixels", 4)),
            context_dilate_pixels=int(aug_cfg.get("context_dilate_pixels", 3)),
            context_apply_to_pre_and_post_independently=bool(
                aug_cfg.get("context_apply_to_pre_and_post_independently", False)
            ),
            context_preserve_instance_strictly=bool(aug_cfg.get("context_preserve_instance_strictly", True)),
            normalize_mean=aug_cfg["normalize_mean"],
            normalize_std=aug_cfg["normalize_std"],
        )
    return EvalPairTransform(
        image_size=image_size,
        normalize_mean=aug_cfg["normalize_mean"],
        normalize_std=aug_cfg["normalize_std"],
    )
