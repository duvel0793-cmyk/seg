from __future__ import annotations

import math
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCALE_NAMES = ("tight", "context", "neighborhood")


def _sample_unit_interval(center: float, amount: float, minimum: float = 0.0) -> float:
    low = max(minimum, center - amount)
    high = center + amount
    return random.uniform(low, high)


class _BaseMultiContextTransform:
    def __init__(
        self,
        *,
        image_sizes: dict[str, int],
        normalize_mean: list[float] | tuple[float, float, float],
        normalize_std: list[float] | tuple[float, float, float],
    ) -> None:
        self.image_sizes = {name: int(image_sizes[name]) for name in SCALE_NAMES}
        self.normalize_mean = list(normalize_mean)
        self.normalize_std = list(normalize_std)

    def resize_triplet(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
        *,
        scale_name: str,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        size = [self.image_sizes[scale_name], self.image_sizes[scale_name]]
        pre = TF.resize(pre, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        post = TF.resize(post, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return pre, post, mask

    def resize_aux_image(
        self,
        image: Image.Image,
        *,
        scale_name: str,
        interpolation: str = "nearest",
    ) -> Image.Image:
        size = [self.image_sizes[scale_name], self.image_sizes[scale_name]]
        interpolation_mode = InterpolationMode.NEAREST if interpolation == "nearest" else InterpolationMode.BILINEAR
        antialias = interpolation_mode == InterpolationMode.BILINEAR
        return TF.resize(image, size, interpolation=interpolation_mode, antialias=antialias)

    def to_tensor_triplet(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_tensor = TF.normalize(TF.to_tensor(pre), mean=self.normalize_mean, std=self.normalize_std)
        post_tensor = TF.normalize(TF.to_tensor(post), mean=self.normalize_mean, std=self.normalize_std)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return pre_tensor, post_tensor, mask_tensor

    def to_tensor_dict(self, prepared: dict[str, dict[str, Image.Image]]) -> dict[str, torch.Tensor]:
        tensor_dict: dict[str, torch.Tensor] = {}
        for scale_name in SCALE_NAMES:
            pre_tensor, post_tensor, mask_tensor = self.to_tensor_triplet(
                prepared[scale_name]["pre"],
                prepared[scale_name]["post"],
                prepared[scale_name]["mask"],
            )
            if mask_tensor.sum().item() <= 0:
                mask_tensor = torch.ones_like(mask_tensor)
            tensor_dict[f"pre_{scale_name}"] = pre_tensor
            tensor_dict[f"post_{scale_name}"] = post_tensor
            tensor_dict[f"mask_{scale_name}"] = mask_tensor
            aux_images = prepared[scale_name].get("aux_images", {})
            aux_image_modes = prepared[scale_name].get("aux_image_modes", {})
            for key, image in aux_images.items():
                interpolation = str(aux_image_modes.get(key, "nearest"))
                tensor = TF.to_tensor(image)
                if interpolation == "nearest":
                    tensor = (tensor > 0.5).float()
                else:
                    tensor = tensor.float()
                tensor_dict[key] = tensor
            aux_stacks = prepared[scale_name].get("aux_stacks", {})
            aux_stack_modes = prepared[scale_name].get("aux_stack_modes", {})
            for key, images in aux_stacks.items():
                interpolation = str(aux_stack_modes.get(key, "nearest"))
                if not images:
                    tensor_dict[key] = torch.zeros(0, self.image_sizes[scale_name], self.image_sizes[scale_name], dtype=torch.float32)
                    continue
                stack_tensors = []
                for image in images:
                    tensor = TF.to_tensor(image)
                    if interpolation == "nearest":
                        tensor = (tensor > 0.5).float()
                    else:
                        tensor = tensor.float()
                    stack_tensors.append(tensor.squeeze(0))
                tensor_dict[key] = torch.stack(stack_tensors, dim=0)
        return tensor_dict


class TrainMultiContextTransform(_BaseMultiContextTransform):
    def __init__(
        self,
        *,
        image_sizes: dict[str, int],
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
        super().__init__(image_sizes=image_sizes, normalize_mean=normalize_mean, normalize_std=normalize_std)
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
        self.context_dropout_prob = float(context_dropout_prob)
        self.context_blur_prob = float(context_blur_prob)
        self.context_grayscale_prob = float(context_grayscale_prob)
        self.context_noise_prob = float(context_noise_prob)
        self.context_mix_prob = float(context_mix_prob)
        self.context_edge_soften_pixels = int(max(context_edge_soften_pixels, 0))
        self.context_dilate_pixels = int(max(context_dilate_pixels, 0))
        self.context_apply_to_pre_and_post_independently = bool(context_apply_to_pre_and_post_independently)
        self.context_preserve_instance_strictly = bool(context_preserve_instance_strictly)
        self.crop_strength = {"tight": 1.0, "context": 1.0, "neighborhood": 0.5}
        self.context_aug_strength = {"tight": 0.0, "context": 1.0, "neighborhood": 0.5}

    def _sample_geometry(self) -> dict[str, float | int | bool]:
        use_crop = random.random() < self.random_resized_crop_prob
        crop_scale = random.uniform(*self.random_resized_crop_scale) if use_crop else 1.0
        crop_ratio = random.uniform(*self.random_resized_crop_ratio) if use_crop else 1.0
        crop_h = min(1.0, math.sqrt(crop_scale / max(crop_ratio, 1e-6)))
        crop_w = min(1.0, math.sqrt(crop_scale * max(crop_ratio, 1e-6)))
        return {
            "hflip": random.random() < self.hflip_prob,
            "vflip": random.random() < self.vflip_prob,
            "rotate": random.choice([0, 90, 180, 270]) if random.random() < self.rotate90_prob else 0,
            "crop_top": 0.0 if crop_h >= 1.0 else random.uniform(0.0, 1.0 - crop_h),
            "crop_left": 0.0 if crop_w >= 1.0 else random.uniform(0.0, 1.0 - crop_w),
            "crop_h": crop_h,
            "crop_w": crop_w,
            "use_crop": use_crop,
        }

    def _apply_geometry(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
        *,
        scale_name: str,
        geometry: dict[str, float | int | bool],
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        if bool(geometry["hflip"]):
            pre = TF.hflip(pre)
            post = TF.hflip(post)
            mask = TF.hflip(mask)
        if bool(geometry["vflip"]):
            pre = TF.vflip(pre)
            post = TF.vflip(post)
            mask = TF.vflip(mask)
        angle = int(geometry["rotate"])
        if angle != 0:
            pre = TF.rotate(pre, angle=angle, interpolation=InterpolationMode.BILINEAR)
            post = TF.rotate(post, angle=angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST)

        if bool(geometry["use_crop"]):
            strength = self.crop_strength[scale_name]
            crop_h = 1.0 - (float(geometry["crop_h"]) - 1.0) * strength if float(geometry["crop_h"]) > 1.0 else 1.0 - (1.0 - float(geometry["crop_h"])) * strength
            crop_w = 1.0 - (float(geometry["crop_w"]) - 1.0) * strength if float(geometry["crop_w"]) > 1.0 else 1.0 - (1.0 - float(geometry["crop_w"])) * strength
            width, height = pre.size
            crop_height = max(1, min(height, int(round(height * crop_h))))
            crop_width = max(1, min(width, int(round(width * crop_w))))
            top = int(round(float(geometry["crop_top"]) * max(height - crop_height, 0)))
            left = int(round(float(geometry["crop_left"]) * max(width - crop_width, 0)))
            pre = TF.crop(pre, top, left, crop_height, crop_width)
            post = TF.crop(post, top, left, crop_height, crop_width)
            mask = TF.crop(mask, top, left, crop_height, crop_width)
        return self.resize_triplet(pre, post, mask, scale_name=scale_name)

    def _apply_geometry_to_aux_image(
        self,
        image: Image.Image,
        *,
        scale_name: str,
        geometry: dict[str, float | int | bool],
        interpolation: str = "nearest",
    ) -> Image.Image:
        if bool(geometry["hflip"]):
            image = TF.hflip(image)
        if bool(geometry["vflip"]):
            image = TF.vflip(image)
        angle = int(geometry["rotate"])
        if angle != 0:
            interpolation_mode = InterpolationMode.NEAREST if interpolation == "nearest" else InterpolationMode.BILINEAR
            image = TF.rotate(image, angle=angle, interpolation=interpolation_mode)
        if bool(geometry["use_crop"]):
            strength = self.crop_strength[scale_name]
            crop_h = 1.0 - (float(geometry["crop_h"]) - 1.0) * strength if float(geometry["crop_h"]) > 1.0 else 1.0 - (1.0 - float(geometry["crop_h"])) * strength
            crop_w = 1.0 - (float(geometry["crop_w"]) - 1.0) * strength if float(geometry["crop_w"]) > 1.0 else 1.0 - (1.0 - float(geometry["crop_w"])) * strength
            width, height = image.size
            crop_height = max(1, min(height, int(round(height * crop_h))))
            crop_width = max(1, min(width, int(round(width * crop_w))))
            top = int(round(float(geometry["crop_top"]) * max(height - crop_height, 0)))
            left = int(round(float(geometry["crop_left"]) * max(width - crop_width, 0)))
            image = TF.crop(image, top, left, crop_height, crop_width)
        return self.resize_aux_image(image, scale_name=scale_name, interpolation=interpolation)

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
        blended = ((1.0 - alpha[..., None]) * source_array) + (alpha[..., None] * transformed_array)
        if self.context_preserve_instance_strictly:
            mask_array = np.asarray(mask, dtype=np.uint8) > 0
            blended[mask_array] = source_array[mask_array]
        return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")

    def _make_context_dropout_image(self, image: Image.Image, *, stronger: bool) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32)
        fill_mode = random.choice(["mean", "black", "noise"])
        if fill_mode == "mean":
            fill_value = image_array.reshape(-1, image_array.shape[-1]).mean(axis=0, keepdims=True)
            fill_array = np.broadcast_to(fill_value.reshape(1, 1, 3), image_array.shape)
        elif fill_mode == "black":
            fill_array = np.zeros_like(image_array)
        else:
            fill_array = np.random.normal(loc=127.5, scale=40.0 if stronger else 28.0, size=image_array.shape).astype(np.float32)
        return Image.fromarray(np.clip(fill_array, 0.0, 255.0).astype(np.uint8), mode="RGB")

    def _make_context_noise_image(self, image: Image.Image, *, stronger: bool) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32)
        noise_std = 12.0 if stronger else 8.0
        noisy = image_array + np.random.normal(loc=0.0, scale=noise_std, size=image_array.shape).astype(np.float32)
        return Image.fromarray(np.clip(noisy, 0.0, 255.0).astype(np.uint8), mode="RGB")

    def _sample_context_flags(self, strength: float) -> dict[str, bool]:
        return {
            "dropout": random.random() < (self.context_dropout_prob * strength),
            "blur": random.random() < (self.context_blur_prob * strength),
            "grayscale": random.random() < (self.context_grayscale_prob * strength),
            "noise": random.random() < (self.context_noise_prob * strength),
            "mix": random.random() < (self.context_mix_prob * strength),
        }

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
        current = image
        if operation_flags["mix"]:
            mixed = Image.blend(current, counterpart, alpha=random.uniform(0.25, 0.50))
            current = self._blend_context(current, mixed, alpha, mask)
        if operation_flags["grayscale"]:
            desaturated = TF.adjust_saturation(current, saturation_factor=0.15 if stronger else 0.25)
            current = self._blend_context(current, desaturated, alpha, mask)
        if operation_flags["blur"]:
            blurred = current.filter(ImageFilter.GaussianBlur(radius=2.0 if stronger else 1.0))
            current = self._blend_context(current, blurred, alpha, mask)
        if operation_flags["dropout"]:
            current = self._blend_context(current, self._make_context_dropout_image(current, stronger=stronger), alpha, mask)
        if operation_flags["noise"]:
            current = self._blend_context(current, self._make_context_noise_image(current, stronger=stronger), alpha, mask)
        return current

    def _apply_scale_context_ops(
        self,
        pre: Image.Image,
        post: Image.Image,
        mask: Image.Image,
        *,
        scale_name: str,
    ) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, float]]:
        strength = self.context_aug_strength[scale_name]
        stats = {
            f"{scale_name}_context_applied": 0.0,
            f"{scale_name}_context_dropout": 0.0,
            f"{scale_name}_context_blur": 0.0,
            f"{scale_name}_context_grayscale": 0.0,
            f"{scale_name}_context_noise": 0.0,
            f"{scale_name}_context_mix": 0.0,
        }
        if strength <= 0.0:
            return pre, post, mask, stats

        alpha = self._build_context_alpha(mask)
        if alpha is None:
            return pre, post, mask, stats

        if self.context_apply_to_pre_and_post_independently:
            pre_flags = self._sample_context_flags(strength)
            post_flags = self._sample_context_flags(strength)
        else:
            shared_flags = self._sample_context_flags(strength)
            pre_flags = shared_flags
            post_flags = shared_flags

        pre = self._apply_context_ops_to_image(pre, post, mask, alpha, pre_flags, stronger=scale_name == "context")
        post = self._apply_context_ops_to_image(post, pre, mask, alpha, post_flags, stronger=scale_name == "context")
        for key in ["dropout", "blur", "grayscale", "noise", "mix"]:
            stats[f"{scale_name}_context_{key}"] = float(pre_flags[key] or post_flags[key])
        stats[f"{scale_name}_context_applied"] = float(any(value > 0.0 for value in stats.values()))
        return pre, post, mask, stats

    def prepare_sample(
        self,
        crops: dict[str, dict[str, Image.Image]],
    ) -> tuple[dict[str, dict[str, Image.Image]], dict[str, float]]:
        geometry = self._sample_geometry()
        prepared: dict[str, dict[str, Image.Image]] = {}
        stats: dict[str, float] = {}

        for scale_name in SCALE_NAMES:
            pre = crops[scale_name]["pre"]
            post = crops[scale_name]["post"]
            mask = crops[scale_name]["mask"]
            aux_images = crops[scale_name].get("aux_images", {})
            aux_image_modes = crops[scale_name].get("aux_image_modes", {})
            aux_stacks = crops[scale_name].get("aux_stacks", {})
            aux_stack_modes = crops[scale_name].get("aux_stack_modes", {})
            original_mask_pixels = float(np.asarray(mask, dtype=np.uint8).sum())
            pre, post, mask = self._apply_geometry(pre, post, mask, scale_name=scale_name, geometry=geometry)
            transformed_aux_images = {
                key: self._apply_geometry_to_aux_image(
                    image,
                    scale_name=scale_name,
                    geometry=geometry,
                    interpolation=str(aux_image_modes.get(key, "nearest")),
                )
                for key, image in aux_images.items()
            }
            transformed_aux_stacks = {
                key: [
                    self._apply_geometry_to_aux_image(
                        image,
                        scale_name=scale_name,
                        geometry=geometry,
                        interpolation=str(aux_stack_modes.get(key, "nearest")),
                    )
                    for image in images
                ]
                for key, images in aux_stacks.items()
            }
            retained_mask = float(np.asarray(mask, dtype=np.uint8).sum())
            retention = retained_mask / max(original_mask_pixels, 1.0) if original_mask_pixels > 0 else 1.0
            if retention < self.min_mask_retention:
                pre, post, mask = self.resize_triplet(crops[scale_name]["pre"], crops[scale_name]["post"], crops[scale_name]["mask"], scale_name=scale_name)
                transformed_aux_images = {
                    key: self.resize_aux_image(image, scale_name=scale_name, interpolation=str(aux_image_modes.get(key, "nearest")))
                    for key, image in aux_images.items()
                }
                transformed_aux_stacks = {
                    key: [
                        self.resize_aux_image(image, scale_name=scale_name, interpolation=str(aux_stack_modes.get(key, "nearest")))
                        for image in images
                    ]
                    for key, images in aux_stacks.items()
                }
                retention = 1.0
            if random.random() < self.color_jitter_prob:
                pre, post = self._apply_shared_color_jitter(pre, post)
            pre, post, mask, scale_stats = self._apply_scale_context_ops(pre, post, mask, scale_name=scale_name)
            prepared[scale_name] = {
                "pre": pre,
                "post": post,
                "mask": mask,
                "aux_images": transformed_aux_images,
                "aux_image_modes": dict(aux_image_modes),
                "aux_stacks": transformed_aux_stacks,
                "aux_stack_modes": dict(aux_stack_modes),
            }
            stats[f"{scale_name}_mask_retention"] = float(retention)
            stats.update(scale_stats)

        stats["shared_hflip"] = float(bool(geometry["hflip"]))
        stats["shared_vflip"] = float(bool(geometry["vflip"]))
        stats["shared_rotate90"] = float(int(geometry["rotate"]) != 0)
        stats["shared_random_resized_crop"] = float(bool(geometry["use_crop"]))
        return prepared, stats


class EvalMultiContextTransform(_BaseMultiContextTransform):
    def __init__(
        self,
        *,
        image_sizes: dict[str, int],
        normalize_mean: list[float] | tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: list[float] | tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__(image_sizes=image_sizes, normalize_mean=normalize_mean, normalize_std=normalize_std)

    def prepare_sample(
        self,
        crops: dict[str, dict[str, Image.Image]],
    ) -> tuple[dict[str, dict[str, Image.Image]], dict[str, float]]:
        prepared: dict[str, dict[str, Image.Image]] = {}
        for scale_name in SCALE_NAMES:
            pre, post, mask = self.resize_triplet(
                crops[scale_name]["pre"],
                crops[scale_name]["post"],
                crops[scale_name]["mask"],
                scale_name=scale_name,
            )
            aux_images = crops[scale_name].get("aux_images", {})
            aux_image_modes = crops[scale_name].get("aux_image_modes", {})
            aux_stacks = crops[scale_name].get("aux_stacks", {})
            aux_stack_modes = crops[scale_name].get("aux_stack_modes", {})
            prepared[scale_name] = {
                "pre": pre,
                "post": post,
                "mask": mask,
                "aux_images": {
                    key: self.resize_aux_image(image, scale_name=scale_name, interpolation=str(aux_image_modes.get(key, "nearest")))
                    for key, image in aux_images.items()
                },
                "aux_image_modes": dict(aux_image_modes),
                "aux_stacks": {
                    key: [
                        self.resize_aux_image(image, scale_name=scale_name, interpolation=str(aux_stack_modes.get(key, "nearest")))
                        for image in images
                    ]
                    for key, images in aux_stacks.items()
                },
                "aux_stack_modes": dict(aux_stack_modes),
            }
        return prepared, {}


def build_transforms(config: dict, is_train: bool) -> TrainMultiContextTransform | EvalMultiContextTransform:
    aug_cfg = config["augmentation"]
    data_cfg = config["dataset"]
    image_sizes = {
        "tight": int(data_cfg["image_size_tight"]),
        "context": int(data_cfg["image_size_context"]),
        "neighborhood": int(data_cfg["image_size_neighborhood"]),
    }
    common_kwargs = {
        "image_sizes": image_sizes,
        "normalize_mean": aug_cfg["normalize_mean"],
        "normalize_std": aug_cfg["normalize_std"],
    }
    if is_train:
        return TrainMultiContextTransform(
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
            **common_kwargs,
        )
    return EvalMultiContextTransform(**common_kwargs)
