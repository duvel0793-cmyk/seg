from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


def resolve_input_mode_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    input_cfg = config.get("input_mode", {})
    return {
        "use_dual_scale": bool(input_cfg.get("use_dual_scale", False)),
        "local_size": int(input_cfg.get("local_size", data_cfg.get("image_size", 224))),
        "context_size": int(input_cfg.get("context_size", data_cfg.get("image_size", 224))),
        "local_margin_ratio": float(input_cfg.get("local_margin_ratio", 0.15)),
        "context_scale": float(input_cfg.get("context_scale", 2.5)),
        "min_crop_size": int(input_cfg.get("min_crop_size", 32)),
    }


def resolve_geometry_prior_config(config: dict[str, Any]) -> dict[str, Any]:
    geometry_cfg = config.get("geometry_prior", {})
    return {
        "use_instance_mask": bool(geometry_cfg.get("use_instance_mask", True)),
        "use_boundary_prior": bool(geometry_cfg.get("use_boundary_prior", False)),
        "encoder_in_channels": int(geometry_cfg.get("encoder_in_channels", 5)),
        "boundary_width_px": int(geometry_cfg.get("boundary_width_px", 3)),
        "dilation_radius": int(geometry_cfg.get("dilation_radius", geometry_cfg.get("boundary_width_px", 3))),
        "erosion_radius": int(geometry_cfg.get("erosion_radius", 1)),
    }


def resolve_diff_input_config(config: dict[str, Any]) -> dict[str, Any]:
    diff_cfg = config.get("diff_input", {})
    return {
        "return_abs_diff": bool(diff_cfg.get("return_abs_diff", False)),
        "feed_abs_diff_to_model": bool(diff_cfg.get("feed_abs_diff_to_model", False)),
    }


def ensure_rgb_image(image: Image.Image, *, convert_mode: str = "rgb") -> Image.Image:
    if convert_mode == "rgb":
        return image.convert("RGB")
    if convert_mode == "grayscale_to_rgb":
        return image.convert("L").convert("RGB")
    raise ValueError(f"Unsupported image convert mode='{convert_mode}'.")


def build_crop_box(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    mode: str,
    local_margin_ratio: float,
    context_scale: float,
    min_crop_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    width = max(float(x2) - float(x1), 1.0)
    height = max(float(y2) - float(y1), 1.0)
    center_x = float(x1 + x2) * 0.5
    center_y = float(y1 + y2) * 0.5

    if mode == "local":
        crop_w = max(width * (1.0 + (2.0 * float(local_margin_ratio))), float(min_crop_size))
        crop_h = max(height * (1.0 + (2.0 * float(local_margin_ratio))), float(min_crop_size))
    elif mode == "context":
        crop_w = max(width * float(context_scale), float(min_crop_size))
        crop_h = max(height * float(context_scale), float(min_crop_size))
    else:
        raise ValueError(f"Unsupported crop mode='{mode}'.")

    crop_x1 = int(math.floor(center_x - (crop_w * 0.5)))
    crop_y1 = int(math.floor(center_y - (crop_h * 0.5)))
    crop_x2 = int(math.ceil(center_x + (crop_w * 0.5)))
    crop_y2 = int(math.ceil(center_y + (crop_h * 0.5)))

    if crop_x2 <= crop_x1:
        crop_x2 = crop_x1 + max(1, int(min_crop_size))
    if crop_y2 <= crop_y1:
        crop_y2 = crop_y1 + max(1, int(min_crop_size))
    return crop_x1, crop_y1, crop_x2, crop_y2


def crop_image(image: Image.Image, crop_box_xyxy: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(crop_box_xyxy)


def build_boundary_tensor(mask_tensor: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    if mask_tensor.ndim == 3:
        mask = mask_tensor.unsqueeze(0)
        squeeze = True
    elif mask_tensor.ndim == 4:
        mask = mask_tensor
        squeeze = False
    else:
        raise ValueError(f"Expected mask tensor with 3 or 4 dims, got shape={tuple(mask_tensor.shape)}.")

    mask = (mask > 0.5).float()
    geometry_cfg = resolve_geometry_prior_config(config)
    if not geometry_cfg["use_boundary_prior"]:
        boundary = torch.zeros_like(mask)
    else:
        dilation_radius = max(int(geometry_cfg["dilation_radius"]), int(geometry_cfg["boundary_width_px"]), 1)
        erosion_radius = max(int(geometry_cfg["erosion_radius"]), 0)
        dilated = F.max_pool2d(mask, kernel_size=(dilation_radius * 2) + 1, stride=1, padding=dilation_radius)
        if erosion_radius > 0:
            eroded = -F.max_pool2d(-mask, kernel_size=(erosion_radius * 2) + 1, stride=1, padding=erosion_radius)
        else:
            eroded = mask
        boundary = (dilated - eroded).clamp_(0.0, 1.0)
        has_mask = mask.flatten(1).sum(dim=1) > 0
        has_boundary = boundary.flatten(1).sum(dim=1) > 0
        fallback = has_mask & ~has_boundary
        if fallback.any():
            boundary[fallback] = mask[fallback]

    if squeeze:
        return boundary.squeeze(0)
    return boundary


def maybe_build_abs_diff(pre_tensor: torch.Tensor, post_tensor: torch.Tensor, config: dict[str, Any]) -> torch.Tensor | None:
    diff_cfg = resolve_diff_input_config(config)
    if not diff_cfg["return_abs_diff"]:
        return None
    return (post_tensor - pre_tensor).abs()


def build_cache_key_summary(config: dict[str, Any]) -> dict[str, Any]:
    input_cfg = resolve_input_mode_config(config)
    geometry_cfg = resolve_geometry_prior_config(config)
    diff_cfg = resolve_diff_input_config(config)
    return {
        "input_mode": input_cfg,
        "geometry_prior": geometry_cfg,
        "diff_input": diff_cfg,
    }


def summarize_dataset_sources(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        source_name = str(sample.get("source_name", "unknown"))
        counts[source_name] = counts.get(source_name, 0) + 1
    return counts


def resolve_list_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    return Path(path_value)
