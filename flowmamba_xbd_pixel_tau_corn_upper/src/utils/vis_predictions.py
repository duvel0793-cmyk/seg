"""Prediction visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from ..models.modules.ordinal_utils import combine_damage_and_loc


PALETTE = {
    0: (0, 0, 0),
    1: (64, 192, 64),
    2: (255, 200, 0),
    3: (255, 96, 0),
    4: (220, 32, 32),
}


def _denorm_to_uint8_rgb(image_tensor: torch.Tensor, mean, std) -> np.ndarray:
    mean = torch.tensor(mean, device=image_tensor.device, dtype=image_tensor.dtype)[:, None, None]
    std = torch.tensor(std, device=image_tensor.device, dtype=image_tensor.dtype)[:, None, None]
    image = image_tensor * std + mean
    image = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, color in PALETTE.items():
        out[mask == idx] = np.asarray(color, dtype=np.uint8)
    return out


def _overlay(base_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(base_rgb.astype(np.float32) * (1.0 - alpha) + mask_rgb.astype(np.float32) * alpha, 0.0, 255.0).astype(
        np.uint8
    )


def save_prediction_bundle(
    save_dir: str | Path,
    sample: Dict[str, Any],
    outputs: Dict[str, Any],
    mean,
    std,
    stem: str,
) -> None:
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    pre_img = _denorm_to_uint8_rgb(sample["pre_image"][0], mean=mean, std=std)
    post_img = _denorm_to_uint8_rgb(sample["post_image"][0], mean=mean, std=std)
    gt = sample["post_target"][0].detach().cpu().numpy().astype(np.int64)

    loc_pred = outputs["loc_logits"][0].argmax(dim=0)
    damage_rank_pred = outputs["damage_rank_pred"][0]
    pred = combine_damage_and_loc(loc_pred=loc_pred, damage_rank_pred=damage_rank_pred).detach().cpu().numpy().astype(np.int64)

    gt_rgb = _mask_to_rgb(gt)
    pred_rgb = _mask_to_rgb(pred)
    loc_rgb = _mask_to_rgb((loc_pred.detach().cpu().numpy() > 0).astype(np.int64))

    strip = np.concatenate([pre_img, post_img, gt_rgb, pred_rgb], axis=1)
    Image.fromarray(strip).save(save_dir / f"{stem}_strip.png")
    Image.fromarray(_overlay(post_img, pred_rgb)).save(save_dir / f"{stem}_pred_overlay.png")
    Image.fromarray(_overlay(post_img, loc_rgb)).save(save_dir / f"{stem}_loc_overlay.png")
    Image.fromarray(pred_rgb).save(save_dir / f"{stem}_damage_pred.png")

