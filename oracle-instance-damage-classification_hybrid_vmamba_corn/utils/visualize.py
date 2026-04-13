from __future__ import annotations

import matplotlib
import numpy as np
import torch
from pathlib import Path
from typing import Any

from utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_to_uint8_image(
    tensor: torch.Tensor,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    array = (array * std + mean).clip(0.0, 1.0)
    return (array * 255.0).round().astype(np.uint8)


def mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return (mask.detach().cpu().float().numpy() > 0.5).astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
    return overlay.clip(0.0, 255.0).astype(np.uint8)


def save_prediction_visual(
    save_path: str | Path,
    pre_image: torch.Tensor,
    post_image: torch.Tensor,
    mask: torch.Tensor,
    gt_label: str,
    pred_label: str,
    confidence: float,
    meta: dict[str, Any],
    extra_lines: list[str] | None = None,
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    pre_np = tensor_to_uint8_image(pre_image)
    post_np = tensor_to_uint8_image(post_image)
    mask_np = mask_to_numpy(mask)
    overlay_np = overlay_mask(post_np, mask_np)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pre_np)
    axes[0].set_title("Pre")
    axes[1].imshow(post_np)
    axes[1].set_title("Post")
    axes[2].imshow(overlay_np)
    axes[2].set_title("Mask Overlay")
    for ax in axes:
        ax.axis("off")

    title_lines = [
        f"GT: {gt_label} | Pred: {pred_label} | Conf: {confidence:.4f}",
        f"tile={meta.get('tile_id')} building={meta.get('building_idx')} split={meta.get('split')}",
    ]
    if extra_lines:
        title_lines.extend(extra_lines)
    title = "\n".join(title_lines)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
