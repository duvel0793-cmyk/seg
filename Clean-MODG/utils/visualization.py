"""Visualization helpers for crops, masks, and errors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 80, 80), alpha: float = 0.35) -> np.ndarray:
    image = image.copy()
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = mask > 0
    overlay = np.zeros_like(image)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    image[binary] = ((1.0 - alpha) * image[binary] + alpha * overlay[binary]).astype(np.uint8)
    return image


def draw_bbox(image: np.ndarray, bbox: Sequence[float], color: tuple[int, int, int] = (0, 255, 0), width: int = 2) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle(list(bbox), outline=color, width=width)
    return np.asarray(pil)


def draw_polygon(image: np.ndarray, polygon: Sequence[Sequence[float]], color: tuple[int, int, int] = (0, 255, 255), width: int = 2) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    if polygon:
        draw.line(list(map(tuple, polygon)) + [tuple(polygon[0])], fill=color, width=width)
    return np.asarray(pil)


def save_image_grid(images: Iterable[np.ndarray], titles: Iterable[str], path: str | Path, cols: int = 2, figsize: tuple[int, int] = (10, 10)) -> Path:
    images = list(images)
    titles = list(titles)
    rows = int(np.ceil(len(images) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < len(images):
            ax.imshow(images[idx])
            if idx < len(titles):
                ax.set_title(titles[idx])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
