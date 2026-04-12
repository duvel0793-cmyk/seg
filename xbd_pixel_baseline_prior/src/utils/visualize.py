from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 165, 0),
    4: (255, 0, 0),
}


def _palette_list() -> list[int]:
    values = [0] * 768
    for index, color in PALETTE.items():
        base = index * 3
        values[base : base + 3] = list(color)
    return values


def mask_to_palette_image(mask: np.ndarray) -> Image.Image:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    image = Image.fromarray(mask_uint8, mode="P")
    image.putpalette(_palette_list())
    return image


def colorize_mask(mask: np.ndarray) -> Image.Image:
    return mask_to_palette_image(mask).convert("RGB")


def blend_overlay(base_image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    if base_image.mode != "RGB":
        base_image = base_image.convert("RGB")
    color_mask = colorize_mask(mask).resize(base_image.size, resample=Image.NEAREST)
    return Image.blend(base_image, color_mask, alpha=alpha)


def save_prediction_mask(mask: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_to_palette_image(mask).save(output_path)


def save_overlay(
    base_image: Image.Image,
    mask: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blend_overlay(base_image, mask, alpha=alpha).save(output_path)
