from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from utils.misc import ensure_dir


def save_mask_overlay(path: str | Path, image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> None:
    image_u8 = np.asarray(image, dtype=np.uint8)
    mask_bool = np.asarray(mask, dtype=bool)
    color = np.zeros_like(image_u8)
    color[..., 0] = 255
    blended = image_u8.astype(np.float32).copy()
    blended[mask_bool] = ((1.0 - alpha) * blended[mask_bool]) + (alpha * color[mask_bool])
    ensure_dir(Path(path).parent)
    Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8)).save(path)
