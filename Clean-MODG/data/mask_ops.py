"""Mask utilities for polygon and bbox driven crops."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def polygon_to_mask(image_shape: Sequence[int], polygon: Sequence[Sequence[float]]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon:
        return mask
    pts = np.array(polygon, dtype=np.float32).round().astype(np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask
    if cv2 is not None:
        cv2.fillPoly(mask, [pts], color=1)
    else:
        pil = Image.fromarray(mask)
        draw = ImageDraw.Draw(pil)
        draw.polygon([tuple(point.tolist()) for point in pts], fill=1)
        mask = np.asarray(pil, dtype=np.uint8)
    return mask


def bbox_to_mask(image_shape: Sequence[int], bbox: Sequence[float]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if bbox is None or len(bbox) != 4:
        return mask
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask


def resize_mask(mask: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        target = (size, size)
    else:
        target = (int(size[1]), int(size[0]))
    if cv2 is not None:
        resized = cv2.resize(mask.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST)
    else:
        resized = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(target, resample=Image.NEAREST))
    return (resized > 0).astype(np.uint8)


def erode_mask(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    for _ in range(iterations):
        pil = pil.filter(ImageFilter.MinFilter(size=kernel_size))
    return (np.asarray(pil) > 0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    for _ in range(iterations):
        pil = pil.filter(ImageFilter.MaxFilter(size=kernel_size))
    return (np.asarray(pil) > 0).astype(np.uint8)


def shift_mask(mask: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    shifted = np.zeros_like(mask)
    height, width = mask.shape[:2]
    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x) if shift_x >= 0 else width
    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y) if shift_y >= 0 else height
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted
