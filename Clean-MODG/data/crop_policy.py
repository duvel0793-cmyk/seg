"""Crop building instance regions with tight and context policies."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without OpenCV.
    cv2 = None


def polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> list[float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def expand_bbox(bbox: Sequence[float], padding_ratio: float, image_size: tuple[int, int] | None = None) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    expanded = [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y]
    return expanded


def crop_with_padding(image: np.ndarray, bbox: Sequence[float], pad_value: int | Sequence[int] = 0) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0i = int(np.floor(x0))
    y0i = int(np.floor(y0))
    x1i = int(np.ceil(x1))
    y1i = int(np.ceil(y1))

    height, width = image.shape[:2]
    pad_left = max(0, -x0i)
    pad_top = max(0, -y0i)
    pad_right = max(0, x1i - width)
    pad_bottom = max(0, y1i - height)

    if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
        if image.ndim == 3:
            if isinstance(pad_value, int):
                constant_values = ((pad_value, pad_value),) * 3
            else:
                constant_values = tuple((int(v), int(v)) for v in pad_value)
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=constant_values,
            )
        else:
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=int(pad_value),
            )
    x0i += pad_left
    x1i += pad_left
    y0i += pad_top
    y1i += pad_top
    return image[y0i:y1i, x0i:x1i]


def resize_crop(image: np.ndarray, size: int | tuple[int, int], interpolation: int | None = None) -> np.ndarray:
    if isinstance(size, int):
        target_w = target_h = size
    else:
        target_h, target_w = int(size[0]), int(size[1])
    if cv2 is not None:
        if interpolation is None:
            interpolation = cv2.INTER_LINEAR if image.ndim == 3 else cv2.INTER_NEAREST
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    resample = Image.BILINEAR if image.ndim == 3 else Image.NEAREST
    return np.asarray(Image.fromarray(image).resize((target_w, target_h), resample=resample))


def build_tight_context_crops(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    mask: np.ndarray,
    bbox: Sequence[float],
    tight_size: int,
    context_size: int,
    tight_padding_ratio: float,
    context_padding_ratio: float,
) -> Dict[str, np.ndarray]:
    height, width = pre_image.shape[:2]
    base_bbox = list(map(float, bbox))
    tight_bbox = expand_bbox(base_bbox, tight_padding_ratio, image_size=(width, height))
    context_bbox = expand_bbox(base_bbox, context_padding_ratio, image_size=(width, height))

    pre_tight = resize_crop(crop_with_padding(pre_image, tight_bbox, pad_value=0), tight_size)
    post_tight = resize_crop(crop_with_padding(post_image, tight_bbox, pad_value=0), tight_size)
    pre_context = resize_crop(crop_with_padding(pre_image, context_bbox, pad_value=0), context_size)
    post_context = resize_crop(crop_with_padding(post_image, context_bbox, pad_value=0), context_size)
    nearest = cv2.INTER_NEAREST if cv2 is not None else None
    mask_tight = resize_crop(crop_with_padding(mask, tight_bbox, pad_value=0), tight_size, interpolation=nearest)
    mask_context = resize_crop(crop_with_padding(mask, context_bbox, pad_value=0), context_size, interpolation=nearest)
    return {
        "pre_tight": pre_tight,
        "post_tight": post_tight,
        "pre_context": pre_context,
        "post_context": post_context,
        "mask_tight": (mask_tight > 0).astype(np.uint8),
        "mask_context": (mask_context > 0).astype(np.uint8),
        "tight_bbox": np.asarray(tight_bbox, dtype=np.float32),
        "context_bbox": np.asarray(context_bbox, dtype=np.float32),
    }
