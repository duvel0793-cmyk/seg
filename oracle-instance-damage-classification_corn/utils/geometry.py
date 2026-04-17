from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def _remove_duplicate_closure(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) >= 2 and points[0] == points[-1]:
        return points[:-1]
    return points


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    if not isinstance(wkt, str):
        raise TypeError("WKT must be a string.")

    text = wkt.strip()
    upper = text.upper()
    if upper.startswith("POLYGON"):
        start = text.find("((")
        end = text.rfind("))")
        if start == -1 or end == -1:
            raise ValueError(f"Invalid POLYGON WKT: {wkt}")
        ring_text = text[start + 2 : end]
    elif upper.startswith("MULTIPOLYGON"):
        start = text.find("(((")
        end = text.rfind(")))")
        if start == -1 or end == -1:
            raise ValueError(f"Invalid MULTIPOLYGON WKT: {wkt}")
        ring_text = text[start + 3 : end].split("),")[0]
    else:
        raise ValueError(f"Unsupported WKT geometry type: {wkt[:32]}")

    ring_text = ring_text.split("),")[0].replace("(", "").replace(")", "")
    points: list[tuple[float, float]] = []
    for pair in ring_text.split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Polygon contains non-finite coordinates.")
        points.append((x, y))

    points = _remove_duplicate_closure(points)
    if len(points) < 3:
        raise ValueError("Polygon must contain at least 3 distinct vertices.")
    return points


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    x = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_valid_polygon(points: Iterable[tuple[float, float]], min_area: float = 1.0) -> bool:
    pts = list(points)
    if len(pts) < 3:
        return False
    unique_count = len({(round(p[0], 4), round(p[1], 4)) for p in pts})
    if unique_count < 3:
        return False
    return polygon_area(pts) >= min_area


def polygon_bbox(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def expand_bbox(bbox: tuple[float, float, float, float], context_ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    x_margin = width * context_ratio
    y_margin = height * context_ratio
    return x1 - x_margin, y1 - y_margin, x2 + x_margin, y2 + y_margin


def clip_bbox_to_image(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1_i = max(0, int(math.floor(x1)))
    y1_i = max(0, int(math.floor(y1)))
    x2_i = min(image_width, int(math.ceil(x2)))
    y2_i = min(image_height, int(math.ceil(y2)))
    if x2_i <= x1_i:
        x2_i = min(image_width, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(image_height, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def out_of_bounds_fraction(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> float:
    x1, y1, x2, y2 = bbox
    original_area = bbox_area(bbox)
    if original_area <= 0:
        return 1.0
    clipped = (
        max(0.0, min(x2, image_width) - max(x1, 0.0)),
        max(0.0, min(y2, image_height) - max(y1, 0.0)),
    )
    clipped_area = clipped[0] * clipped[1]
    return float(max(0.0, 1.0 - clipped_area / original_area))


def polygon_to_mask(
    points: Iterable[tuple[float, float]],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)
    ox, oy = offset
    shifted = [(float(x - ox), float(y - oy)) for x, y in points]
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon(shifted, outline=1, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def is_small_target(points: Iterable[tuple[float, float]], min_area: float) -> bool:
    return polygon_area(points) < min_area
