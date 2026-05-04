from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from utils.misc import read_json


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    text = str(wkt).strip()
    upper = text.upper()
    if upper.startswith("POLYGON"):
        start = text.find("((")
        end = text.rfind("))")
        ring_text = text[start + 2 : end]
    elif upper.startswith("MULTIPOLYGON"):
        start = text.find("(((")
        end = text.rfind(")))")
        ring_text = text[start + 3 : end].split("),")[0]
    else:
        raise ValueError(f"Unsupported WKT geometry: {wkt[:32]}")
    ring_text = ring_text.replace("(", "").replace(")", "")
    points: list[tuple[float, float]] = []
    for pair in ring_text.split(","):
        chunks = pair.strip().split()
        if len(chunks) < 2:
            continue
        points.append((float(chunks[0]), float(chunks[1])))
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("Polygon has fewer than 3 points.")
    return points


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    x_coords = np.asarray([point[0] for point in pts], dtype=np.float64)
    y_coords = np.asarray([point[1] for point in pts], dtype=np.float64)
    return float(0.5 * abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))))


def polygon_bbox(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [point[0] for point in pts]
    ys = [point[1] for point in pts]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def clip_polygon_to_box(
    points: Iterable[tuple[float, float]],
    bbox_xyxy: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        return []
    x_min, y_min, x_max, y_max = [float(value) for value in bbox_xyxy]

    def inside(point: tuple[float, float], edge: str) -> bool:
        x, y = point
        if edge == "left":
            return x >= x_min
        if edge == "right":
            return x <= x_max
        if edge == "top":
            return y >= y_min
        if edge == "bottom":
            return y <= y_max
        raise ValueError(f"Unsupported edge={edge}")

    def intersect(start: tuple[float, float], end: tuple[float, float], edge: str) -> tuple[float, float]:
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        if edge in {"left", "right"}:
            x_edge = x_min if edge == "left" else x_max
            if abs(dx) < 1e-8:
                return x_edge, y1
            t = (x_edge - x1) / dx
            return x_edge, y1 + (t * dy)
        y_edge = y_min if edge == "top" else y_max
        if abs(dy) < 1e-8:
            return x1, y_edge
        t = (y_edge - y1) / dy
        return x1 + (t * dx), y_edge

    output = pts
    for edge in ("left", "right", "top", "bottom"):
        if not output:
            break
        input_pts = output
        output = []
        start = input_pts[-1]
        for end in input_pts:
            start_inside = inside(start, edge)
            end_inside = inside(end, edge)
            if start_inside and end_inside:
                output.append(end)
            elif start_inside and not end_inside:
                output.append(intersect(start, end, edge))
            elif not start_inside and end_inside:
                output.append(intersect(start, end, edge))
                output.append(end)
            start = end

    if len(output) >= 2 and output[0] == output[-1]:
        output = output[:-1]
    if len(output) < 3:
        return []
    return [(float(x), float(y)) for x, y in output]


def translate_polygon(points: Iterable[tuple[float, float]], offset_x: float, offset_y: float) -> list[tuple[float, float]]:
    return [(float(x - offset_x), float(y - offset_y)) for x, y in points]


def clip_bbox_to_image(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1_i = max(0, int(math.floor(x1)))
    y1_i = max(0, int(math.floor(y1)))
    x2_i = min(int(width), int(math.ceil(x2)))
    y2_i = min(int(height), int(math.ceil(y2)))
    if x2_i <= x1_i:
        x2_i = min(int(width), x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(int(height), y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def polygon_to_mask(
    points: Iterable[tuple[float, float]],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)
    offset_x, offset_y = offset
    shifted_points = [(float(x - offset_x), float(y - offset_y)) for x, y in points]
    image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(shifted_points, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)


def is_valid_polygon(points: Iterable[tuple[float, float]], min_area: float = 1.0) -> bool:
    pts = list(points)
    if len(pts) < 3:
        return False
    return polygon_area(pts) >= float(min_area)


def load_label_png(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.uint8)
    if array.ndim == 3:
        array = array[..., 0]
    return array


def infer_disaster_name(tile_id: str, payload: dict[str, Any] | None = None) -> str:
    if payload is not None:
        metadata = payload.get("metadata", {})
        if metadata.get("disaster"):
            return str(metadata["disaster"])
        if metadata.get("disaster_type"):
            return str(metadata["disaster_type"])
    return str(tile_id).rsplit("_", 1)[0] if "_" in str(tile_id) else str(tile_id)


def read_label_metadata(label_json_path: str | Path) -> tuple[int, int]:
    payload = read_json(label_json_path)
    metadata = payload.get("metadata", {})
    width = int(metadata.get("width", metadata.get("original_width", 1024)))
    height = int(metadata.get("height", metadata.get("original_height", 1024)))
    return height, width
