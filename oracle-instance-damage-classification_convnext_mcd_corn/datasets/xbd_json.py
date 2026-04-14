from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from datasets.label_mapping import LABEL_TO_INDEX


@dataclass
class TilePaths:
    source_subset: str
    pre_image: str
    post_image: str
    post_label: str


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _remove_duplicate_closure(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) >= 2 and points[0] == points[-1]:
        return points[:-1]
    return points


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    text = str(wkt).strip()
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
        raise ValueError(f"Unsupported geometry: {wkt[:32]}")

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
        raise ValueError("Polygon must contain at least 3 vertices.")
    return points


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    x = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bbox(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_valid_polygon(points: Iterable[tuple[float, float]], min_area: float) -> bool:
    pts = list(points)
    if len(pts) < 3:
        return False
    unique_count = len({(round(x, 4), round(y, 4)) for x, y in pts})
    return unique_count >= 3 and polygon_area(pts) >= float(min_area)


def expand_bbox_with_context(
    bbox: tuple[float, float, float, float],
    context_ratio: float,
    min_crop_size: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    width = max(width * (1.0 + (2.0 * float(context_ratio))), float(min_crop_size))
    height = max(height * (1.0 + (2.0 * float(context_ratio))), float(min_crop_size))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = width * 0.5
    half_h = height * 0.5
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


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
    original_area = bbox_area(bbox)
    if original_area <= 0:
        return 1.0
    x1, y1, x2, y2 = bbox
    clipped_w = max(0.0, min(x2, image_width) - max(x1, 0.0))
    clipped_h = max(0.0, min(y2, image_height) - max(y1, 0.0))
    return float(max(0.0, 1.0 - ((clipped_w * clipped_h) / original_area)))


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


def _read_list_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _discover_tile_ids_from_labels(label_dir: Path) -> list[str]:
    tile_ids = []
    for label_path in sorted(label_dir.glob("*_post_disaster.json")):
        tile_ids.append(label_path.name.replace("_post_disaster.json", ""))
    return tile_ids


def resolve_split_tile_ids(root: str | Path, split: str) -> list[str]:
    root_path = Path(root).expanduser().resolve()
    split_name = str(split).lower()
    list_dir = root_path / "xBD_list"

    candidates: list[Path] = []
    if split_name == "train":
        candidates.append(list_dir / "train_all.txt")
    elif split_name in {"val", "hold"}:
        candidates.extend([list_dir / "val_all.txt", list_dir / "hold_all.txt"])
    elif split_name == "test":
        candidates.extend([list_dir / "test_all.txt", list_dir / "test.txt"])
    else:
        path_candidate = Path(split)
        if path_candidate.is_file():
            return _read_list_file(path_candidate)

    for candidate in candidates:
        if candidate.is_file():
            return _read_list_file(candidate)

    subset_dir = "hold" if split_name in {"val", "hold"} else split_name
    label_dir = root_path / subset_dir / "labels"
    if label_dir.is_dir():
        return _discover_tile_ids_from_labels(label_dir)
    raise FileNotFoundError(f"Unable to resolve tile ids for split='{split}'.")


def resolve_tile_paths(root: str | Path, tile_id: str, allow_tier3: bool = False) -> TilePaths | None:
    root_path = Path(root).expanduser().resolve()
    subsets = ["train", "hold", "test"]
    if allow_tier3:
        subsets.append("tier3")

    for subset in subsets:
        base = root_path / subset
        pre_image = base / "images" / f"{tile_id}_pre_disaster.png"
        post_image = base / "images" / f"{tile_id}_post_disaster.png"
        post_label = base / "labels" / f"{tile_id}_post_disaster.json"
        if pre_image.is_file() and post_image.is_file() and post_label.is_file():
            return TilePaths(
                source_subset=subset,
                pre_image=str(pre_image),
                post_image=str(post_image),
                post_label=str(post_label),
            )
    return None


def build_instance_samples(
    *,
    root: str | Path,
    split: str,
    crop_context_ratio: float,
    min_crop_size: int,
    min_polygon_area: float,
    min_mask_pixels: int,
    max_out_of_bound_ratio: float,
    allow_tier3: bool = False,
) -> list[dict[str, Any]]:
    tile_ids = resolve_split_tile_ids(root, split)
    samples: list[dict[str, Any]] = []

    for tile_id in tile_ids:
        tile_paths = resolve_tile_paths(root, tile_id, allow_tier3=allow_tier3)
        if tile_paths is None:
            continue
        payload = read_json(tile_paths.post_label)
        metadata = payload.get("metadata", {})
        image_width = int(metadata.get("width", metadata.get("original_width", 1024)))
        image_height = int(metadata.get("height", metadata.get("original_height", 1024)))
        features = payload.get("features", {}).get("xy", [])

        for building_idx, feature in enumerate(features):
            properties = feature.get("properties", {})
            subtype = properties.get("subtype")
            if properties.get("feature_type") != "building":
                continue
            if subtype not in LABEL_TO_INDEX:
                continue
            try:
                polygon_xy = parse_wkt_polygon(feature.get("wkt", ""))
            except Exception:
                continue
            if not is_valid_polygon(polygon_xy, min_polygon_area):
                continue

            tight_bbox = polygon_bbox(polygon_xy)
            crop_bbox_float = expand_bbox_with_context(
                tight_bbox,
                context_ratio=float(crop_context_ratio),
                min_crop_size=float(min_crop_size),
            )
            if out_of_bounds_fraction(crop_bbox_float, image_width, image_height) > float(max_out_of_bound_ratio):
                continue

            crop_bbox = clip_bbox_to_image(crop_bbox_float, image_width, image_height)
            crop_w = crop_bbox[2] - crop_bbox[0]
            crop_h = crop_bbox[3] - crop_bbox[1]
            mask = polygon_to_mask(polygon_xy, crop_h, crop_w, offset=(crop_bbox[0], crop_bbox[1]))
            mask_pixels = int(mask.sum())
            if mask_pixels < int(min_mask_pixels):
                continue

            samples.append(
                {
                    "tile_id": tile_id,
                    "building_idx": building_idx,
                    "uid": properties.get("uid"),
                    "label": int(LABEL_TO_INDEX[subtype]),
                    "original_subtype": subtype,
                    "polygon_xy": [(float(x), float(y)) for x, y in polygon_xy],
                    "bbox_xyxy": [float(v) for v in tight_bbox],
                    "crop_bbox_xyxy": [int(v) for v in crop_bbox],
                    "mask_pixels": mask_pixels,
                    "source_subset": tile_paths.source_subset,
                    "split": split,
                    "pre_image": tile_paths.pre_image,
                    "post_image": tile_paths.post_image,
                    "post_label": tile_paths.post_label,
                }
            )
    return samples

