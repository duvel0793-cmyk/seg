"""xBD polygon parsing and geometry transforms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.wkt import loads as load_wkt


DAMAGE_SUBTYPE_TO_RANK = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


def _clean_geometry(geometry: BaseGeometry) -> BaseGeometry | None:
    if geometry is None or geometry.is_empty:
        return None
    try:
        geometry = geometry.buffer(0)
    except Exception:
        return None
    if geometry.is_empty:
        return None
    return geometry


def _iter_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            if not poly.is_empty:
                yield poly


def load_xbd_polygons(json_path: str | Path, min_area: float = 1.0) -> List[Dict[str, object]]:
    path = Path(json_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Polygon JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", {}).get("xy", [])
    polygons: List[Dict[str, object]] = []

    for feature in features:
        properties = feature.get("properties", {})
        subtype = properties.get("subtype", "").strip().lower()
        try:
            geometry = _clean_geometry(load_wkt(feature.get("wkt", "")))
        except Exception:
            geometry = None
        if geometry is None:
            continue
        if geometry.area < min_area:
            continue
        polygons.append(
            {
                "uid": properties.get("uid", ""),
                "subtype": subtype,
                "damage_rank": DAMAGE_SUBTYPE_TO_RANK.get(subtype),
                "geometry": geometry,
                "area": float(geometry.area),
            }
        )
    return polygons


def crop_polygons(
    polygons: List[Dict[str, object]],
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int,
    min_area: float = 1.0,
) -> List[Dict[str, object]]:
    crop_box = box(crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    kept: List[Dict[str, object]] = []
    for polygon in polygons:
        geometry = polygon["geometry"].intersection(crop_box)
        geometry = _clean_geometry(geometry)
        if geometry is None or geometry.area < min_area:
            continue
        shifted = affinity.translate(geometry, xoff=-crop_left, yoff=-crop_top)
        kept.append({**polygon, "geometry": shifted, "area": float(shifted.area)})
    return kept


def flip_polygons_horizontal(polygons: List[Dict[str, object]], width: int) -> List[Dict[str, object]]:
    flipped: List[Dict[str, object]] = []
    for polygon in polygons:
        geometry = affinity.scale(polygon["geometry"], xfact=-1, yfact=1, origin=(width / 2.0, 0.0))
        flipped.append({**polygon, "geometry": geometry})
    return flipped


def flip_polygons_vertical(polygons: List[Dict[str, object]], height: int) -> List[Dict[str, object]]:
    flipped: List[Dict[str, object]] = []
    for polygon in polygons:
        geometry = affinity.scale(polygon["geometry"], xfact=1, yfact=-1, origin=(0.0, height / 2.0))
        flipped.append({**polygon, "geometry": geometry})
    return flipped


def scale_polygons(polygons: List[Dict[str, object]], scale_x: float, scale_y: float) -> List[Dict[str, object]]:
    scaled: List[Dict[str, object]] = []
    for polygon in polygons:
        geometry = affinity.scale(polygon["geometry"], xfact=scale_x, yfact=scale_y, origin=(0.0, 0.0))
        scaled.append({**polygon, "geometry": geometry, "area": float(geometry.area)})
    return scaled


def rotate_polygons_90(polygons: List[Dict[str, object]], width: int, height: int, k: int) -> List[Dict[str, object]]:
    k = int(k) % 4
    if k == 0:
        return polygons

    def _coord_map(x, y, z=None):
        if k == 1:
            return y, width - x
        if k == 2:
            return width - x, height - y
        return height - y, x

    rotated: List[Dict[str, object]] = []
    for polygon in polygons:
        geometry = shapely_transform(_coord_map, polygon["geometry"])
        rotated.append({**polygon, "geometry": geometry, "area": float(geometry.area)})
    return rotated


def serialize_polygons(polygons: List[Dict[str, object]]) -> List[Dict[str, object]]:
    serialized: List[Dict[str, object]] = []
    for polygon in polygons:
        parts = []
        holes = []
        for poly in _iter_polygons(polygon["geometry"]):
            parts.append([(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]])
            holes.append(
                [[(float(x), float(y)) for x, y in list(interior.coords)[:-1]] for interior in poly.interiors]
            )
        if not parts:
            continue
        serialized.append(
            {
                "uid": polygon["uid"],
                "subtype": polygon["subtype"],
                "damage_rank": polygon["damage_rank"],
                "area": float(polygon["geometry"].area),
                "parts": parts,
                "holes": holes,
                "bounds": tuple(float(v) for v in polygon["geometry"].bounds),
            }
        )
    return serialized
