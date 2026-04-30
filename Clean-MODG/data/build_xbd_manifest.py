"""Build a manifest CSV from the standard xBD directory layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from shapely import wkt


DAMAGE_TO_LABEL = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


def _sample_id_from_name(path: Path) -> str:
    stem = path.stem
    for suffix in ["_pre_disaster", "_post_disaster"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _disaster_id_from_sample(sample_id: str) -> str:
    if "_" not in sample_id:
        return sample_id
    return sample_id.rsplit("_", 1)[0]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _feature_properties(feature: Dict[str, Any]) -> Dict[str, Any]:
    return feature.get("properties", {}) or {}


def _feature_to_polygon(feature: Dict[str, Any]) -> List[List[float]] | None:
    geometry = feature.get("wkt")
    if geometry:
        try:
            shape = wkt.loads(geometry)
            if hasattr(shape, "exterior"):
                return [[float(x), float(y)] for x, y in shape.exterior.coords[:-1]]
        except Exception:
            return None
    coords = feature.get("coordinates")
    if isinstance(coords, list) and coords:
        if isinstance(coords[0][0], (int, float)):
            return [[float(x), float(y)] for x, y in coords]
        if isinstance(coords[0][0], list):
            return [[float(x), float(y)] for x, y in coords[0]]
    return None


def _polygon_to_bbox(polygon: List[List[float]]) -> List[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _iter_building_features(label_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    features = label_json.get("features", {})
    for key in ["xy", "features", "lng_lat"]:
        group = features.get(key) if isinstance(features, dict) else None
        if isinstance(group, list) and group:
            for feature in group:
                properties = _feature_properties(feature)
                feature_type = str(properties.get("feature_type", "building")).lower()
                if feature_type == "building":
                    yield feature


def parse_post_label_json(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    records: List[Dict[str, Any]] = []
    for idx, feature in enumerate(_iter_building_features(payload)):
        polygon = _feature_to_polygon(feature)
        if polygon is None or len(polygon) < 3:
            continue
        props = _feature_properties(feature)
        subtype = str(props.get("subtype", props.get("damage", ""))).strip().lower()
        if subtype not in DAMAGE_TO_LABEL:
            continue
        building_id = props.get("uid") or props.get("building_id") or f"{path.stem}_{idx:04d}"
        records.append(
            {
                "building_id": str(building_id),
                "label": DAMAGE_TO_LABEL[subtype],
                "polygon": polygon,
                "bbox": _polygon_to_bbox(polygon),
            }
        )
    return records


def scan_split(xbd_root: Path, split: str) -> List[Dict[str, Any]]:
    split_root = xbd_root / split
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for post_image in sorted(image_dir.glob("*_post_disaster.png")):
        sample_id = _sample_id_from_name(post_image)
        pre_image = image_dir / f"{sample_id}_pre_disaster.png"
        label_path = label_dir / f"{sample_id}_post_disaster.json"
        if not pre_image.exists() or not label_path.exists():
            continue
        for instance in parse_post_label_json(label_path):
            rows.append(
                {
                    "pre_image": str(pre_image.resolve()),
                    "post_image": str(post_image.resolve()),
                    "label": int(instance["label"]),
                    "polygon": json.dumps(instance["polygon"]),
                    "bbox": json.dumps(instance["bbox"]),
                    "building_id": instance["building_id"],
                    "disaster_id": _disaster_id_from_sample(sample_id),
                    "split": split,
                }
            )
    return rows


def build_manifest(xbd_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split in ["train", "tier3", "test", "hold", "val"]:
        normalized = "test" if split == "tier3" else split
        rows.extend(
            [{**row, "split": normalized} for row in scan_split(xbd_root, split)]
        )
    if not rows:
        raise RuntimeError(
            f"No xBD samples were found under {xbd_root}. "
            "Expected split/images/*.png and split/labels/*.json."
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest CSV for Clean-MODG from xBD raw data.")
    parser.add_argument("--xbd-root", type=str, required=True, help="Path to the xBD root directory.")
    parser.add_argument("--output", type=str, default="data/manifest.csv", help="Output CSV path.")
    args = parser.parse_args()

    xbd_root = Path(args.xbd_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest = build_manifest(xbd_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Saved manifest with {len(manifest)} rows to {output}")


if __name__ == "__main__":
    main()
