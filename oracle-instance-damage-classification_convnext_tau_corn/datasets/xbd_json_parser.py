"""Robust xBD JSON parser with pre-polygon priority and post-label alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets.label_mapping import DAMAGE_LABEL_MAP

try:
    from shapely import wkt as shapely_wkt
except Exception:  # pragma: no cover - optional dependency
    shapely_wkt = None


Point = Tuple[float, float]
PolygonRings = List[List[Point]]


class XBDJsonParser:
    """Parse xBD pre/post disaster annotations into instance-level records."""

    def __init__(
        self,
        data_root: str,
        train_split_file: str = "",
        val_split_file: str = "",
    ) -> None:
        self.data_root = Path(data_root)
        self.train_split_file = Path(train_split_file) if train_split_file else None
        self.val_split_file = Path(val_split_file) if val_split_file else None
        self._cache: Dict[str, List[Dict]] = {}
        self._split_stats: Dict[str, Dict[str, int]] = {}

    def parse_split(self, split: str) -> List[Dict]:
        """Parse one split into a standardized list of instance dicts."""
        if split in self._cache:
            return self._cache[split]

        label_dir, image_dir = self._resolve_split_dirs(split)
        if label_dir is None or image_dir is None:
            raise FileNotFoundError(f"Could not resolve label/image directories for split='{split}'.")

        base_ids = self._resolve_base_ids(split=split, label_dir=label_dir)
        records: List[Dict] = []
        split_stats = {
            "num_base_ids": len(base_ids),
            "num_records": 0,
            "num_missing_polygon": 0,
            "num_post_polygon_fallback": 0,
            "num_missing_label": 0,
        }
        for base_id in base_ids:
            instance_records, base_stats = self._parse_base_id(
                base_id=base_id,
                label_dir=label_dir,
                image_dir=image_dir,
                split=split,
            )
            records.extend(instance_records)
            for key, value in base_stats.items():
                split_stats[key] = split_stats.get(key, 0) + int(value)

        self._cache[split] = records
        split_stats["num_records"] = len(records)
        self._split_stats[split] = split_stats
        return records

    def get_split_stats(self, split: str) -> Dict[str, int]:
        """Return parser statistics for a split after parsing."""
        return dict(self._split_stats.get(split, {}))

    def _resolve_split_dirs(self, split: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Resolve label and image directories based on local xBD structure."""
        if split in {"train", "val"}:
            candidates = [
                (self.data_root / "train" / "labels", self.data_root / "train" / "images"),
                (self.data_root / "test" / "labels", self.data_root / "test" / "images"),
                (self.data_root / "tier3" / "labels", self.data_root / "tier3" / "images"),
                (self.data_root / "labels", self.data_root / "images"),
            ]
        elif split == "test":
            candidates = [
                (self.data_root / "test" / "labels", self.data_root / "test" / "images"),
                (self.data_root / "tier3" / "labels", self.data_root / "tier3" / "images"),
                (self.data_root / "train" / "labels", self.data_root / "train" / "images"),
                (self.data_root / "labels", self.data_root / "images"),
            ]
        else:
            raise ValueError(f"Unsupported split: {split}")

        valid_candidates = [(label_dir, image_dir) for label_dir, image_dir in candidates if label_dir.is_dir() and image_dir.is_dir()]
        if not valid_candidates:
            return None, None

        split_file = None
        if split == "train" and self.train_split_file and self.train_split_file.is_file():
            split_file = self.train_split_file
        elif split == "val" and self.val_split_file and self.val_split_file.is_file():
            split_file = self.val_split_file

        if split_file is None:
            return valid_candidates[0]

        with open(split_file, "r", encoding="utf-8") as handle:
            probe_base_ids = [line.strip() for line in handle if line.strip()][:32]

        best_candidate = valid_candidates[0]
        best_score = -1
        for label_dir, image_dir in valid_candidates:
            score = 0
            for base_id in probe_base_ids:
                if (label_dir / f"{base_id}_post_disaster.json").is_file() and (image_dir / f"{base_id}_post_disaster.png").is_file():
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = (label_dir, image_dir)
        return best_candidate

    def _resolve_base_ids(self, split: str, label_dir: Path) -> List[str]:
        """Resolve split members from split files if available, else scan labels."""
        split_file = None
        if split == "train" and self.train_split_file and self.train_split_file.is_file():
            split_file = self.train_split_file
        elif split == "val" and self.val_split_file and self.val_split_file.is_file():
            split_file = self.val_split_file

        if split_file is not None:
            with open(split_file, "r", encoding="utf-8") as handle:
                base_ids = [line.strip() for line in handle if line.strip()]
            return sorted(set(base_ids))

        pattern = "*_post_disaster.json"
        base_ids = []
        for json_path in sorted(label_dir.glob(pattern)):
            stem = json_path.stem
            if stem.endswith("_post_disaster"):
                base_ids.append(stem[: -len("_post_disaster")])
        return sorted(set(base_ids))

    def _parse_base_id(self, base_id: str, label_dir: Path, image_dir: Path, split: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Parse one xBD tile pair into per-building records."""
        pre_json_path = label_dir / f"{base_id}_pre_disaster.json"
        post_json_path = label_dir / f"{base_id}_post_disaster.json"
        pre_image_path = image_dir / f"{base_id}_pre_disaster.png"
        post_image_path = image_dir / f"{base_id}_post_disaster.png"

        stats = {
            "num_missing_polygon": 0,
            "num_post_polygon_fallback": 0,
            "num_missing_label": 0,
        }

        if not pre_json_path.is_file() or not post_json_path.is_file():
            return [], stats
        if not pre_image_path.is_file() or not post_image_path.is_file():
            return [], stats

        pre_objects = self._load_buildings(pre_json_path)
        post_objects = self._load_buildings(post_json_path)
        if not post_objects:
            return [], stats

        pre_lookup = {item["building_id"]: item for item in pre_objects}
        records: List[Dict] = []
        for post_item in post_objects:
            raw_subtype = post_item.get("raw_subtype", "")
            if raw_subtype not in DAMAGE_LABEL_MAP:
                stats["num_missing_label"] += 1
                continue

            building_id = post_item["building_id"]
            pre_item = pre_lookup.get(building_id)
            polygon = None
            polygon_source = ""
            if pre_item and pre_item.get("polygon"):
                polygon = pre_item["polygon"]
                polygon_source = "pre"
            elif post_item.get("polygon"):
                polygon = post_item["polygon"]
                polygon_source = "post_fallback"
                stats["num_post_polygon_fallback"] += 1
            if not polygon:
                stats["num_missing_polygon"] += 1
                continue

            bbox = self._compute_bbox(polygon)
            if bbox is None:
                continue

            records.append(
                {
                    "sample_id": f"{base_id}__{building_id}",
                    "image_id": base_id,
                    "building_id": building_id,
                    "pre_image_path": str(pre_image_path),
                    "post_image_path": str(post_image_path),
                    "polygon": polygon,
                    "polygon_source": polygon_source,
                    "bbox_xyxy": bbox,
                    "raw_subtype": raw_subtype,
                    "label": DAMAGE_LABEL_MAP[raw_subtype],
                    "split": split,
                }
            )
        return records, stats

    def _load_buildings(self, json_path: Path) -> List[Dict]:
        """Load valid building objects from an xBD json file."""
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        xy_features = data.get("features", {}).get("xy", [])
        buildings = []
        for item in xy_features:
            properties = item.get("properties", {})
            if properties.get("feature_type") != "building":
                continue

            building_id = properties.get("uid")
            if not building_id:
                continue

            polygon = self._parse_polygon_from_item(item)
            raw_subtype = properties.get("subtype", "")

            buildings.append(
                {
                    "building_id": building_id,
                    "polygon": polygon,
                    "raw_subtype": raw_subtype,
                }
            )
        return buildings

    def _parse_polygon_from_item(self, item: Dict) -> Optional[PolygonRings]:
        """Parse polygon coordinates from a structured field or WKT."""
        polygon_data = item.get("polygon")
        polygon = self._normalize_polygon_data(polygon_data)
        if polygon:
            return polygon
        return self._parse_wkt(item.get("wkt", ""))

    def _normalize_polygon_data(self, polygon_data) -> Optional[PolygonRings]:
        """Normalize optional polygon arrays into the internal ring format."""
        if not polygon_data:
            return None

        if isinstance(polygon_data, dict):
            for candidate_key in ("coordinates", "rings", "points"):
                polygon = self._normalize_polygon_data(polygon_data.get(candidate_key))
                if polygon:
                    return polygon
            return None

        if not isinstance(polygon_data, list):
            return None

        if polygon_data and self._looks_like_ring(polygon_data):
            ring = self._normalize_ring(polygon_data)
            return [ring] if ring else None

        polygons: PolygonRings = []
        for candidate in polygon_data:
            if self._looks_like_ring(candidate):
                ring = self._normalize_ring(candidate)
                if ring:
                    polygons.append(ring)
            elif isinstance(candidate, list):
                nested = self._normalize_polygon_data(candidate)
                if nested:
                    polygons.extend(nested)
        return polygons or None

    @staticmethod
    def _looks_like_ring(candidate) -> bool:
        return bool(candidate) and isinstance(candidate, list) and isinstance(candidate[0], (list, tuple))

    def _normalize_ring(self, points) -> Optional[List[Point]]:
        ring: List[Point] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                ring.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        return ring if self._is_valid_ring(ring) else None

    def _parse_wkt(self, wkt_text: str) -> Optional[PolygonRings]:
        """Parse POLYGON / MULTIPOLYGON into a list of outer rings."""
        if not wkt_text:
            return None

        if shapely_wkt is not None:
            try:
                geometry = shapely_wkt.loads(wkt_text)
                polygons: PolygonRings = []
                if geometry.geom_type == "Polygon":
                    polygons.extend(self._shapely_polygon_to_rings(geometry))
                elif geometry.geom_type == "MultiPolygon":
                    for poly in geometry.geoms:
                        polygons.extend(self._shapely_polygon_to_rings(poly))
                return polygons or None
            except Exception:
                pass

        return self._parse_wkt_fallback(wkt_text)

    @staticmethod
    def _shapely_polygon_to_rings(geometry) -> PolygonRings:
        coords = list(geometry.exterior.coords)
        ring = [(float(x), float(y)) for x, y in coords]
        return [ring] if len(ring) >= 3 else []

    def _parse_wkt_fallback(self, wkt_text: str) -> Optional[PolygonRings]:
        """Lightweight fallback parser supporting POLYGON and MULTIPOLYGON."""
        text = " ".join(wkt_text.strip().split())
        upper = text.upper()
        if upper.startswith("POLYGON"):
            body = text[text.find("(") :]
            polygons = self._parse_polygon_body(body)
        elif upper.startswith("MULTIPOLYGON"):
            body = text[text.find("(") :]
            polygons = []
            inner = self._strip_one_level(body)
            for polygon_text in self._split_top_level(inner):
                polygons.extend(self._parse_polygon_body(polygon_text))
        else:
            return None

        polygons = [ring for ring in polygons if self._is_valid_ring(ring)]
        return polygons or None

    def _parse_polygon_body(self, text: str) -> PolygonRings:
        inner = self._strip_one_level(text)
        rings = self._split_top_level(inner)
        polygons: PolygonRings = []
        for ring_text in rings[:1]:
            ring_inner = self._strip_one_level(ring_text)
            ring = self._parse_ring(ring_inner)
            if ring:
                polygons.append(ring)
        return polygons

    @staticmethod
    def _strip_one_level(text: str) -> str:
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            return text[1:-1].strip()
        return text

    def _split_top_level(self, text: str) -> List[str]:
        parts = []
        start = 0
        depth = 0
        for idx, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(text[start:idx].strip())
                start = idx + 1
        parts.append(text[start:].strip())
        return [part for part in parts if part]

    @staticmethod
    def _parse_ring(ring_text: str) -> List[Point]:
        points = []
        for pair in ring_text.split(","):
            parts = pair.strip().split()
            if len(parts) < 2:
                continue
            try:
                x_coord = float(parts[0])
                y_coord = float(parts[1])
            except ValueError:
                continue
            points.append((x_coord, y_coord))
        return points

    @staticmethod
    def _is_valid_ring(ring: Sequence[Point]) -> bool:
        unique_points = {(round(x, 3), round(y, 3)) for x, y in ring}
        return len(unique_points) >= 3

    @staticmethod
    def _compute_bbox(polygons: PolygonRings) -> Optional[List[float]]:
        xs = [x for polygon in polygons for x, _ in polygon]
        ys = [y for polygon in polygons for _, y in polygon]
        if not xs or not ys:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 <= x1 or y2 <= y1:
            return None
        return [float(x1), float(y1), float(x2), float(y2)]


def parse_xbd_split(
    data_root: str,
    split: str,
    train_split_file: str = "",
    val_split_file: str = "",
) -> List[Dict]:
    """Convenience function for one-off split parsing."""
    parser = XBDJsonParser(data_root=data_root, train_split_file=train_split_file, val_split_file=val_split_file)
    return parser.parse_split(split)
