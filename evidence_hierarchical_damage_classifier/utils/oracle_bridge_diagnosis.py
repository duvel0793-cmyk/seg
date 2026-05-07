from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from datasets.xbd_instance_dataset import _resolve_tile_paths, _scale_bbox
from utils.misc import (
    CLASS_NAMES,
    LABEL_TO_INDEX,
    SCALE_NAMES,
    clip_bbox_to_image,
    ensure_dir,
    infer_disaster_name,
    is_valid_polygon,
    out_of_bounds_fraction,
    parse_wkt_polygon,
    polygon_area,
    polygon_bbox,
    polygon_to_mask,
    read_json,
    write_csv_rows,
    write_json,
)
from utils.pixel_bridge_eval import (
    PIXEL_CLASS_NAMES,
    build_class_report_text,
    compute_confusion_matrix,
    compute_metrics_from_confusion_matrix,
    rasterize_instances_to_damage_map,
    resolve_official_target_png,
    save_tile_outputs,
)


AREA_BINS: list[tuple[str, float, float | None]] = [
    ("0-64", 0.0, 64.0),
    ("64-128", 64.0, 128.0),
    ("128-256", 128.0, 256.0),
    ("256-512", 256.0, 512.0),
    ("512-1024", 512.0, 1024.0),
    ("1024-2048", 1024.0, 2048.0),
    ("2048-4096", 2048.0, 4096.0),
    ("4096-8192", 4096.0, 8192.0),
    ("8192+", 8192.0, None),
]
PIXEL_CLASS_ID_TO_INSTANCE_LABEL = {1: 0, 2: 1, 3: 2, 4: 3}


@dataclass
class TileInfo:
    tile_id: str
    source_subset: str | None
    pre_image: str | None
    post_image: str | None
    post_label: str | None
    post_target: str | None
    disaster_name: str
    image_size: tuple[int, int]

    def as_sample_dict(self) -> dict[str, Any]:
        height, width = self.image_size
        return {
            "tile_id": self.tile_id,
            "source_subset": self.source_subset,
            "pre_image": self.pre_image,
            "post_image": self.post_image,
            "post_label": self.post_label,
            "post_target": self.post_target,
            "disaster_name": self.disaster_name,
            "image_size": [int(height), int(width)],
        }


def read_split_tile_ids(path: str | Path, max_tiles: int | None = None) -> list[str]:
    tile_ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_tiles is not None and int(max_tiles) > 0:
        tile_ids = tile_ids[: int(max_tiles)]
    return tile_ids


def _label_name_from_index(label: int) -> str:
    return str(CLASS_NAMES[int(label)]).replace("-", "_")


def _safe_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _safe_int(value: Any) -> int:
    return int(value) if value is not None else 0


def _pixel_label_summary_from_counts(counts: dict[int, int]) -> dict[str, int]:
    return {PIXEL_CLASS_NAMES[int(class_id)]: int(value) for class_id, value in sorted(counts.items())}


def _serialize_confusion_dict(confusion: dict[int, int]) -> str:
    return json.dumps(_pixel_label_summary_from_counts(confusion), ensure_ascii=False, sort_keys=True)


def _area_bin_name(area_value: float) -> str:
    area_value = float(area_value)
    for bin_name, lower, upper in AREA_BINS:
        if area_value >= float(lower) and (upper is None or area_value < float(upper)):
            return bin_name
    return AREA_BINS[-1][0]


def _disk_structure(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx) + (yy * yy)) <= (radius * radius)


def _tile_sort_key(tile_id: str) -> tuple[str, str]:
    if "_" in str(tile_id):
        prefix, suffix = str(tile_id).rsplit("_", 1)
        return prefix, suffix
    return str(tile_id), ""


def build_tile_info(tile_id: str, config: dict[str, Any]) -> TileInfo | None:
    allow_tier3 = bool(config.get("dataset", {}).get("allow_tier3", False))
    tile_paths = _resolve_tile_paths(config["data"]["root"], tile_id, allow_tier3=allow_tier3)
    if tile_paths is None:
        return None
    payload = read_json(tile_paths.post_label)
    metadata = payload.get("metadata", {})
    image_width = int(metadata.get("width", metadata.get("original_width", 1024)))
    image_height = int(metadata.get("height", metadata.get("original_height", 1024)))
    disaster_name = infer_disaster_name(tile_id, payload)
    return TileInfo(
        tile_id=str(tile_id),
        source_subset=tile_paths.subset,
        pre_image=tile_paths.pre_image,
        post_image=tile_paths.post_image,
        post_label=tile_paths.post_label,
        post_target=tile_paths.post_target,
        disaster_name=disaster_name,
        image_size=(image_height, image_width),
    )


def build_tile_info_map(tile_ids: list[str], config: dict[str, Any]) -> tuple[dict[str, TileInfo], list[dict[str, Any]]]:
    tile_info_map: dict[str, TileInfo] = {}
    missing_tiles: list[dict[str, Any]] = []
    for tile_id in tile_ids:
        tile_info = build_tile_info(tile_id, config)
        if tile_info is None:
            missing_tiles.append({"tile_id": str(tile_id), "reason": "missing_tile_paths"})
            continue
        tile_info_map[str(tile_id)] = tile_info
    return tile_info_map, missing_tiles


def _load_official_target_map(tile_info: TileInfo, config: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    target_path = resolve_official_target_png(tile_info.as_sample_dict(), config)
    if target_path is None:
        return None, None
    from utils.misc import load_label_png  # Imported lazily to keep module startup lighter.

    gt_map = load_label_png(target_path)
    expected_shape = tuple(int(v) for v in tile_info.image_size)
    if tuple(gt_map.shape) != expected_shape:
        raise ValueError(
            f"Official target PNG shape mismatch for tile {tile_info.tile_id}: "
            f"expected {expected_shape}, got {tuple(gt_map.shape)} from {target_path}."
        )
    return gt_map.astype(np.uint8), str(target_path)


def _bbox_list(points: list[tuple[float, float]]) -> list[float]:
    x1, y1, x2, y2 = polygon_bbox(points)
    return [float(x1), float(y1), float(x2), float(y2)]


def _mask_pixel_count(points: list[tuple[float, float]], image_size: tuple[int, int]) -> int:
    height, width = image_size
    return int(polygon_to_mask(points, height, width, offset=(0.0, 0.0)).sum())


def _iter_building_feature_reports(
    tile_id: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tile_info = build_tile_info(tile_id, config)
    if tile_info is None:
        return [], {"tile_id": str(tile_id), "missing_tile_paths": True}
    payload = read_json(tile_info.post_label or "")
    features = payload.get("features", {}).get("xy", [])
    min_polygon_area = float(config.get("dataset", {}).get("min_polygon_area", 16.0))
    min_mask_pixels = int(config.get("dataset", {}).get("min_mask_pixels", 16))
    max_oob = float(config.get("dataset", {}).get("max_out_of_bound_ratio", 0.4))
    scale_cfg = config.get("dataset", {}).get("crop_scales", {})
    enabled_scales = [
        scale_name
        for scale_name in SCALE_NAMES
        if bool(scale_cfg.get(scale_name, {}).get("enabled", True))
    ]
    image_height, image_width = tile_info.image_size
    reports: list[dict[str, Any]] = []
    counts = Counter()
    counts["num_features_total"] = int(len(features))
    for building_idx, feature in enumerate(features):
        props = feature.get("properties", {}) or {}
        feature_type = str(props.get("feature_type", ""))
        subtype = props.get("subtype")
        base_row: dict[str, Any] = {
            "tile_id": str(tile_id),
            "building_idx": int(building_idx),
            "feature_type": feature_type,
            "subtype": subtype,
            "original_subtype": subtype,
            "source_subset": tile_info.source_subset,
            "disaster_name": tile_info.disaster_name,
            "image_size": [int(image_height), int(image_width)],
            "post_image": tile_info.post_image,
            "post_label": tile_info.post_label,
            "post_target": tile_info.post_target,
            "bbox_xyxy": [],
            "polygon_area": 0.0,
            "mask_pixels_full_image": 0,
            "skip_reason": None,
            "is_dataset_valid": False,
            "is_full_json_valid": False,
        }
        if feature_type != "building":
            base_row["skip_reason"] = "not_building"
            reports.append(base_row)
            counts["not_building"] += 1
            continue
        counts["num_building_features"] += 1
        if subtype not in LABEL_TO_INDEX:
            base_row["skip_reason"] = "unknown_subtype"
            reports.append(base_row)
            counts["unknown_subtype"] += 1
            continue
        base_row["gt_label"] = int(LABEL_TO_INDEX[subtype])
        wkt_text = feature.get("wkt", "")
        try:
            polygon_xy = [(float(x), float(y)) for x, y in parse_wkt_polygon(wkt_text)]
        except Exception:
            base_row["skip_reason"] = "invalid_wkt"
            reports.append(base_row)
            counts["invalid_wkt"] += 1
            continue
        area_value = float(polygon_area(polygon_xy))
        mask_pixels = _mask_pixel_count(polygon_xy, tile_info.image_size)
        base_row["polygon_xy"] = polygon_xy
        base_row["bbox_xyxy"] = _bbox_list(polygon_xy)
        base_row["polygon_area"] = area_value
        base_row["mask_pixels_full_image"] = int(mask_pixels)
        base_row["is_full_json_valid"] = True
        if not is_valid_polygon(polygon_xy, min_area=min_polygon_area):
            base_row["skip_reason"] = "invalid_polygon_or_small_area"
            reports.append(base_row)
            counts["invalid_polygon_or_small_area"] += 1
            continue
        tight_bbox_raw = polygon_bbox(polygon_xy)
        scale_bboxes: dict[str, list[int] | None] = {scale_name: None for scale_name in SCALE_NAMES}
        out_of_bounds_reason: str | None = None
        for scale_name in enabled_scales:
            scaled_bbox = _scale_bbox(tight_bbox_raw, float(scale_cfg[scale_name]["context_factor"]))
            if scale_name != "tight" and out_of_bounds_fraction(scaled_bbox, image_width, image_height) > max_oob:
                out_of_bounds_reason = f"{scale_name}_oob"
                break
            scale_bboxes[scale_name] = [
                int(v) for v in clip_bbox_to_image(scaled_bbox, image_width, image_height)
            ]
        if out_of_bounds_reason is not None:
            base_row["skip_reason"] = out_of_bounds_reason
            reports.append(base_row)
            counts[out_of_bounds_reason] += 1
            continue
        tight_bbox = tuple(scale_bboxes["tight"] or clip_bbox_to_image(tight_bbox_raw, image_width, image_height))
        tight_mask = polygon_to_mask(
            polygon_xy,
            tight_bbox[3] - tight_bbox[1],
            tight_bbox[2] - tight_bbox[0],
            offset=(tight_bbox[0], tight_bbox[1]),
        )
        if int(tight_mask.sum()) < min_mask_pixels:
            base_row["skip_reason"] = "tight_mask_too_small"
            reports.append(base_row)
            counts["tight_mask_too_small"] += 1
            continue
        base_row["skip_reason"] = None
        base_row["is_dataset_valid"] = True
        base_row["tight_bbox_xyxy"] = scale_bboxes["tight"]
        base_row["context_bbox_xyxy"] = scale_bboxes["context"]
        base_row["neighborhood_bbox_xyxy"] = scale_bboxes["neighborhood"]
        reports.append(base_row)
        counts["dataset_valid"] += 1
    counts["full_json_valid"] = int(
        sum(1 for row in reports if row.get("feature_type") == "building" and row.get("is_full_json_valid"))
    )
    counts["dataset_valid"] = int(sum(1 for row in reports if row.get("is_dataset_valid")))
    return reports, dict(counts)


def load_full_json_instances_for_tile(tile_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    reports, _ = _iter_building_feature_reports(tile_id, config)
    instances: list[dict[str, Any]] = []
    for row in reports:
        if row.get("feature_type") != "building":
            continue
        if not row.get("is_full_json_valid"):
            continue
        polygon_xy = row.get("polygon_xy")
        if not polygon_xy:
            continue
        instances.append(
            {
                "tile_id": str(tile_id),
                "building_idx": int(row["building_idx"]),
                "polygon_xy": polygon_xy,
                "polygon_area": float(row["polygon_area"]),
                "gt_label": int(row["gt_label"]),
                "label": int(row["gt_label"]),
                "original_subtype": row.get("original_subtype"),
                "post_image": row.get("post_image"),
                "post_label": row.get("post_label"),
                "post_target": row.get("post_target"),
                "source_subset": row.get("source_subset"),
                "image_size": row.get("image_size"),
                "disaster_name": row.get("disaster_name"),
                "bbox_xyxy": row.get("bbox_xyxy"),
            }
        )
    return instances


def collect_full_json_diagnostics(
    tile_ids: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    instances_by_tile: dict[str, list[dict[str, Any]]] = {}
    reports_by_tile: dict[str, list[dict[str, Any]]] = {}
    global_counts = Counter()
    for tile_id in tile_ids:
        reports, counts = _iter_building_feature_reports(tile_id, config)
        reports_by_tile[str(tile_id)] = reports
        valid_instances: list[dict[str, Any]] = []
        for row in reports:
            if row.get("feature_type") != "building":
                continue
            if not row.get("is_full_json_valid"):
                continue
            valid_instances.append(
                {
                    "tile_id": str(tile_id),
                    "building_idx": int(row["building_idx"]),
                    "polygon_xy": row["polygon_xy"],
                    "polygon_area": float(row["polygon_area"]),
                    "gt_label": int(row["gt_label"]),
                    "label": int(row["gt_label"]),
                    "original_subtype": row.get("original_subtype"),
                    "post_image": row.get("post_image"),
                    "post_label": row.get("post_label"),
                    "post_target": row.get("post_target"),
                    "source_subset": row.get("source_subset"),
                    "image_size": row.get("image_size"),
                    "disaster_name": row.get("disaster_name"),
                    "bbox_xyxy": row.get("bbox_xyxy"),
                }
            )
        instances_by_tile[str(tile_id)] = valid_instances
        for key, value in counts.items():
            global_counts[key] += int(value)
    stats = {
        "num_tiles_requested": int(len(tile_ids)),
        "num_tiles_with_reports": int(len(reports_by_tile)),
        "num_full_json_valid_instances": int(sum(len(items) for items in instances_by_tile.values())),
        "invalid_polygon_count": int(global_counts.get("invalid_polygon_or_small_area", 0)),
        "invalid_wkt_count": int(global_counts.get("invalid_wkt", 0)),
        "unknown_subtype_count": int(global_counts.get("unknown_subtype", 0)),
        "not_building_count": int(global_counts.get("not_building", 0)),
        "counts": {key: int(value) for key, value in sorted(global_counts.items())},
    }
    return instances_by_tile, reports_by_tile, stats


def group_dataset_samples_by_tile(dataset_samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in dataset_samples:
        grouped[str(sample["tile_id"])].append(sample)
    return dict(grouped)


def build_prediction_records_by_tile(prediction_records: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    by_tile: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in prediction_records:
        tile_id = str(record["tile_id"])
        building_idx = int(record["building_idx"])
        by_tile[tile_id][building_idx] = record
    return dict(by_tile)


def build_dataset_gt_instances_by_tile(dataset_samples_by_tile: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    instances_by_tile: dict[str, list[dict[str, Any]]] = {}
    for tile_id, samples in dataset_samples_by_tile.items():
        instances_by_tile[str(tile_id)] = [
            {
                "tile_id": str(tile_id),
                "building_idx": int(sample["building_idx"]),
                "polygon_xy": [(float(x), float(y)) for x, y in sample["polygon_xy"]],
                "polygon_area": float(sample["polygon_area"]),
                "gt_label": int(sample["label"]),
                "label": int(sample["label"]),
                "pred_label": int(sample["label"]),
                "original_subtype": sample.get("original_subtype"),
                "post_image": sample.get("post_image"),
                "post_label": sample.get("post_label"),
                "post_target": sample.get("post_target"),
                "source_subset": sample.get("source_subset"),
                "image_size": sample.get("image_size"),
                "disaster_name": sample.get("disaster_name"),
                "bbox_xyxy": sample.get("bbox_xyxy"),
            }
            for sample in samples
        ]
    return instances_by_tile


def build_current_model_instances_by_tile(
    dataset_samples_by_tile: dict[str, list[dict[str, Any]]],
    prediction_records_by_tile: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    instances_by_tile: dict[str, list[dict[str, Any]]] = {}
    for tile_id, samples in dataset_samples_by_tile.items():
        pred_items: list[dict[str, Any]] = []
        pred_lookup = prediction_records_by_tile.get(str(tile_id), {})
        for sample in samples:
            building_idx = int(sample["building_idx"])
            record = pred_lookup.get(building_idx)
            if record is None:
                continue
            pred_items.append(
                {
                    "tile_id": str(tile_id),
                    "building_idx": building_idx,
                    "polygon_xy": [(float(x), float(y)) for x, y in sample["polygon_xy"]],
                    "polygon_area": float(sample["polygon_area"]),
                    "gt_label": int(sample["label"]),
                    "label": int(sample["label"]),
                    "pred_label": int(record.get("final_pred_label", record.get("pred_label"))),
                    "final_class_probabilities": record.get(
                        "final_class_probabilities",
                        record.get("class_probabilities"),
                    ),
                    "prediction_record": record,
                    "original_subtype": sample.get("original_subtype"),
                    "post_image": sample.get("post_image"),
                    "post_label": sample.get("post_label"),
                    "post_target": sample.get("post_target"),
                    "source_subset": sample.get("source_subset"),
                    "image_size": sample.get("image_size"),
                    "disaster_name": sample.get("disaster_name"),
                    "bbox_xyxy": sample.get("bbox_xyxy"),
                }
            )
        instances_by_tile[str(tile_id)] = pred_items
    return instances_by_tile


def diagnose_skipped_buildings(
    tile_ids: list[str],
    config: dict[str, Any],
    dataset_samples_by_tile: dict[str, list[dict[str, Any]]],
    full_json_reports_by_tile: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    missing_rows: list[dict[str, Any]] = []
    missing_by_reason = Counter()
    missing_by_class = Counter()
    missing_by_disaster = Counter()
    num_full_json_buildings = 0
    missing_area_pixels_total = 0
    for tile_id in tile_ids:
        dataset_building_ids = {int(sample["building_idx"]) for sample in dataset_samples_by_tile.get(str(tile_id), [])}
        reports = full_json_reports_by_tile.get(str(tile_id), [])
        for row in reports:
            if row.get("feature_type") != "building":
                continue
            num_full_json_buildings += 1
            building_idx = int(row["building_idx"])
            if building_idx in dataset_building_ids:
                continue
            skip_reason = str(row.get("skip_reason") or "unknown_missing_from_dataset")
            if row.get("is_dataset_valid") and skip_reason == "None":
                skip_reason = "unknown_missing_from_dataset"
            if row.get("is_dataset_valid") and (row.get("skip_reason") is None):
                skip_reason = "unknown_missing_from_dataset"
            bbox_xyxy = row.get("bbox_xyxy") or []
            gt_label = row.get("gt_label")
            class_name = "" if gt_label is None else str(CLASS_NAMES[int(gt_label)])
            missing_row = {
                "tile_id": str(tile_id),
                "building_idx": building_idx,
                "subtype": row.get("subtype"),
                "polygon_area": float(row.get("polygon_area", 0.0)),
                "mask_pixels_full_image": int(row.get("mask_pixels_full_image", 0)),
                "bbox_xyxy": json.dumps(bbox_xyxy, ensure_ascii=False),
                "skip_reason": skip_reason,
                "disaster_name": row.get("disaster_name"),
                "gt_label": "" if gt_label is None else int(gt_label),
                "class_name": class_name,
            }
            missing_rows.append(missing_row)
            missing_by_reason[skip_reason] += 1
            missing_by_disaster[str(row.get("disaster_name", ""))] += 1
            if class_name:
                missing_by_class[class_name] += 1
            missing_area_pixels_total += int(row.get("mask_pixels_full_image", 0))
    num_dataset_samples = int(sum(len(items) for items in dataset_samples_by_tile.values()))
    num_missing = int(len(missing_rows))
    summary = {
        "num_tiles_requested": int(len(tile_ids)),
        "num_full_json_buildings": int(num_full_json_buildings),
        "num_dataset_samples": num_dataset_samples,
        "num_missing_from_dataset": num_missing,
        "missing_ratio": float(num_missing / max(num_full_json_buildings, 1)),
        "missing_area_pixels_total": int(missing_area_pixels_total),
        "missing_by_class": {key: int(value) for key, value in sorted(missing_by_class.items())},
        "missing_by_reason": {key: int(value) for key, value in sorted(missing_by_reason.items())},
        "missing_by_disaster": {key: int(value) for key, value in sorted(missing_by_disaster.items())},
    }
    return {"rows": missing_rows, "summary": summary}


def export_skipped_buildings_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    rows = report.get("rows", [])
    fieldnames = [
        "tile_id",
        "building_idx",
        "subtype",
        "polygon_area",
        "mask_pixels_full_image",
        "bbox_xyxy",
        "skip_reason",
        "disaster_name",
        "gt_label",
        "class_name",
    ]
    write_csv_rows(output_dir / "skipped_buildings_report.csv", fieldnames, rows)
    write_json(output_dir / "skipped_buildings_report.json", report)


def export_tile_coverage_report(
    output_dir: str | Path,
    tile_ids: list[str],
    tile_info_map: dict[str, TileInfo],
    dataset_instances_by_tile: dict[str, list[dict[str, Any]]],
    full_json_instances_by_tile: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for tile_id in tile_ids:
        tile_info = tile_info_map.get(str(tile_id))
        if tile_info is None:
            rows.append(
                {
                    "tile_id": str(tile_id),
                    "source_subset": "",
                    "disaster_name": "",
                    "target_exists": False,
                    "target_building_pixels": 0,
                    "dataset_instances": 0,
                    "full_json_instances": 0,
                    "dataset_building_pixels_raster": 0,
                    "full_json_building_pixels_raster": 0,
                    "dataset_missing_vs_full_json": 0,
                    "reason": "missing_tile_paths",
                }
            )
            continue
        gt_map, target_path = _load_official_target_map(tile_info, config)
        height, width = tile_info.image_size
        dataset_instances = dataset_instances_by_tile.get(str(tile_id), [])
        full_json_instances = full_json_instances_by_tile.get(str(tile_id), [])
        dataset_map = rasterize_instances_to_damage_map(dataset_instances, height, width, label_field="gt_label")
        full_json_map = rasterize_instances_to_damage_map(full_json_instances, height, width, label_field="gt_label")
        rows.append(
            {
                "tile_id": str(tile_id),
                "source_subset": tile_info.source_subset or "",
                "disaster_name": tile_info.disaster_name,
                "target_exists": bool(target_path),
                "target_building_pixels": int(0 if gt_map is None else int((gt_map > 0).sum())),
                "dataset_instances": int(len(dataset_instances)),
                "full_json_instances": int(len(full_json_instances)),
                "dataset_building_pixels_raster": int((dataset_map > 0).sum()),
                "full_json_building_pixels_raster": int((full_json_map > 0).sum()),
                "dataset_missing_vs_full_json": int(max(len(full_json_instances) - len(dataset_instances), 0)),
                "reason": "",
            }
        )
    write_csv_rows(
        Path(output_dir) / "geometry_reconstruction" / "tile_coverage.csv",
        [
            "tile_id",
            "source_subset",
            "disaster_name",
            "target_exists",
            "target_building_pixels",
            "dataset_instances",
            "full_json_instances",
            "dataset_building_pixels_raster",
            "full_json_building_pixels_raster",
            "dataset_missing_vs_full_json",
            "reason",
        ],
        rows,
    )
    write_json(Path(output_dir) / "geometry_reconstruction" / "tile_coverage.json", {"rows": rows})


def build_dilated_instances(
    instances_by_tile: dict[str, list[dict[str, Any]]],
    tile_info_map: dict[str, TileInfo],
    dilation_radius: int,
) -> dict[str, list[dict[str, Any]]]:
    del tile_info_map, dilation_radius
    return {
        tile_id: [dict(instance) for instance in items]
        for tile_id, items in instances_by_tile.items()
    }


def rasterize_instances_to_damage_map_with_dilation(
    instances: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    label_field: str,
    overlap_policy: str,
    dilation_radius: int = 0,
) -> np.ndarray:
    if int(dilation_radius) <= 0:
        return rasterize_instances_to_damage_map(
            instances,
            height=height,
            width=width,
            label_field=label_field,
            overlap_policy=overlap_policy,
        )
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"Invalid raster size: height={height}, width={width}.")
    structure = _disk_structure(int(dilation_radius))
    label_map = np.zeros((int(height), int(width)), dtype=np.uint8)
    area_winner = np.full((int(height), int(width)), fill_value=-1.0, dtype=np.float32)

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        building_id = item.get("building_idx", item.get("building_id", 0))
        sample_index = item.get("sample_index", 0)
        return int(building_id), int(sample_index)

    for instance in sorted(instances, key=sort_key):
        if label_field not in instance:
            raise KeyError(f"Instance is missing label field '{label_field}'.")
        pixel_label = int(instance[label_field]) + 1
        cropped_mask = instance.get("predicted_mask_cropped")
        mask_offset = instance.get("predicted_mask_offset")
        if cropped_mask is not None and mask_offset is not None:
            cropped_mask_arr = np.asarray(cropped_mask, dtype=bool)
            offset_x = int(mask_offset[0])
            offset_y = int(mask_offset[1])
            if cropped_mask_arr.ndim != 2:
                raise ValueError(
                    f"predicted_mask_cropped must be 2D, got shape {cropped_mask_arr.shape}."
                )
            if int(dilation_radius) > 0:
                cropped_mask_arr = ndimage.binary_dilation(cropped_mask_arr, structure=structure)
            mask_h, mask_w = cropped_mask_arr.shape
            x1 = max(0, offset_x)
            y1 = max(0, offset_y)
            x2 = min(int(width), offset_x + int(mask_w))
            y2 = min(int(height), offset_y + int(mask_h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop_x1 = x1 - offset_x
            crop_y1 = y1 - offset_y
            crop_x2 = crop_x1 + (x2 - x1)
            crop_y2 = crop_y1 + (y2 - y1)
            local_mask = cropped_mask_arr[crop_y1:crop_y2, crop_x1:crop_x2]
            label_view = label_map[y1:y2, x1:x2]
            if overlap_policy == "max_label":
                label_view[local_mask] = np.maximum(label_view[local_mask], pixel_label)
            elif overlap_policy == "last_wins":
                label_view[local_mask] = pixel_label
            elif overlap_policy == "area_larger_wins":
                region_area = float(local_mask.sum())
                area_view = area_winner[y1:y2, x1:x2]
                replace = local_mask & (region_area >= area_view)
                label_view[replace] = pixel_label
                area_view[replace] = region_area
            else:
                raise ValueError(
                    f"Unsupported overlap policy '{overlap_policy}'. "
                    "Expected one of: max_label, last_wins, area_larger_wins."
                )
            continue
        if instance.get("predicted_mask") is not None:
            mask = np.asarray(instance["predicted_mask"], dtype=bool)
            if mask.shape != (int(height), int(width)):
                raise ValueError(
                    f"predicted_mask shape mismatch: expected {(int(height), int(width))}, got {mask.shape}."
                )
        else:
            mask = polygon_to_mask(instance["polygon_xy"], height, width, offset=(0.0, 0.0)).astype(bool)
            mask = ndimage.binary_dilation(mask, structure=structure)
        if overlap_policy == "max_label":
            label_map[mask] = np.maximum(label_map[mask], pixel_label)
        elif overlap_policy == "last_wins":
            label_map[mask] = pixel_label
        elif overlap_policy == "area_larger_wins":
            region_area = float(mask.sum())
            replace = mask & (region_area >= area_winner)
            label_map[replace] = pixel_label
            area_winner[replace] = region_area
        else:
            raise ValueError(
                f"Unsupported overlap policy '{overlap_policy}'. "
                "Expected one of: max_label, last_wins, area_larger_wins."
            )
    return label_map


def _connected_components(binary_map: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, num_components = ndimage.label(binary_map.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    return labeled.astype(np.int32), int(num_components)


def build_target_component_oracle(
    dataset_instances_by_tile: dict[str, list[dict[str, Any]]],
    tile_info_map: dict[str, TileInfo],
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    pred_instances_by_tile: dict[str, list[dict[str, Any]]] = {}
    report_rows: list[dict[str, Any]] = []
    global_counts = Counter()
    for tile_id, dataset_instances in sorted(dataset_instances_by_tile.items(), key=lambda item: _tile_sort_key(item[0])):
        tile_info = tile_info_map.get(str(tile_id))
        if tile_info is None:
            continue
        gt_map, target_path = _load_official_target_map(tile_info, config)
        if gt_map is None:
            global_counts["tiles_missing_target"] += 1
            continue
        height, width = tile_info.image_size
        component_map, num_components = _connected_components(gt_map > 0)
        component_slices = ndimage.find_objects(component_map)
        global_counts["gt_components_total"] += int(num_components)
        component_sizes = np.bincount(component_map.reshape(-1), minlength=num_components + 1)
        best_match_by_component: dict[int, dict[str, Any]] = {}
        matched_components: set[int] = set()
        component_hit_counts: Counter[int] = Counter()
        for instance in dataset_instances:
            poly_mask = polygon_to_mask(instance["polygon_xy"], height, width, offset=(0.0, 0.0)).astype(bool)
            overlap_component_ids, overlap_counts = np.unique(component_map[poly_mask], return_counts=True)
            overlap_pairs = [
                (int(comp_id), int(count))
                for comp_id, count in zip(overlap_component_ids.tolist(), overlap_counts.tolist())
                if int(comp_id) > 0 and int(count) > 0
            ]
            if not overlap_pairs:
                report_rows.append(
                    {
                        "tile_id": str(tile_id),
                        "building_idx": int(instance["building_idx"]),
                        "matched_component_id": "",
                        "overlap_pixels": 0,
                        "polygon_pixels": int(poly_mask.sum()),
                        "component_pixels": 0,
                        "iou": 0.0,
                        "selected_for_component": False,
                        "duplicate_component_match": False,
                        "target_path": target_path,
                    }
                )
                global_counts["unmatched_sample_count"] += 1
                continue
            best_component_id, best_overlap = max(overlap_pairs, key=lambda item: (item[1], -item[0]))
            component_mask = component_map == int(best_component_id)
            union_pixels = int((poly_mask | component_mask).sum())
            iou = float(best_overlap / max(union_pixels, 1))
            selected = {
                "tile_id": str(tile_id),
                "building_idx": int(instance["building_idx"]),
                "matched_component_id": int(best_component_id),
                "overlap_pixels": int(best_overlap),
                "polygon_pixels": int(poly_mask.sum()),
                "component_pixels": int(component_sizes[int(best_component_id)]),
                "iou": iou,
                "selected_for_component": True,
                "duplicate_component_match": False,
                "target_path": target_path,
                "instance": instance,
            }
            report_rows.append(dict(selected))
            component_hit_counts[int(best_component_id)] += 1
            current_best = best_match_by_component.get(int(best_component_id))
            if current_best is None or (
                int(best_overlap),
                float(iou),
                -int(instance["building_idx"]),
            ) > (
                int(current_best["overlap_pixels"]),
                float(current_best["iou"]),
                -int(current_best["building_idx"]),
            ):
                if current_best is not None:
                    current_best["selected_for_component"] = False
                    current_best["duplicate_component_match"] = True
                best_match_by_component[int(best_component_id)] = selected
            else:
                selected["selected_for_component"] = False
                selected["duplicate_component_match"] = True
            if component_hit_counts[int(best_component_id)] > 1:
                global_counts["duplicate_component_matches"] += 1
        pred_items: list[dict[str, Any]] = []
        for component_id, match_info in sorted(best_match_by_component.items()):
            matched_components.add(int(component_id))
            instance = match_info["instance"]
            component_slice = component_slices[int(component_id) - 1]
            if component_slice is None:
                continue
            component_mask = (component_map[component_slice] == int(component_id)).astype(np.uint8)
            offset_y = int(component_slice[0].start or 0)
            offset_x = int(component_slice[1].start or 0)
            pred_items.append(
                {
                    **{
                        key: value
                        for key, value in instance.items()
                        if key not in {"predicted_mask", "predicted_mask_cropped", "predicted_mask_offset"}
                    },
                    "predicted_mask_cropped": component_mask,
                    "predicted_mask_offset": [offset_x, offset_y],
                    "pred_label": int(instance["gt_label"]),
                    "polygon_area": float(component_mask.sum()),
                }
            )
        pred_instances_by_tile[str(tile_id)] = pred_items
        global_counts["matched_component_count"] += int(len(matched_components))
        global_counts["unmatched_gt_component_count"] += int(max(num_components - len(matched_components), 0))
    metrics_extra = {
        "duplicate_component_matches": int(global_counts.get("duplicate_component_matches", 0)),
        "unmatched_sample_count": int(global_counts.get("unmatched_sample_count", 0)),
        "unmatched_gt_component_count": int(global_counts.get("unmatched_gt_component_count", 0)),
        "matched_component_count": int(global_counts.get("matched_component_count", 0)),
        "gt_components_total": int(global_counts.get("gt_components_total", 0)),
    }
    clean_rows: list[dict[str, Any]] = []
    for row in report_rows:
        clean_rows.append({key: value for key, value in row.items() if key != "instance"})
    return pred_instances_by_tile, clean_rows, metrics_extra


def _per_tile_average(per_tile_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_tile_metrics:
        return {}
    keys = [
        "localization_precision",
        "localization_recall",
        "localization_f1",
        "damage_f1_hmean",
        "damage_macro_f1",
        "damage_weighted_f1",
        "xview2_overall_score",
    ]
    averaged: dict[str, Any] = {}
    for key in keys:
        averaged[key] = float(np.mean([float(item.get(key, 0.0)) for item in per_tile_metrics]))
    return averaged


def _experiment_bridge_cfg(save_maps: bool, save_visuals: bool) -> dict[str, Any]:
    return {
        "save_pixel_predictions": bool(save_maps),
        "save_ground_truth": bool(save_maps),
        "save_official_style": bool(save_maps),
        "save_visuals": bool(save_visuals),
    }


def _save_experiment_outputs(
    output_dir: Path,
    *,
    tile_id: str,
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    save_maps: bool,
    save_visuals: bool,
) -> None:
    save_tile_outputs(
        output_dir,
        tile_id,
        pred_map,
        gt_map,
        bridge_cfg=_experiment_bridge_cfg(save_maps=save_maps, save_visuals=save_visuals),
    )


def run_bridge_experiment(
    *,
    experiment_name: str,
    output_dir: str | Path,
    tile_ids: list[str],
    tile_info_map: dict[str, TileInfo],
    pred_instances_by_tile: dict[str, list[dict[str, Any]]],
    gt_instances_by_tile: dict[str, list[dict[str, Any]]] | None,
    config: dict[str, Any],
    overlap_policy: str,
    use_official_target: bool,
    save_maps: bool,
    save_visuals: bool,
    extra_global_fields: dict[str, Any] | None = None,
    pred_dilation_radius: int = 0,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    per_tile_metrics: list[dict[str, Any]] = []
    skipped_tiles: list[dict[str, Any]] = []
    global_confusion = np.zeros((5, 5), dtype=np.int64)
    num_pred_instances_total = 0
    num_gt_instances_total = 0
    evaluated_tiles = 0
    for tile_id in sorted(tile_ids, key=_tile_sort_key):
        tile_info = tile_info_map.get(str(tile_id))
        if tile_info is None:
            skipped_tiles.append({"tile_id": str(tile_id), "reason": "missing_tile_paths"})
            continue
        height, width = tile_info.image_size
        pred_instances = pred_instances_by_tile.get(str(tile_id), [])
        gt_instances = [] if gt_instances_by_tile is None else gt_instances_by_tile.get(str(tile_id), [])
        pred_map = rasterize_instances_to_damage_map_with_dilation(
            pred_instances,
            height=height,
            width=width,
            label_field="pred_label",
            overlap_policy=overlap_policy,
            dilation_radius=int(pred_dilation_radius),
        )
        gt_source: str
        gt_map: np.ndarray
        if use_official_target:
            official_gt_map, gt_source = _load_official_target_map(tile_info, config)
            if official_gt_map is None:
                skipped_tiles.append({"tile_id": str(tile_id), "reason": "missing_official_target"})
                continue
            gt_map = official_gt_map
        else:
            gt_source = "rasterized_gt_instances"
            if gt_instances:
                gt_map = rasterize_instances_to_damage_map(
                    gt_instances,
                    height=height,
                    width=width,
                    label_field="gt_label",
                    overlap_policy=overlap_policy,
                )
            else:
                gt_map = np.zeros((height, width), dtype=np.uint8)
        tile_confusion = compute_confusion_matrix(pred_map=pred_map, gt_map=gt_map, num_classes=5)
        tile_metrics = compute_metrics_from_confusion_matrix(tile_confusion, num_classes=5)
        tile_metrics.update(
            {
                "experiment": experiment_name,
                "tile_id": str(tile_id),
                "gt_source": gt_source,
                "num_pred_instances": int(len(pred_instances)),
                "num_gt_instances": int(len(gt_instances)),
                "pred_building_pixels": int((pred_map > 0).sum()),
                "gt_building_pixels": int((gt_map > 0).sum()),
            }
        )
        per_tile_metrics.append(tile_metrics)
        global_confusion += tile_confusion
        num_pred_instances_total += int(len(pred_instances))
        num_gt_instances_total += int(len(gt_instances))
        evaluated_tiles += 1
        if save_maps or save_visuals:
            _save_experiment_outputs(
                Path(output_dir),
                tile_id=str(tile_id),
                pred_map=pred_map,
                gt_map=gt_map,
                save_maps=save_maps,
                save_visuals=save_visuals,
            )
    global_metrics = compute_metrics_from_confusion_matrix(global_confusion, num_classes=5)
    if extra_global_fields:
        global_metrics.update(extra_global_fields)
    gt_building_pixels_global = int(global_confusion[1:, :].sum())
    pred_building_pixels_global = int(global_confusion[:, 1:].sum())
    global_metrics["fn_over_gt_building_pixels"] = float(
        global_metrics["localization_fn"] / max(gt_building_pixels_global, 1)
    )
    global_metrics["fp_over_pred_building_pixels"] = float(
        global_metrics["localization_fp"] / max(pred_building_pixels_global, 1)
    )
    global_metrics["experiment"] = experiment_name
    global_metrics["num_tiles_requested"] = int(len(tile_ids))
    global_metrics["num_tiles_evaluated"] = int(evaluated_tiles)
    global_metrics["num_pred_instances"] = int(num_pred_instances_total)
    global_metrics["num_gt_instances"] = int(num_gt_instances_total)
    global_metrics["skipped_tiles"] = skipped_tiles
    confusion_payload = {
        "class_names": [PIXEL_CLASS_NAMES[idx] for idx in range(5)],
        "matrix": global_metrics["confusion_matrix_5x5"],
    }
    per_tile_average = _per_tile_average(per_tile_metrics)
    write_json(Path(output_dir) / "global_metrics.json", global_metrics)
    write_json(Path(output_dir) / "confusion_matrix_5x5.json", confusion_payload)
    write_json(Path(output_dir) / "per_tile_metrics.json", per_tile_metrics)
    write_json(Path(output_dir) / "skipped_tiles.json", skipped_tiles)
    class_report = build_class_report_text(
        global_metrics=global_metrics,
        per_tile_average=per_tile_average,
        num_tiles=int(evaluated_tiles),
        num_instances=int(num_pred_instances_total),
        num_gt_instances_in_evaluated_tiles=int(num_gt_instances_total),
    )
    Path(output_dir, "class_report_pixel.txt").write_text(class_report, encoding="utf-8")
    return {
        "experiment": experiment_name,
        "global_metrics": global_metrics,
        "per_tile_metrics": per_tile_metrics,
        "per_tile_average": per_tile_average,
        "skipped_tiles": skipped_tiles,
        "output_dir": str(output_dir),
    }


def run_direct_map_experiment(
    *,
    experiment_name: str,
    output_dir: str | Path,
    tile_ids: list[str],
    tile_info_map: dict[str, TileInfo],
    pred_maps_by_tile: dict[str, np.ndarray],
    gt_maps_by_tile: dict[str, np.ndarray],
    save_maps: bool,
    save_visuals: bool,
    extra_global_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    per_tile_metrics: list[dict[str, Any]] = []
    skipped_tiles: list[dict[str, Any]] = []
    global_confusion = np.zeros((5, 5), dtype=np.int64)
    for tile_id in sorted(tile_ids, key=_tile_sort_key):
        tile_info = tile_info_map.get(str(tile_id))
        if tile_info is None:
            skipped_tiles.append({"tile_id": str(tile_id), "reason": "missing_tile_paths"})
            continue
        pred_map = pred_maps_by_tile.get(str(tile_id))
        gt_map = gt_maps_by_tile.get(str(tile_id))
        if pred_map is None or gt_map is None:
            skipped_tiles.append({"tile_id": str(tile_id), "reason": "missing_pred_or_gt_map"})
            continue
        expected_shape = tuple(int(v) for v in tile_info.image_size)
        if tuple(pred_map.shape) != expected_shape or tuple(gt_map.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch for tile {tile_id}: expected {expected_shape}, "
                f"pred={tuple(pred_map.shape)} gt={tuple(gt_map.shape)}."
            )
        tile_confusion = compute_confusion_matrix(pred_map=pred_map, gt_map=gt_map, num_classes=5)
        tile_metrics = compute_metrics_from_confusion_matrix(tile_confusion, num_classes=5)
        tile_metrics.update(
            {
                "experiment": experiment_name,
                "tile_id": str(tile_id),
                "gt_source": "official_target_png",
                "num_pred_instances": 0,
                "num_gt_instances": 0,
                "pred_building_pixels": int((pred_map > 0).sum()),
                "gt_building_pixels": int((gt_map > 0).sum()),
            }
        )
        per_tile_metrics.append(tile_metrics)
        global_confusion += tile_confusion
        if save_maps or save_visuals:
            _save_experiment_outputs(
                Path(output_dir),
                tile_id=str(tile_id),
                pred_map=pred_map,
                gt_map=gt_map,
                save_maps=save_maps,
                save_visuals=save_visuals,
            )
    global_metrics = compute_metrics_from_confusion_matrix(global_confusion, num_classes=5)
    if extra_global_fields:
        global_metrics.update(extra_global_fields)
    global_metrics["experiment"] = experiment_name
    global_metrics["num_tiles_requested"] = int(len(tile_ids))
    global_metrics["num_tiles_evaluated"] = int(len(per_tile_metrics))
    global_metrics["num_pred_instances"] = 0
    global_metrics["num_gt_instances"] = 0
    global_metrics["skipped_tiles"] = skipped_tiles
    global_metrics["fn_over_gt_building_pixels"] = float(
        global_metrics["localization_fn"] / max(int(global_confusion[1:, :].sum()), 1)
    )
    global_metrics["fp_over_pred_building_pixels"] = float(
        global_metrics["localization_fp"] / max(int(global_confusion[:, 1:].sum()), 1)
    )
    write_json(Path(output_dir) / "global_metrics.json", global_metrics)
    write_json(
        Path(output_dir) / "confusion_matrix_5x5.json",
        {
            "class_names": [PIXEL_CLASS_NAMES[idx] for idx in range(5)],
            "matrix": global_metrics["confusion_matrix_5x5"],
        },
    )
    write_json(Path(output_dir) / "per_tile_metrics.json", per_tile_metrics)
    write_json(Path(output_dir) / "skipped_tiles.json", skipped_tiles)
    Path(output_dir, "class_report_pixel.txt").write_text(
        build_class_report_text(
            global_metrics=global_metrics,
            per_tile_average=_per_tile_average(per_tile_metrics),
            num_tiles=int(len(per_tile_metrics)),
            num_instances=0,
            num_gt_instances_in_evaluated_tiles=0,
        ),
        encoding="utf-8",
    )
    return {
        "experiment": experiment_name,
        "global_metrics": global_metrics,
        "per_tile_metrics": per_tile_metrics,
        "per_tile_average": _per_tile_average(per_tile_metrics),
        "skipped_tiles": skipped_tiles,
        "output_dir": str(output_dir),
    }


def write_component_match_report(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        Path(output_dir) / "component_match_report.csv",
        [
            "tile_id",
            "building_idx",
            "matched_component_id",
            "overlap_pixels",
            "polygon_pixels",
            "component_pixels",
            "iou",
            "selected_for_component",
            "duplicate_component_match",
            "target_path",
        ],
        rows,
    )
    write_json(Path(output_dir) / "component_match_report.json", {"rows": rows})


def build_area_bin_class_report(
    *,
    experiment_name: str,
    tile_ids: list[str],
    tile_info_map: dict[str, TileInfo],
    instances_by_tile: dict[str, list[dict[str, Any]]],
    pred_instances_by_tile: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    use_component_masks: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    area_stats: dict[tuple[int, str], dict[str, Any]] = {}
    minor_rows: list[dict[str, Any]] = []
    total_instances = int(sum(len(items) for items in instances_by_tile.values()))
    total_mask_pixels = 0
    for tile_id in tile_ids:
        tile_info = tile_info_map.get(str(tile_id))
        if tile_info is None:
            continue
        gt_map, _ = _load_official_target_map(tile_info, config)
        if gt_map is None:
            continue
        pred_items = pred_instances_by_tile.get(str(tile_id), [])
        height, width = tile_info.image_size
        pred_map = rasterize_instances_to_damage_map(pred_items, height, width, label_field="pred_label")
        component_map: np.ndarray | None = None
        component_sizes: np.ndarray | None = None
        if use_component_masks:
            component_map, _ = _connected_components(gt_map > 0)
            component_sizes = np.bincount(component_map.reshape(-1))
        pred_lookup = {int(item["building_idx"]): item for item in pred_items}
        for instance in instances_by_tile.get(str(tile_id), []):
            building_idx = int(instance["building_idx"])
            assigned_mask: np.ndarray
            component_id = None
            component_pixels = 0
            if use_component_masks and component_map is not None and component_sizes is not None:
                poly_mask = polygon_to_mask(instance["polygon_xy"], height, width, offset=(0.0, 0.0)).astype(bool)
                overlap_ids, overlap_counts = np.unique(component_map[poly_mask], return_counts=True)
                overlap_pairs = [
                    (int(comp_id), int(count))
                    for comp_id, count in zip(overlap_ids.tolist(), overlap_counts.tolist())
                    if int(comp_id) > 0 and int(count) > 0
                ]
                if overlap_pairs:
                    component_id, _ = max(overlap_pairs, key=lambda item: (item[1], -item[0]))
                    assigned_mask = component_map == int(component_id)
                    component_pixels = int(component_sizes[int(component_id)])
                else:
                    assigned_mask = poly_mask
            else:
                assigned_mask = polygon_to_mask(instance["polygon_xy"], height, width, offset=(0.0, 0.0)).astype(bool)
            mask_pixels = int(assigned_mask.sum())
            total_mask_pixels += mask_pixels
            gt_label = int(instance["gt_label"])
            pred_label = int(pred_lookup.get(building_idx, {}).get("pred_label", gt_label))
            area_bin = _area_bin_name(float(instance.get("polygon_area", mask_pixels)))
            key = (gt_label, area_bin)
            stats = area_stats.setdefault(
                key,
                {
                    "experiment": experiment_name,
                    "class_name": _label_name_from_index(gt_label),
                    "gt_label": int(gt_label),
                    "area_bin": area_bin,
                    "num_instances": 0,
                    "total_polygon_area": 0.0,
                    "total_mask_pixels": 0,
                    "correct_pixels": 0,
                    "wrong_pixels": 0,
                    "background_fn_pixels": 0,
                    "class_to_class_confusion": Counter(),
                },
            )
            stats["num_instances"] += 1
            stats["total_polygon_area"] += float(instance.get("polygon_area", 0.0))
            stats["total_mask_pixels"] += mask_pixels
            pred_counts = Counter(
                int(class_id) for class_id in pred_map[assigned_mask].reshape(-1).tolist()
            )
            gt_class_pixels = (gt_map == (gt_label + 1)) & assigned_mask
            correct_pixels = int(((pred_map == gt_map) & gt_class_pixels).sum())
            wrong_pixels = int((gt_class_pixels & (pred_map != gt_map)).sum())
            background_fn_pixels = int((gt_class_pixels & (pred_map == 0)).sum())
            stats["correct_pixels"] += correct_pixels
            stats["wrong_pixels"] += wrong_pixels
            stats["background_fn_pixels"] += background_fn_pixels
            for pred_pixel_label, count in pred_counts.items():
                stats["class_to_class_confusion"][int(pred_pixel_label)] += int(count)
            if gt_label == 1:
                probs = pred_lookup.get(building_idx, {}).get("final_class_probabilities") or []
                minor_gt_mask = (gt_map == 2) & assigned_mask
                minor_rows.append(
                    {
                        "experiment": experiment_name,
                        "tile_id": str(tile_id),
                        "building_idx": building_idx,
                        "polygon_area": float(instance.get("polygon_area", 0.0)),
                        "mask_pixels": mask_pixels,
                        "gt_label": gt_label,
                        "pred_label": pred_label,
                        "prob_no_damage": "" if len(probs) < 1 else float(probs[0]),
                        "prob_minor_damage": "" if len(probs) < 2 else float(probs[1]),
                        "prob_major_damage": "" if len(probs) < 3 else float(probs[2]),
                        "prob_destroyed": "" if len(probs) < 4 else float(probs[3]),
                        "minor_pixels_total": int(minor_gt_mask.sum()),
                        "minor_to_background": int((minor_gt_mask & (pred_map == 0)).sum()),
                        "minor_to_no_damage": int((minor_gt_mask & (pred_map == 1)).sum()),
                        "minor_to_major_damage": int((minor_gt_mask & (pred_map == 3)).sum()),
                        "minor_to_destroyed": int((minor_gt_mask & (pred_map == 4)).sum()),
                        "matched_component_id": "" if component_id is None else int(component_id),
                        "matched_component_pixels": int(component_pixels),
                    }
                )
    report_rows: list[dict[str, Any]] = []
    for (gt_label, area_bin), stats in sorted(area_stats.items(), key=lambda item: (item[0][0], AREA_BINS.index(next(bin_def for bin_def in AREA_BINS if bin_def[0] == item[0][1])))):
        report_rows.append(
            {
                "experiment": experiment_name,
                "class_name": stats["class_name"],
                "gt_label": int(gt_label),
                "area_bin": area_bin,
                "num_instances": int(stats["num_instances"]),
                "total_polygon_area": float(stats["total_polygon_area"]),
                "total_mask_pixels": int(stats["total_mask_pixels"]),
                "instance_count_ratio": float(stats["num_instances"] / max(total_instances, 1)),
                "pixel_area_ratio": float(stats["total_mask_pixels"] / max(total_mask_pixels, 1)),
                "correct_pixels": int(stats["correct_pixels"]),
                "wrong_pixels": int(stats["wrong_pixels"]),
                "background_fn_pixels": int(stats["background_fn_pixels"]),
                "class_to_class_confusion": _serialize_confusion_dict(stats["class_to_class_confusion"]),
            }
        )
    return report_rows, minor_rows


def write_area_and_minor_reports(
    output_dir: str | Path,
    area_rows: list[dict[str, Any]],
    minor_rows: list[dict[str, Any]],
) -> None:
    write_csv_rows(
        Path(output_dir) / "area_bin_class_report.csv",
        [
            "experiment",
            "class_name",
            "gt_label",
            "area_bin",
            "num_instances",
            "total_polygon_area",
            "total_mask_pixels",
            "instance_count_ratio",
            "pixel_area_ratio",
            "correct_pixels",
            "wrong_pixels",
            "background_fn_pixels",
            "class_to_class_confusion",
        ],
        area_rows,
    )
    write_csv_rows(
        Path(output_dir) / "minor_error_report.csv",
        [
            "experiment",
            "tile_id",
            "building_idx",
            "polygon_area",
            "mask_pixels",
            "gt_label",
            "pred_label",
            "prob_no_damage",
            "prob_minor_damage",
            "prob_major_damage",
            "prob_destroyed",
            "minor_pixels_total",
            "minor_to_background",
            "minor_to_no_damage",
            "minor_to_major_damage",
            "minor_to_destroyed",
            "matched_component_id",
            "matched_component_pixels",
        ],
        minor_rows,
    )


def write_prediction_records_jsonl(path: str | Path, prediction_records: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summary_row_from_experiment(
    experiment_name: str,
    global_metrics: dict[str, Any],
) -> dict[str, Any]:
    damage_per_class = global_metrics.get("damage_per_class", {})
    return {
        "experiment": experiment_name,
        "localization_f1": float(global_metrics.get("localization_f1", 0.0)),
        "localization_precision": float(global_metrics.get("localization_precision", 0.0)),
        "localization_recall": float(global_metrics.get("localization_recall", 0.0)),
        "damage_f1_hmean": float(global_metrics.get("damage_f1_hmean", 0.0)),
        "minor_f1": float(damage_per_class.get("minor_damage", {}).get("f1", 0.0)),
        "major_f1": float(damage_per_class.get("major_damage", {}).get("f1", 0.0)),
        "destroyed_f1": float(damage_per_class.get("destroyed", {}).get("f1", 0.0)),
        "no_damage_f1": float(damage_per_class.get("no_damage", {}).get("f1", 0.0)),
        "xview2_overall_score": float(global_metrics.get("xview2_overall_score", 0.0)),
        "localization_fp": int(global_metrics.get("localization_fp", 0)),
        "localization_fn": int(global_metrics.get("localization_fn", 0)),
        "num_tiles_evaluated": int(global_metrics.get("num_tiles_evaluated", 0)),
        "num_pred_instances": int(global_metrics.get("num_pred_instances", 0)),
        "num_gt_instances": int(global_metrics.get("num_gt_instances", 0)),
    }


def write_summary_table(output_dir: str | Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    output_dir = Path(output_dir)
    write_csv_rows(
        output_dir / "summary_table.csv",
        [
            "experiment",
            "localization_f1",
            "localization_precision",
            "localization_recall",
            "damage_f1_hmean",
            "minor_f1",
            "major_f1",
            "destroyed_f1",
            "no_damage_f1",
            "xview2_overall_score",
            "localization_fp",
            "localization_fn",
            "num_tiles_evaluated",
            "num_pred_instances",
            "num_gt_instances",
        ],
        rows,
    )
    write_json(output_dir / "summary_table.json", {"rows": rows, "warnings": warnings})


def write_per_tile_comparison(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        Path(output_dir) / "per_tile_comparison.csv",
        [
            "experiment",
            "tile_id",
            "gt_source",
            "num_pred_instances",
            "num_gt_instances",
            "pred_building_pixels",
            "gt_building_pixels",
            "localization_precision",
            "localization_recall",
            "localization_f1",
            "damage_f1_hmean",
            "xview2_overall_score",
            "localization_fp",
            "localization_fn",
        ],
        rows,
    )


def write_dilation_summary(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        Path(output_dir) / "summary.csv",
        [
            "dilation",
            "localization_precision",
            "localization_recall",
            "localization_f1",
            "damage_f1_hmean",
            "xview2_overall_score",
            "minor_f1",
            "major_f1",
            "destroyed_f1",
            "no_damage_f1",
            "localization_fp",
            "localization_fn",
        ],
        rows,
    )


def format_summary_table(rows: list[dict[str, Any]]) -> str:
    header = "Oracle Bridge Diagnosis Summary"
    divider = "=" * 60
    columns = [
        ("Experiment", "experiment", 32),
        ("Loc_F1", "localization_f1", 8),
        ("Loc_P", "localization_precision", 8),
        ("Loc_R", "localization_recall", 8),
        ("Damage_HMean", "damage_f1_hmean", 13),
        ("Minor_F1", "minor_f1", 9),
        ("Score", "xview2_overall_score", 8),
    ]
    lines = [header, divider]
    header_line = "  ".join(title.ljust(width) for title, _, width in columns)
    lines.append(header_line)
    for row in rows:
        parts: list[str] = []
        for _, key, width in columns:
            value = row.get(key, "")
            if key == "experiment":
                parts.append(str(value).ljust(width))
            else:
                parts.append(f"{float(value):<{width}.4f}")
        lines.append("  ".join(parts))
    lines.extend(
        [
            "",
            "Key Findings:",
            "- If target_png_self_check != 1.0, metric code has a critical bug.",
            "- If full_json_gt_polygon_bridge >> dataset_gt_polygon_bridge, dataset filtering causes oracle bridge missing buildings.",
            "- If dilation improves recall and score, polygon rasterization is too conservative.",
            "- If target_component_oracle_bridge >> polygon_raster, polygon/target alignment is the main issue.",
            "- If geometry oracle is high but current_model is low, classification/minor confusion is the main issue.",
        ]
    )
    return "\n".join(lines)
