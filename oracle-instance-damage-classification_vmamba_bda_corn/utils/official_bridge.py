from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.xbd_oracle_instance_damage import (
    CLASS_NAMES,
    LABEL_TO_INDEX,
    XBDOracleInstanceDamageDataset,
    _read_split_list,
    _resolve_tile_paths,
)
from utils.geometry import parse_wkt_polygon, polygon_area, polygon_to_mask
from utils.io import ensure_dir, read_json, write_json, write_text

OFFICIAL_BRIDGE_EVAL_NAME = "oracle-localization official-score bridge evaluation"
OFFICIAL_BRIDGE_EVAL_ALIAS = "GT-instance-to-pixel official-score bridge"
OFFICIAL_SCORE_REFERENCE_URL = "https://github.com/DIUx-xView/xView2_scoring/blob/master/xview2_metrics.py"
BRIDGE_MODE_CHOICES = ["oracle_localization"]
BRIDGE_TARGET_SOURCE_CHOICES = ["auto", "targets_png", "json_rasterized"]
BRIDGE_TIE_BREAK_CHOICES = ["deterministic_building_idx", "area_desc"]
OFFICIAL_IMAGE_HEIGHT = 1024
OFFICIAL_IMAGE_WIDTH = 1024
OFFICIAL_ALLOWED_VALUES = {0, 1, 2, 3, 4}
OFFICIAL_DAMAGE_VALUE_BY_LABEL_INDEX = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
}
OFFICIAL_DAMAGE_NAMES = {
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}


@dataclass(frozen=True)
class RawGTBuilding:
    building_idx: int
    uid: str | None
    subtype: str | None
    label_index: int | None
    polygon_xy: tuple[tuple[float, float], ...] | None
    polygon_area: float | None


@dataclass(frozen=True)
class TileContext:
    tile_id: str
    split: str
    source_subset: str
    post_label_path: Path
    post_target_png_path: Path | None
    image_height: int
    image_width: int
    raw_gt_building_count: int
    raw_supported_gt_building_count: int
    raw_rasterizable_gt_building_count: int
    invalid_polygon_count: int
    raw_buildings: tuple[RawGTBuilding, ...]


def _label_name(label_index: int | None) -> str | None:
    if label_index is None or not (0 <= int(label_index) < len(CLASS_NAMES)):
        return None
    return CLASS_NAMES[int(label_index)]


def _polygon_to_json_ready(points: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


def _as_float_list(values: list[Any] | tuple[Any, ...] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in values]


def _validate_bridge_mode(bridge_mode: str, bridge_target_source: str, bridge_tie_break: str, split_prefix: str) -> None:
    if bridge_mode not in BRIDGE_MODE_CHOICES:
        raise ValueError(f"Unsupported bridge_mode='{bridge_mode}'. Choices: {BRIDGE_MODE_CHOICES}")
    if bridge_target_source not in BRIDGE_TARGET_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported bridge_target_source='{bridge_target_source}'. Choices: {BRIDGE_TARGET_SOURCE_CHOICES}"
        )
    if bridge_tie_break not in BRIDGE_TIE_BREAK_CHOICES:
        raise ValueError(f"Unsupported bridge_tie_break='{bridge_tie_break}'. Choices: {BRIDGE_TIE_BREAK_CHOICES}")
    if "_" in split_prefix:
        raise ValueError(
            f"bridge_split_prefix='{split_prefix}' must not contain underscores because official-style filenames are underscore-delimited."
        )


def export_bridge_instance_records(
    dataset: XBDOracleInstanceDamageDataset,
    prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Export auditable instance-level bridge records from evaluated predictions."""

    bridge_records: list[dict[str, Any]] = []
    for record in prediction_records:
        sample_index = int(record["sample_index"])
        sample = dataset.samples[sample_index]
        gt_label_index = int(record["gt"])
        sample_label_index = int(sample["label"])
        if gt_label_index != sample_label_index:
            raise ValueError(
                f"GT mismatch for sample_index={sample_index}: prediction_records={gt_label_index}, dataset.samples={sample_label_index}"
            )

        pred_label_index = int(record["pred"])
        polygon_xy = tuple((float(x), float(y)) for x, y in sample["polygon_xy"])
        bridge_records.append(
            {
                "sample_index": sample_index,
                "tile_id": str(sample["tile_id"]),
                "building_idx": int(sample["building_idx"]),
                "uid": sample.get("uid"),
                "gt_label_index": gt_label_index,
                "pred_label_index": pred_label_index,
                "gt_label_name": CLASS_NAMES[gt_label_index],
                "pred_label_name": CLASS_NAMES[pred_label_index],
                "confidence": float(record["confidence"]),
                "probabilities": _as_float_list(record.get("probabilities")) or [],
                "threshold_probabilities": _as_float_list(record.get("threshold_probabilities")),
                "tau": None if record.get("tau") is None else float(record["tau"]),
                "raw_delta_tau": None if record.get("raw_delta_tau") is None else float(record["raw_delta_tau"]),
                "true_probability": float(record.get("true_probability", 0.0)),
                "difficulty": float(record.get("difficulty", 0.0)),
                "expected_severity": float(record.get("expected_severity", 0.0)),
                "gt_severity": float(record.get("gt_severity", 0.0)),
                "severity_abs_error": float(record.get("severity_abs_error", 0.0)),
                "polygon_area": float(sample["polygon_area"]),
                "polygon_xy": _polygon_to_json_ready(polygon_xy),
                "bbox_xyxy": [float(value) for value in sample["bbox_xyxy"]],
                "crop_bbox_xyxy": [int(value) for value in sample["crop_bbox_xyxy"]],
                "mask_pixels": int(sample.get("mask_pixels", 0)),
                "source_subset": str(sample["source_subset"]),
                "split": str(sample["split"]),
                "original_subtype": str(sample["original_subtype"]),
                "pre_image_path": str(sample["pre_image"]),
                "post_image_path": str(sample["post_image"]),
                "post_label_path": str(sample["post_label"]),
                "bridge_localization_value": 1,
                "bridge_damage_value": OFFICIAL_DAMAGE_VALUE_BY_LABEL_INDEX[pred_label_index],
                "dataset_trace": {
                    "sample_index": sample_index,
                    "tile_id": str(sample["tile_id"]),
                    "building_idx": int(sample["building_idx"]),
                    "post_label_path": str(sample["post_label"]),
                    "source_subset": str(sample["source_subset"]),
                    "split": str(sample["split"]),
                },
            }
        )
    return bridge_records


def _load_tile_context(post_label_path: Path, *, tile_id: str, split: str, source_subset: str) -> TileContext:
    payload = read_json(post_label_path)
    metadata = payload.get("metadata", {})
    image_width = int(metadata.get("width", metadata.get("original_width", OFFICIAL_IMAGE_WIDTH)))
    image_height = int(metadata.get("height", metadata.get("original_height", OFFICIAL_IMAGE_HEIGHT)))

    raw_gt_building_count = 0
    raw_supported_gt_building_count = 0
    raw_rasterizable_gt_building_count = 0
    invalid_polygon_count = 0
    raw_buildings: list[RawGTBuilding] = []

    for building_idx, feature in enumerate(payload.get("features", {}).get("xy", [])):
        properties = feature.get("properties", {})
        if properties.get("feature_type") != "building":
            continue
        raw_gt_building_count += 1
        subtype = properties.get("subtype")
        label_index = LABEL_TO_INDEX.get(subtype)
        if label_index is not None:
            raw_supported_gt_building_count += 1

        polygon_xy: tuple[tuple[float, float], ...] | None = None
        polygon_area_value: float | None = None
        try:
            polygon_xy = tuple((float(x), float(y)) for x, y in parse_wkt_polygon(feature.get("wkt", "")))
            polygon_area_value = float(polygon_area(polygon_xy))
        except Exception:
            invalid_polygon_count += 1

        if label_index is not None and polygon_xy is not None:
            raw_rasterizable_gt_building_count += 1

        raw_buildings.append(
            RawGTBuilding(
                building_idx=int(building_idx),
                uid=properties.get("uid"),
                subtype=subtype,
                label_index=label_index,
                polygon_xy=polygon_xy,
                polygon_area=polygon_area_value,
            )
        )

    post_target_png_path = post_label_path.parent.parent / "targets" / f"{tile_id}_post_disaster_target.png"
    return TileContext(
        tile_id=tile_id,
        split=split,
        source_subset=source_subset,
        post_label_path=post_label_path,
        post_target_png_path=post_target_png_path if post_target_png_path.exists() else None,
        image_height=image_height,
        image_width=image_width,
        raw_gt_building_count=raw_gt_building_count,
        raw_supported_gt_building_count=raw_supported_gt_building_count,
        raw_rasterizable_gt_building_count=raw_rasterizable_gt_building_count,
        invalid_polygon_count=invalid_polygon_count,
        raw_buildings=tuple(raw_buildings),
    )


def _build_tile_contexts(dataset: XBDOracleInstanceDamageDataset) -> tuple[dict[str, TileContext], dict[str, Any]]:
    requested_tile_ids = _read_split_list(dataset.list_path)
    tile_counter = Counter(requested_tile_ids)
    unique_tile_ids = sorted(tile_counter.keys())
    root_dir = dataset.config["data"]["root_dir"]
    allow_tier3 = bool(dataset.config["data"]["allow_tier3"])

    tile_contexts: dict[str, TileContext] = {}
    missing_tile_ids: list[str] = []
    for tile_id in unique_tile_ids:
        tile_paths = _resolve_tile_paths(root_dir, tile_id, allow_tier3=allow_tier3)
        if tile_paths is None:
            missing_tile_ids.append(tile_id)
            continue
        tile_contexts[tile_id] = _load_tile_context(
            Path(tile_paths.post_label),
            tile_id=tile_id,
            split=dataset.split_name,
            source_subset=str(tile_paths.source_subset),
        )

    return tile_contexts, {
        "requested_tile_count": len(requested_tile_ids),
        "unique_requested_tile_count": len(unique_tile_ids),
        "duplicate_tile_ids": sorted([tile_id for tile_id, count in tile_counter.items() if count > 1]),
        "missing_tile_ids": missing_tile_ids,
    }


def _make_bridge_img_id_map(tile_ids: list[str]) -> dict[str, str]:
    width = max(6, len(str(max(len(tile_ids) - 1, 0))))
    return {tile_id: f"{index:0{width}d}" for index, tile_id in enumerate(sorted(tile_ids))}


def _ensure_official_canvas(tile_context: TileContext) -> None:
    if tile_context.image_height != OFFICIAL_IMAGE_HEIGHT or tile_context.image_width != OFFICIAL_IMAGE_WIDTH:
        raise ValueError(
            f"Tile '{tile_context.tile_id}' has shape {(tile_context.image_height, tile_context.image_width)}; "
            f"official bridge evaluation requires {(OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH)}."
        )


def _load_png_array(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{path} has dtype={array.dtype}; expected uint8.")
    if array.shape != (OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH):
        raise ValueError(
            f"{path} has shape={array.shape}; expected {(OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH)}."
        )
    unique_values = set(np.unique(array).tolist())
    if not unique_values.issubset(OFFICIAL_ALLOWED_VALUES):
        raise ValueError(f"{path} has unsupported values {sorted(unique_values)}; expected subset of {sorted(OFFICIAL_ALLOWED_VALUES)}.")
    return array


def _resolve_existing_targets(tile_context: TileContext) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if tile_context.post_target_png_path is None:
        raise FileNotFoundError(f"No existing post-disaster target PNG found for tile '{tile_context.tile_id}'.")
    damage_target = _load_png_array(tile_context.post_target_png_path)
    localization_target = (damage_target > 0).astype(np.uint8)
    return localization_target, damage_target, {
        "target_source": "targets_png",
        "description": "existing xBD post-disaster target png adapted into official scorer layout",
        "source_path": str(tile_context.post_target_png_path),
    }


def _rasterize_json_targets(tile_context: TileContext) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _ensure_official_canvas(tile_context)
    localization_target = np.zeros((OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH), dtype=np.uint8)
    damage_target = np.zeros((OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH), dtype=np.uint8)
    cover_count = np.zeros((OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH), dtype=np.uint16)

    rasterized_buildings = 0
    skipped_unsupported_label_count = 0
    skipped_invalid_polygon_count = 0
    for building in sorted(tile_context.raw_buildings, key=lambda item: item.building_idx):
        if building.label_index is None:
            skipped_unsupported_label_count += 1
            continue
        if building.polygon_xy is None:
            skipped_invalid_polygon_count += 1
            continue
        mask = polygon_to_mask(building.polygon_xy, OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH).astype(bool)
        if not mask.any():
            skipped_invalid_polygon_count += 1
            continue
        cover_count += mask.astype(np.uint16)
        localization_target[mask] = 1
        damage_target[mask] = OFFICIAL_DAMAGE_VALUE_BY_LABEL_INDEX[int(building.label_index)]
        rasterized_buildings += 1

    return localization_target, damage_target, {
        "target_source": "json_rasterized",
        "description": "official-format local target reproduction from GT json",
        "source_path": str(tile_context.post_label_path),
        "rasterized_building_count": rasterized_buildings,
        "skipped_unsupported_label_count": skipped_unsupported_label_count,
        "skipped_invalid_polygon_count": skipped_invalid_polygon_count,
        "overlap_pixels": int(np.count_nonzero(cover_count > 1)),
    }


def _resolve_target_arrays(
    tile_context: TileContext,
    bridge_target_source: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if bridge_target_source == "targets_png":
        return _resolve_existing_targets(tile_context)
    if bridge_target_source == "json_rasterized":
        return _rasterize_json_targets(tile_context)

    try:
        return _resolve_existing_targets(tile_context)
    except Exception as error:
        localization_target, damage_target, target_info = _rasterize_json_targets(tile_context)
        target_info["auto_fallback_reason"] = str(error)
        return localization_target, damage_target, target_info


def _prediction_sort_key(record: dict[str, Any], bridge_tie_break: str) -> tuple[float, int, int]:
    if bridge_tie_break == "area_desc":
        return (-float(record["polygon_area"]), int(record["building_idx"]), int(record["sample_index"]))
    return (float(record["building_idx"]), int(record["sample_index"]), 0)


def _rasterize_prediction_tile(
    tile_id: str,
    bridge_records: list[dict[str, Any]],
    bridge_tie_break: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    localization_prediction = np.zeros((OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH), dtype=np.uint8)
    damage_prediction = np.zeros((OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH), dtype=np.uint8)
    cover_count = np.zeros((OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH), dtype=np.uint16)

    valid_records: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    failed_instances: list[dict[str, Any]] = []

    for record in bridge_records:
        polygon_xy = tuple((float(x), float(y)) for x, y in record["polygon_xy"])
        if not polygon_xy:
            failed_instances.append(
                {
                    "tile_id": tile_id,
                    "sample_index": int(record["sample_index"]),
                    "building_idx": int(record["building_idx"]),
                    "reason": "missing_polygon_xy",
                }
            )
            continue

        mask = polygon_to_mask(polygon_xy, OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH).astype(bool)
        if not mask.any():
            failed_instances.append(
                {
                    "tile_id": tile_id,
                    "sample_index": int(record["sample_index"]),
                    "building_idx": int(record["building_idx"]),
                    "reason": "polygon_rasterized_to_zero_pixels",
                }
            )
            continue

        cover_count += mask.astype(np.uint16)
        valid_records.append(record)
        masks.append(mask)

    conflict_zone = cover_count > 1
    conflicted_instances: list[dict[str, Any]] = []
    for record, mask in zip(valid_records, masks):
        if np.any(mask & conflict_zone):
            conflicted_instances.append(
                {
                    "tile_id": tile_id,
                    "sample_index": int(record["sample_index"]),
                    "building_idx": int(record["building_idx"]),
                }
            )

    ordered_pairs = sorted(
        zip(valid_records, masks),
        key=lambda item: _prediction_sort_key(item[0], bridge_tie_break),
    )
    for record, mask in ordered_pairs:
        localization_prediction[mask] = 1
        damage_prediction[mask] = OFFICIAL_DAMAGE_VALUE_BY_LABEL_INDEX[int(record["pred_label_index"])]

    return localization_prediction, damage_prediction, {
        "raw_instance_count": len(bridge_records),
        "bridged_instance_count": len(valid_records),
        "failed_instance_count": len(failed_instances),
        "failed_instances": failed_instances,
        "overlap_pixels": int(np.count_nonzero(conflict_zone)),
        "num_conflicted_instances": len(conflicted_instances),
        "conflicted_instances": conflicted_instances,
    }


def _write_png(path: Path, array: np.ndarray) -> None:
    if array.dtype != np.uint8:
        raise ValueError(f"{path} has dtype={array.dtype}; expected uint8.")
    if array.shape != (OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH):
        raise ValueError(
            f"{path} has shape={array.shape}; expected {(OFFICIAL_IMAGE_HEIGHT, OFFICIAL_IMAGE_WIDTH)}."
        )
    ensure_dir(path.parent)
    Image.fromarray(array, mode="L").save(path)


def _compute_tp_fn_fp(pred: np.ndarray, targ: np.ndarray, positive_class: int) -> tuple[int, int, int]:
    true_positive = int(np.logical_and(pred == positive_class, targ == positive_class).sum())
    false_negative = int(np.logical_and(pred != positive_class, targ == positive_class).sum())
    false_positive = int(np.logical_and(pred == positive_class, targ != positive_class).sum())
    return true_positive, false_negative, false_positive


def _precision(tp: int, fp: int) -> float:
    if tp == 0:
        return 0.0
    return float(tp / (tp + fp))


def _recall(tp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    return float(tp / (tp + fn))


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _precision(tp, fp)
    recall = _recall(tp, fn)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def compute_official_xview2_bridge_score(
    pred_dir: Path,
    targ_dir: Path,
    *,
    split_prefix: str,
) -> dict[str, Any]:
    """Compute an xView2-scoring-equivalent metric set over bridge PNGs."""

    localization_target_paths = sorted(targ_dir.glob(f"{split_prefix}_localization_*_target.png"))
    if not localization_target_paths:
        raise FileNotFoundError(
            f"No target files matching '{split_prefix}_localization_*_target.png' were found under {targ_dir}."
        )

    localization_counts = {"tp": 0, "fp": 0, "fn": 0}
    damage_counts = {class_value: {"tp": 0, "fp": 0, "fn": 0} for class_value in range(1, 5)}
    img_ids: list[str] = []

    for localization_target_path in localization_target_paths:
        parts = localization_target_path.stem.split("_")
        if len(parts) != 4:
            raise ValueError(
                f"Unexpected target filename '{localization_target_path.name}'. Expected four underscore-delimited tokens."
            )
        _, _, img_id, suffix = parts
        if suffix != "target":
            raise ValueError(f"Unexpected target filename '{localization_target_path.name}'.")

        localization_prediction_path = pred_dir / f"{split_prefix}_localization_{img_id}_prediction.png"
        damage_prediction_path = pred_dir / f"{split_prefix}_damage_{img_id}_prediction.png"
        damage_target_path = targ_dir / f"{split_prefix}_damage_{img_id}_target.png"

        localization_prediction = _load_png_array(localization_prediction_path)
        damage_prediction = _load_png_array(damage_prediction_path)
        localization_target = _load_png_array(localization_target_path)
        damage_target = _load_png_array(damage_target_path)

        localization_prediction_buildings = (localization_prediction > 0).astype(np.uint8)
        localization_target_buildings = (localization_target > 0).astype(np.uint8)
        damage_target_buildings = (damage_target > 0).astype(np.uint8)

        ltp, lfn, lfp = _compute_tp_fn_fp(localization_prediction_buildings, localization_target_buildings, 1)
        localization_counts["tp"] += ltp
        localization_counts["fn"] += lfn
        localization_counts["fp"] += lfp

        masked_damage_prediction = damage_prediction * localization_prediction_buildings
        damage_prediction_eval = masked_damage_prediction[damage_target_buildings == 1]
        damage_target_eval = damage_target[damage_target_buildings == 1]
        for class_value in range(1, 5):
            dtp, dfn, dfp = _compute_tp_fn_fp(damage_prediction_eval, damage_target_eval, class_value)
            damage_counts[class_value]["tp"] += dtp
            damage_counts[class_value]["fn"] += dfn
            damage_counts[class_value]["fp"] += dfp
        img_ids.append(img_id)

    localization_f1 = _f1(
        localization_counts["tp"],
        localization_counts["fp"],
        localization_counts["fn"],
    )
    damage_f1_values = {
        class_value: _f1(counts["tp"], counts["fp"], counts["fn"])
        for class_value, counts in damage_counts.items()
    }
    harmonic_mean_denominator = sum((damage_f1_values[class_value] + 1e-6) ** -1 for class_value in range(1, 5))
    damage_f1 = float(4.0 / harmonic_mean_denominator) if harmonic_mean_denominator > 0 else 0.0
    score = float(0.3 * localization_f1 + 0.7 * damage_f1)

    per_class_details: dict[str, Any] = {}
    for class_value in range(1, 5):
        counts = damage_counts[class_value]
        per_class_details[OFFICIAL_DAMAGE_NAMES[class_value]] = {
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "precision": _precision(counts["tp"], counts["fp"]),
            "recall": _recall(counts["tp"], counts["fn"]),
            "f1": damage_f1_values[class_value],
        }

    return {
        "bridge_evaluation_name": OFFICIAL_BRIDGE_EVAL_NAME,
        "bridge_evaluation_alias": OFFICIAL_BRIDGE_EVAL_ALIAS,
        "official_logic_reference": OFFICIAL_SCORE_REFERENCE_URL,
        "official_logic_equivalence": (
            "Equivalent local scorer wrapper matching DIUx-xView/xView2_scoring xview2_metrics.py "
            "(damage predictions are masked by predicted buildings and evaluated only where target buildings exist)."
        ),
        "split_prefix": split_prefix,
        "num_tiles": len(img_ids),
        "img_ids": img_ids,
        "localization_f1": localization_f1,
        "damage_f1_no_damage": damage_f1_values[1],
        "damage_f1_minor_damage": damage_f1_values[2],
        "damage_f1_major_damage": damage_f1_values[3],
        "damage_f1_destroyed": damage_f1_values[4],
        "damage_f1": damage_f1,
        "score": score,
        "localization_counts": {
            "tp": int(localization_counts["tp"]),
            "fp": int(localization_counts["fp"]),
            "fn": int(localization_counts["fn"]),
            "precision": _precision(localization_counts["tp"], localization_counts["fp"]),
            "recall": _recall(localization_counts["tp"], localization_counts["fn"]),
            "f1": localization_f1,
        },
        "damage_counts": per_class_details,
    }


def _build_metrics_text(metrics: dict[str, Any]) -> str:
    lines = [
        OFFICIAL_BRIDGE_EVAL_NAME,
        "",
        f"localization_f1={metrics['localization_f1']:.6f}",
        f"damage_f1_no_damage={metrics['damage_f1_no_damage']:.6f}",
        f"damage_f1_minor_damage={metrics['damage_f1_minor_damage']:.6f}",
        f"damage_f1_major_damage={metrics['damage_f1_major_damage']:.6f}",
        f"damage_f1_destroyed={metrics['damage_f1_destroyed']:.6f}",
        f"damage_f1={metrics['damage_f1']:.6f}",
        f"score={metrics['score']:.6f}",
        "",
        "This is an oracle-localization official-score bridge result in official scoring space.",
        "It is not an end-to-end pixel-level damage assessment score.",
    ]
    return "\n".join(lines) + "\n"


def _build_notes_text(
    *,
    bridge_mode: str,
    bridge_target_source: str,
    bridge_tie_break: str,
    target_source_counter: Counter[str],
    coverage_audit: dict[str, Any],
) -> str:
    tie_break_description = {
        "deterministic_building_idx": (
            "strict polygon fill with instances written in ascending building_idx order; later writes overwrite earlier pixels"
        ),
        "area_desc": (
            "strict polygon fill with instances written in descending polygon area order; later writes overwrite earlier pixels"
        ),
    }[bridge_tie_break]

    lines = [
        f"# {OFFICIAL_BRIDGE_EVAL_NAME}",
        "",
        f"- Alias: {OFFICIAL_BRIDGE_EVAL_ALIAS}",
        f"- bridge_mode: {bridge_mode}",
        f"- requested_target_source: {bridge_target_source}",
        f"- realized_target_sources: {dict(target_source_counter)}",
        f"- tie_break_policy: {bridge_tie_break}",
        f"- tie_break_description: {tie_break_description}",
        "",
        "- Localization geometry comes from GT instance polygons that this repository actually indexed and evaluated.",
        "- This is not an end-to-end pixel-level damage assessment method.",
        "- The reported overall score is a conditional upper-bound reference in official scoring space and should not be used as a fair apples-to-apples comparison against end-to-end methods.",
        "- damage_f1 is the more appropriate indicator for classifier capacity under known building boundaries.",
        "- Default bridge rasterization uses strict polygon fill only; no dilation, erosion, blur, smoothing, or confidence-based spatial arbitration is applied.",
        "- Prediction coverage is limited to the eligible repo instances. Ground-truth targets still reflect the full tile-level GT, so dropped buildings remain auditable.",
        "- Local scorer is implemented as a strict equivalent wrapper of the official xView2 metric logic.",
        f"- Official logic reference: {OFFICIAL_SCORE_REFERENCE_URL}",
        "",
        "## Coverage",
        f"- raw_gt_building_count={coverage_audit['raw_gt_building_count']}",
        f"- eligible_repo_instance_count={coverage_audit['eligible_repo_instance_count']}",
        f"- bridged_instance_count={coverage_audit['bridged_instance_count']}",
        f"- dropped_gt_buildings_count={coverage_audit['dropped_gt_buildings_count']}",
        f"- dropped_ratio={coverage_audit['dropped_ratio']:.6f}",
        "",
        "## Target Note",
        "- If any tile used json_rasterized targets, that tile is an official-format local target reproduction from GT json rather than an original released target pair.",
    ]
    return "\n".join(lines) + "\n"


def run_official_bridge_evaluation(
    *,
    dataset: XBDOracleInstanceDamageDataset,
    prediction_records: list[dict[str, Any]],
    eval_dir: str | Path,
    instance_metrics: dict[str, Any],
    bridge_mode: str = "oracle_localization",
    bridge_target_source: str = "auto",
    bridge_output_dir: str | Path | None = None,
    bridge_split_prefix: str = "test",
    bridge_tie_break: str = "deterministic_building_idx",
    save_bridge_instance_records: bool = True,
) -> dict[str, Any]:
    """Bridge instance-level oracle predictions into official xView2 scoring space."""

    _validate_bridge_mode(bridge_mode, bridge_target_source, bridge_tie_break, bridge_split_prefix)
    if bridge_mode != "oracle_localization":
        raise ValueError(
            f"bridge_mode='{bridge_mode}' is not supported. This repository only implements oracle-localization bridge evaluation."
        )

    output_dir = ensure_dir(Path(bridge_output_dir) if bridge_output_dir is not None else Path(eval_dir) / "official_bridge")
    pred_dir = ensure_dir(output_dir / "predictions")
    targ_dir = ensure_dir(output_dir / "targets")

    bridge_instance_records = export_bridge_instance_records(dataset, prediction_records)
    if save_bridge_instance_records:
        write_json(output_dir / "bridge_instance_predictions.json", bridge_instance_records)

    tile_contexts, split_audit = _build_tile_contexts(dataset)
    records_by_tile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in bridge_instance_records:
        records_by_tile[str(record["tile_id"])].append(record)

    extra_prediction_tiles = sorted(set(records_by_tile.keys()) - set(tile_contexts.keys()))
    if extra_prediction_tiles:
        raise ValueError(
            "Bridge predictions contain tiles that are absent from the resolved evaluation split: "
            + ", ".join(extra_prediction_tiles[:10])
        )

    bridge_img_id_map = _make_bridge_img_id_map(list(tile_contexts.keys()))
    eligible_counts = Counter(str(sample["tile_id"]) for sample in dataset.samples)

    target_source_counter: Counter[str] = Counter()
    tile_manifest_entries: list[dict[str, Any]] = []
    overlap_pixels_per_tile: dict[str, int] = {}
    failed_bridge_instances_per_tile: dict[str, list[dict[str, Any]]] = {}
    target_generation_per_tile: dict[str, Any] = {}

    coverage_totals = {
        "raw_gt_building_count": 0,
        "raw_supported_gt_building_count": 0,
        "raw_rasterizable_gt_building_count": 0,
        "eligible_repo_instance_count": 0,
        "bridged_instance_count": 0,
        "eligible_but_unbridged_count": 0,
    }
    conflict_instance_keys: set[tuple[str, int]] = set()

    for tile_id in sorted(tile_contexts.keys()):
        tile_context = tile_contexts[tile_id]
        _ensure_official_canvas(tile_context)
        tile_bridge_img_id = bridge_img_id_map[tile_id]
        tile_bridge_records = records_by_tile.get(tile_id, [])
        localization_prediction, damage_prediction, prediction_audit = _rasterize_prediction_tile(
            tile_id,
            tile_bridge_records,
            bridge_tie_break,
        )
        localization_target, damage_target, target_info = _resolve_target_arrays(tile_context, bridge_target_source)

        target_source_counter[str(target_info["target_source"])] += 1
        target_generation_per_tile[tile_id] = target_info

        localization_prediction_path = pred_dir / f"{bridge_split_prefix}_localization_{tile_bridge_img_id}_prediction.png"
        damage_prediction_path = pred_dir / f"{bridge_split_prefix}_damage_{tile_bridge_img_id}_prediction.png"
        localization_target_path = targ_dir / f"{bridge_split_prefix}_localization_{tile_bridge_img_id}_target.png"
        damage_target_path = targ_dir / f"{bridge_split_prefix}_damage_{tile_bridge_img_id}_target.png"

        _write_png(localization_prediction_path, localization_prediction)
        _write_png(damage_prediction_path, damage_prediction)
        _write_png(localization_target_path, localization_target)
        _write_png(damage_target_path, damage_target)

        for conflict_record in prediction_audit["conflicted_instances"]:
            conflict_instance_keys.add((str(conflict_record["tile_id"]), int(conflict_record["sample_index"])))
        if prediction_audit["failed_instances"]:
            failed_bridge_instances_per_tile[tile_id] = prediction_audit["failed_instances"]

        eligible_count = int(eligible_counts.get(tile_id, 0))
        bridged_count = int(prediction_audit["bridged_instance_count"])
        dropped_count = max(int(tile_context.raw_gt_building_count) - bridged_count, 0)
        eligible_but_unbridged_count = max(eligible_count - bridged_count, 0)
        overlap_pixels = int(prediction_audit["overlap_pixels"])
        overlap_pixels_per_tile[tile_id] = overlap_pixels

        coverage_totals["raw_gt_building_count"] += int(tile_context.raw_gt_building_count)
        coverage_totals["raw_supported_gt_building_count"] += int(tile_context.raw_supported_gt_building_count)
        coverage_totals["raw_rasterizable_gt_building_count"] += int(tile_context.raw_rasterizable_gt_building_count)
        coverage_totals["eligible_repo_instance_count"] += eligible_count
        coverage_totals["bridged_instance_count"] += bridged_count
        coverage_totals["eligible_but_unbridged_count"] += eligible_but_unbridged_count

        tile_manifest_entries.append(
            {
                "bridge_img_id": tile_bridge_img_id,
                "raw_tile_id": tile_id,
                "source_subset": tile_context.source_subset,
                "split": tile_context.split,
                "num_instances": bridged_count,
                "num_overlap_pixels": overlap_pixels,
                "coverage": {
                    "raw_gt_building_count": int(tile_context.raw_gt_building_count),
                    "raw_supported_gt_building_count": int(tile_context.raw_supported_gt_building_count),
                    "raw_rasterizable_gt_building_count": int(tile_context.raw_rasterizable_gt_building_count),
                    "eligible_repo_instance_count": eligible_count,
                    "bridged_instance_count": bridged_count,
                    "eligible_but_unbridged_count": eligible_but_unbridged_count,
                    "dropped_gt_buildings_count": dropped_count,
                    "dropped_ratio": float(dropped_count / tile_context.raw_gt_building_count)
                    if tile_context.raw_gt_building_count > 0
                    else 0.0,
                },
                "prediction_audit": {
                    "raw_instance_count": int(prediction_audit["raw_instance_count"]),
                    "bridged_instance_count": bridged_count,
                    "failed_instance_count": int(prediction_audit["failed_instance_count"]),
                    "num_conflicted_instances": int(prediction_audit["num_conflicted_instances"]),
                },
                "target_source": str(target_info["target_source"]),
                "target_source_description": str(target_info["description"]),
                "post_label_path": str(tile_context.post_label_path),
                "prediction_files": {
                    "localization": str(localization_prediction_path),
                    "damage": str(damage_prediction_path),
                },
                "target_files": {
                    "localization": str(localization_target_path),
                    "damage": str(damage_target_path),
                },
            }
        )

    coverage_audit = {
        "bridge_evaluation_name": OFFICIAL_BRIDGE_EVAL_NAME,
        "tiles_in_split_count": int(split_audit["unique_requested_tile_count"]),
        "scored_tiles_count": len(tile_manifest_entries),
        "missing_tile_paths_count": len(split_audit["missing_tile_ids"]),
        "missing_tile_ids": split_audit["missing_tile_ids"],
        "duplicate_tile_ids": split_audit["duplicate_tile_ids"],
        "raw_gt_building_count": int(coverage_totals["raw_gt_building_count"]),
        "raw_supported_gt_building_count": int(coverage_totals["raw_supported_gt_building_count"]),
        "raw_rasterizable_gt_building_count": int(coverage_totals["raw_rasterizable_gt_building_count"]),
        "eligible_repo_instance_count": int(coverage_totals["eligible_repo_instance_count"]),
        "bridged_instance_count": int(coverage_totals["bridged_instance_count"]),
        "eligible_but_unbridged_count": int(coverage_totals["eligible_but_unbridged_count"]),
        "dropped_gt_buildings_count": max(
            int(coverage_totals["raw_gt_building_count"]) - int(coverage_totals["bridged_instance_count"]),
            0,
        ),
        "dropped_ratio": float(
            max(int(coverage_totals["raw_gt_building_count"]) - int(coverage_totals["bridged_instance_count"]), 0)
            / coverage_totals["raw_gt_building_count"]
        )
        if coverage_totals["raw_gt_building_count"] > 0
        else 0.0,
        "per_tile": {
            entry["raw_tile_id"]: entry["coverage"]
            for entry in tile_manifest_entries
        },
    }

    rasterization_audit = {
        "bridge_evaluation_name": OFFICIAL_BRIDGE_EVAL_NAME,
        "tie_break_policy": bridge_tie_break,
        "tie_break_description": {
            "deterministic_building_idx": (
                "Instances are written in ascending building_idx order and later writes overwrite earlier pixels."
            ),
            "area_desc": (
                "Instances are written in descending polygon area order and later writes overwrite earlier pixels."
            ),
        }[bridge_tie_break],
        "overlap_pixels_total": int(sum(overlap_pixels_per_tile.values())),
        "overlap_pixels_per_tile": overlap_pixels_per_tile,
        "conflicted_tiles_count": int(sum(1 for value in overlap_pixels_per_tile.values() if value > 0)),
        "num_tiles_with_overlap": int(sum(1 for value in overlap_pixels_per_tile.values() if value > 0)),
        "num_conflicted_instances": len(conflict_instance_keys),
        "failed_bridge_instances_count": int(sum(len(values) for values in failed_bridge_instances_per_tile.values())),
        "failed_bridge_instances_per_tile": failed_bridge_instances_per_tile,
    }

    metrics = compute_official_xview2_bridge_score(pred_dir, targ_dir, split_prefix=bridge_split_prefix)
    metrics["bridge_mode"] = bridge_mode
    metrics["target_source_request"] = bridge_target_source
    metrics["target_source_realized_counts"] = dict(target_source_counter)
    metrics["tie_break_policy"] = bridge_tie_break
    write_json(output_dir / "metrics.json", metrics)
    write_text(output_dir / "metrics.txt", _build_metrics_text(metrics))

    tile_manifest = {
        "bridge_evaluation_name": OFFICIAL_BRIDGE_EVAL_NAME,
        "bridge_evaluation_alias": OFFICIAL_BRIDGE_EVAL_ALIAS,
        "bridge_mode": bridge_mode,
        "split_prefix": bridge_split_prefix,
        "tie_break_policy": bridge_tie_break,
        "entries": tile_manifest_entries,
    }
    write_json(output_dir / "tile_manifest.json", tile_manifest)
    write_json(output_dir / "coverage_audit.json", coverage_audit)
    write_json(output_dir / "rasterization_audit.json", rasterization_audit)

    bridge_vs_instance_summary = {
        "instance_macro_f1": float(instance_metrics.get("macro_f1", 0.0)),
        "instance_weighted_f1": float(instance_metrics.get("weighted_f1", 0.0)),
        "instance_qwk": float(instance_metrics.get("quadratic_weighted_kappa", 0.0)),
        "instance_far_error_rate": float(instance_metrics.get("far_error_rate", 0.0)),
        "bridge_localization_f1": float(metrics["localization_f1"]),
        "bridge_damage_f1": float(metrics["damage_f1"]),
        "bridge_score": float(metrics["score"]),
    }
    write_json(output_dir / "bridge_vs_instance_summary.json", bridge_vs_instance_summary)

    write_json(output_dir / "target_generation_summary.json", target_generation_per_tile)
    write_text(
        output_dir / "README.md",
        _build_notes_text(
            bridge_mode=bridge_mode,
            bridge_target_source=bridge_target_source,
            bridge_tie_break=bridge_tie_break,
            target_source_counter=target_source_counter,
            coverage_audit=coverage_audit,
        ),
    )

    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "coverage_audit": coverage_audit,
        "rasterization_audit": rasterization_audit,
        "tile_manifest": tile_manifest,
        "bridge_vs_instance_summary": bridge_vs_instance_summary,
    }
