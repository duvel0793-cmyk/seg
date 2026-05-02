from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

from utils.geometry import polygon_to_mask
from utils.io import ensure_dir, read_json
from utils.metrics import save_matrix_heatmap_plot

PIXEL_DAMAGE_CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
INSTANCE_TO_PIXEL_LABEL_OFFSET = 1
OFFICIAL_DAMAGE_CLASS_IDS = [1, 2, 3, 4]
OFFICIAL_DAMAGE_CLASS_SHORT_NAMES = {
    1: "no_damage",
    2: "minor",
    3: "major",
    4: "destroyed",
}
LABEL_COLORS = np.asarray(
    [
        [0, 0, 0],
        [78, 121, 167],
        [242, 142, 43],
        [225, 87, 89],
        [118, 183, 178],
    ],
    dtype=np.uint8,
)


def convert_instance_label_to_pixel_label(instance_label: int) -> int:
    return int(instance_label) + INSTANCE_TO_PIXEL_LABEL_OFFSET


def colorize_damage_label_map(label_map: np.ndarray) -> np.ndarray:
    label_array = np.asarray(label_map, dtype=np.int64)
    clipped = np.clip(label_array, 0, len(LABEL_COLORS) - 1)
    return LABEL_COLORS[clipped]


def save_label_png(path: str | Path, label_map: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(np.asarray(label_map, dtype=np.uint8), mode="L").save(path)


def save_colorized_label_png(path: str | Path, label_map: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(colorize_damage_label_map(label_map), mode="RGB").save(path)


def _resolve_target_candidates(tile_id: str) -> dict[str, list[str]]:
    return {
        "damage": [
            f"{tile_id}_post_disaster_target.png",
            f"{tile_id}_damage_target.png",
            f"{tile_id}_post_disaster_damage_target.png",
        ],
        "localization": [
            f"{tile_id}_pre_disaster_target.png",
            f"{tile_id}_localization_target.png",
            f"{tile_id}_building_target.png",
        ],
    }


def resolve_target_png_paths(tile_sample: dict[str, Any]) -> dict[str, Path | None]:
    post_image_path = Path(tile_sample["post_image"])
    subset_dir = post_image_path.parent.parent
    targets_dir = subset_dir / "targets"
    candidates = _resolve_target_candidates(str(tile_sample["tile_id"]))
    resolved: dict[str, Path | None] = {"damage": None, "localization": None}
    for key, names in candidates.items():
        for name in names:
            candidate = targets_dir / name
            if candidate.exists():
                resolved[key] = candidate
                break
    return resolved


def load_label_png(path: str | Path) -> np.ndarray:
    label_map = np.asarray(Image.open(path), dtype=np.uint8)
    if label_map.ndim == 3:
        label_map = label_map[..., 0]
    return label_map


def load_tile_shape_from_post_label(post_label_path: str | Path) -> tuple[int, int]:
    payload = read_json(post_label_path)
    metadata = payload.get("metadata", {})
    width = int(metadata.get("width", metadata.get("original_width", 1024)))
    height = int(metadata.get("height", metadata.get("original_height", 1024)))
    return height, width


def rasterize_damage_target_from_tile_instances(
    tile_instances: list[dict[str, Any]],
    *,
    tile_height: int,
    tile_width: int,
    overlap_policy: str = "max_label",
) -> np.ndarray:
    target_map = np.zeros((tile_height, tile_width), dtype=np.uint8)
    for instance in sorted(tile_instances, key=lambda item: int(item["building_idx"])):
        polygon_mask = polygon_to_mask(instance["polygon_xy"], tile_height, tile_width, offset=(0.0, 0.0)).astype(bool)
        pixel_label = convert_instance_label_to_pixel_label(int(instance["gt_label"]))
        if overlap_policy == "max_label":
            target_map[polygon_mask] = np.maximum(target_map[polygon_mask], pixel_label)
        elif overlap_policy == "last_wins":
            target_map[polygon_mask] = pixel_label
        else:
            raise ValueError(f"Unsupported overlap_policy='{overlap_policy}'.")
    return target_map


def load_bridge_targets(
    tile_sample: dict[str, Any],
    tile_instances: list[dict[str, Any]],
    *,
    target_source: str,
    use_official_target_png: bool,
    overlap_policy: str,
) -> dict[str, Any]:
    target_source = str(target_source)
    target_paths = resolve_target_png_paths(tile_sample)
    if target_source == "png" and not use_official_target_png:
        raise ValueError("target_source='png' requires use_official_target_png=true.")

    if target_source == "png":
        damage_path = target_paths["damage"]
        if damage_path is None:
            raise FileNotFoundError(f"Official damage target png not found for tile_id={tile_sample['tile_id']}.")
        damage_target = load_label_png(damage_path)
        localization_path = target_paths["localization"]
        if localization_path is not None:
            localization_target = (load_label_png(localization_path) > 0).astype(np.uint8)
        else:
            localization_target = (damage_target > 0).astype(np.uint8)
        return {
            "damage_target_map": damage_target.astype(np.uint8),
            "localization_target_map": localization_target.astype(np.uint8),
            "target_source_used": "png",
            "target_paths": {key: None if value is None else str(value) for key, value in target_paths.items()},
        }

    if target_source != "polygon_rasterize":
        raise ValueError(f"Unsupported target_source='{target_source}'.")

    tile_height, tile_width = load_tile_shape_from_post_label(tile_sample["post_label"])
    damage_target = rasterize_damage_target_from_tile_instances(
        tile_instances,
        tile_height=tile_height,
        tile_width=tile_width,
        overlap_policy=overlap_policy,
    )
    return {
        "damage_target_map": damage_target,
        "localization_target_map": (damage_target > 0).astype(np.uint8),
        "target_source_used": "polygon_rasterize",
        "target_paths": {key: None if value is None else str(value) for key, value in target_paths.items()},
    }


def build_pixel_damage_map_from_instances(
    tile_instances: list[dict[str, Any]],
    *,
    tile_height: int,
    tile_width: int,
    overlap_policy: str = "max_label",
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction_map = np.zeros((tile_height, tile_width), dtype=np.uint8)
    overlap_pixel_count = 0
    painted_pixels = np.zeros((tile_height, tile_width), dtype=np.uint16)
    for instance in sorted(tile_instances, key=lambda item: int(item["building_idx"])):
        polygon_mask = polygon_to_mask(instance["polygon_xy"], tile_height, tile_width, offset=(0.0, 0.0)).astype(bool)
        painted_pixels[polygon_mask] += 1
        overlap_pixel_count += int(np.logical_and(prediction_map > 0, polygon_mask).sum())
        pixel_label = convert_instance_label_to_pixel_label(int(instance["pred_label"]))
        if overlap_policy == "max_label":
            prediction_map[polygon_mask] = np.maximum(prediction_map[polygon_mask], pixel_label)
        elif overlap_policy == "last_wins":
            prediction_map[polygon_mask] = pixel_label
        else:
            raise ValueError(f"Unsupported overlap_policy='{overlap_policy}'.")
    return prediction_map, {
        "num_instances": len(tile_instances),
        "overlap_pixel_count": int(overlap_pixel_count),
        "num_overlapped_pixels": int((painted_pixels > 1).sum()),
    }


def _compute_binary_f1(pred_binary: np.ndarray, target_binary: np.ndarray) -> dict[str, Any]:
    pred_binary = np.asarray(pred_binary, dtype=np.uint8)
    target_binary = np.asarray(target_binary, dtype=np.uint8)
    true_positive = int(np.logical_and(pred_binary == 1, target_binary == 1).sum())
    false_positive = int(np.logical_and(pred_binary == 1, target_binary == 0).sum())
    false_negative = int(np.logical_and(pred_binary == 0, target_binary == 1).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _compute_multiclass_f1_on_gt_buildings(pred_damage_map: np.ndarray, gt_damage_map: np.ndarray) -> dict[str, Any]:
    gt_damage = np.asarray(gt_damage_map, dtype=np.uint8)
    pred_damage = np.asarray(pred_damage_map, dtype=np.uint8)
    gt_building_mask = gt_damage > 0
    pred_damage_masked = pred_damage.copy()
    pred_damage_masked[~(pred_damage > 0)] = 0

    if not np.any(gt_building_mask):
        class_metrics = {
            OFFICIAL_DAMAGE_CLASS_SHORT_NAMES[class_id]: {
                "class_id": class_id,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            for class_id in OFFICIAL_DAMAGE_CLASS_IDS
        }
        return {
            "class_metrics": class_metrics,
            "damage_f1_hmean": 0.0,
            "confusion_matrix_on_gt_building_pixels": np.zeros((4, 5), dtype=np.int64),
        }

    gt_building_damage = gt_damage[gt_building_mask]
    pred_building_damage = pred_damage_masked[gt_building_mask]
    class_metrics: dict[str, Any] = {}
    f1_values: list[float] = []
    for class_id in OFFICIAL_DAMAGE_CLASS_IDS:
        tp = int(np.logical_and(pred_building_damage == class_id, gt_building_damage == class_id).sum())
        fp = int(np.logical_and(pred_building_damage == class_id, gt_building_damage != class_id).sum())
        fn = int(np.logical_and(pred_building_damage != class_id, gt_building_damage == class_id).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
        class_metrics[OFFICIAL_DAMAGE_CLASS_SHORT_NAMES[class_id]] = {
            "class_id": class_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        f1_values.append(float(f1))
    damage_f1_hmean = 0.0 if any(value <= 0.0 for value in f1_values) else float(len(f1_values) / np.sum([1.0 / value for value in f1_values]))
    matrix = confusion_matrix(gt_building_damage, pred_building_damage, labels=[0, 1, 2, 3, 4])[1:, :]
    return {
        "class_metrics": class_metrics,
        "damage_f1_hmean": float(damage_f1_hmean),
        "confusion_matrix_on_gt_building_pixels": matrix,
    }


def compute_xview2_style_pixel_metrics(
    pred_damage_map: np.ndarray,
    gt_damage_map: np.ndarray,
    *,
    localization_target_map: np.ndarray | None = None,
) -> dict[str, Any]:
    pred_damage = np.asarray(pred_damage_map, dtype=np.uint8)
    gt_damage = np.asarray(gt_damage_map, dtype=np.uint8)
    if pred_damage.shape != gt_damage.shape:
        raise ValueError(f"Prediction and GT damage map shapes must match, got {pred_damage.shape} vs {gt_damage.shape}.")
    gt_localization = (gt_damage > 0).astype(np.uint8) if localization_target_map is None else (
        np.asarray(localization_target_map, dtype=np.uint8) > 0
    ).astype(np.uint8)
    pred_localization = (pred_damage > 0).astype(np.uint8)
    localization = _compute_binary_f1(pred_localization, gt_localization)
    damage = _compute_multiclass_f1_on_gt_buildings(pred_damage, gt_damage)
    overall = (0.3 * float(localization["f1"])) + (0.7 * float(damage["damage_f1_hmean"]))
    return {
        "localization_f1": float(localization["f1"]),
        "localization_precision": float(localization["precision"]),
        "localization_recall": float(localization["recall"]),
        "localization_counts": {"tp": localization["tp"], "fp": localization["fp"], "fn": localization["fn"]},
        "damage_f1_no_damage": float(damage["class_metrics"]["no_damage"]["f1"]),
        "damage_f1_minor": float(damage["class_metrics"]["minor"]["f1"]),
        "damage_f1_major": float(damage["class_metrics"]["major"]["f1"]),
        "damage_f1_destroyed": float(damage["class_metrics"]["destroyed"]["f1"]),
        "damage_class_metrics": damage["class_metrics"],
        "damage_f1_hmean": float(damage["damage_f1_hmean"]),
        "official_style_confusion_matrix_on_gt_building_pixels": damage["confusion_matrix_on_gt_building_pixels"].tolist(),
        "overall_xview2_style_score": float(overall),
    }


def aggregate_xview2_style_metrics(per_tile_scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_tile_scores:
        return {
            "localization_f1": 0.0,
            "damage_f1_no_damage": 0.0,
            "damage_f1_minor": 0.0,
            "damage_f1_major": 0.0,
            "damage_f1_destroyed": 0.0,
            "damage_f1_hmean": 0.0,
            "overall_xview2_style_score": 0.0,
            "official_style_confusion_matrix_on_gt_building_pixels": [[0, 0, 0, 0, 0] for _ in range(4)],
        }

    localization_tp = sum(int(item["localization_counts"]["tp"]) for item in per_tile_scores)
    localization_fp = sum(int(item["localization_counts"]["fp"]) for item in per_tile_scores)
    localization_fn = sum(int(item["localization_counts"]["fn"]) for item in per_tile_scores)
    localization_precision = localization_tp / max(localization_tp + localization_fp, 1)
    localization_recall = localization_tp / max(localization_tp + localization_fn, 1)
    localization_f1 = 0.0 if (localization_precision + localization_recall) <= 0.0 else (
        2.0 * localization_precision * localization_recall
    ) / (localization_precision + localization_recall)

    per_class_counts: dict[str, dict[str, int]] = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in ["no_damage", "minor", "major", "destroyed"]
    }
    confusion_matrix_total = np.zeros((4, 5), dtype=np.int64)
    for item in per_tile_scores:
        confusion_matrix_total += np.asarray(item["official_style_confusion_matrix_on_gt_building_pixels"], dtype=np.int64)
        for class_name in per_class_counts:
            class_metrics = item["damage_class_metrics"][class_name]
            per_class_counts[class_name]["tp"] += int(class_metrics["tp"])
            per_class_counts[class_name]["fp"] += int(class_metrics["fp"])
            per_class_counts[class_name]["fn"] += int(class_metrics["fn"])

    class_f1: dict[str, float] = {}
    f1_values: list[float] = []
    for class_name, counts in per_class_counts.items():
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
        class_f1[class_name] = float(f1)
        f1_values.append(float(f1))
    damage_f1_hmean = 0.0 if any(value <= 0.0 for value in f1_values) else float(len(f1_values) / np.sum([1.0 / value for value in f1_values]))
    overall = (0.3 * float(localization_f1)) + (0.7 * float(damage_f1_hmean))
    return {
        "localization_f1": float(localization_f1),
        "damage_f1_no_damage": float(class_f1["no_damage"]),
        "damage_f1_minor": float(class_f1["minor"]),
        "damage_f1_major": float(class_f1["major"]),
        "damage_f1_destroyed": float(class_f1["destroyed"]),
        "damage_f1_hmean": float(damage_f1_hmean),
        "overall_xview2_style_score": float(overall),
        "official_style_confusion_matrix_on_gt_building_pixels": confusion_matrix_total.tolist(),
        "localization_counts": {"tp": localization_tp, "fp": localization_fp, "fn": localization_fn},
        "damage_class_counts": per_class_counts,
    }


def save_damage_confusion_matrix_plot(matrix: np.ndarray | list[list[int]], save_path: str | Path) -> None:
    save_matrix_heatmap_plot(
        matrix,
        row_labels=["gt:no", "gt:minor", "gt:major", "gt:destroyed"],
        col_labels=["pred:bg", "pred:no", "pred:minor", "pred:major", "pred:destroyed"],
        save_path=save_path,
        title="Official-Style Damage Confusion On GT Building Pixels",
        cmap="magma",
        value_format="d",
        xlabel="Predicted Damage Label",
        ylabel="Ground-Truth Damage Label",
    )


def build_tile_instance_groups(dataset_samples: list[dict[str, Any]], prediction_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in prediction_records:
        sample_index = int(record["sample_index"])
        sample = dataset_samples[sample_index]
        tile_id = str(sample["tile_id"])
        tile_group = grouped.setdefault(
            tile_id,
            {
                "tile_id": tile_id,
                "source_subset": sample["source_subset"],
                "post_image": sample["post_image"],
                "pre_image": sample["pre_image"],
                "post_label": sample["post_label"],
                "instances": [],
            },
        )
        tile_group["instances"].append(
            {
                "building_idx": int(sample["building_idx"]),
                "uid": sample.get("uid"),
                "polygon_xy": sample["polygon_xy"],
                "gt_label": int(sample["label"]),
                "pred_label": int(record["prediction"]),
                "prediction_name": str(record["prediction_name"]),
                "target_name": str(record["target_name"]),
                "probabilities": [float(value) for value in record["probabilities"]],
                "expected_severity": float(record.get("expected_severity", 0.0)),
                "meta": record["meta"],
            }
        )
    return grouped


def build_tile_bridge_result(
    tile_group: dict[str, Any],
    *,
    target_source: str,
    use_official_target_png: bool,
    overlap_policy: str,
) -> dict[str, Any]:
    tile_sample = {
        "tile_id": tile_group["tile_id"],
        "post_image": tile_group["post_image"],
        "post_label": tile_group["post_label"],
    }
    target_payload = load_bridge_targets(
        tile_sample,
        tile_group["instances"],
        target_source=target_source,
        use_official_target_png=use_official_target_png,
        overlap_policy=overlap_policy,
    )
    gt_damage_map = np.asarray(target_payload["damage_target_map"], dtype=np.uint8)
    tile_height, tile_width = gt_damage_map.shape
    pred_damage_map, bridge_stats = build_pixel_damage_map_from_instances(
        tile_group["instances"],
        tile_height=tile_height,
        tile_width=tile_width,
        overlap_policy=overlap_policy,
    )
    tile_scores = compute_xview2_style_pixel_metrics(
        pred_damage_map,
        gt_damage_map,
        localization_target_map=target_payload["localization_target_map"],
    )
    tile_scores.update(
        {
            "tile_id": tile_group["tile_id"],
            "source_subset": tile_group["source_subset"],
            "target_source_used": target_payload["target_source_used"],
            "target_paths": target_payload["target_paths"],
            "tile_shape": [int(tile_height), int(tile_width)],
            "bridge_stats": bridge_stats,
        }
    )
    return {
        "prediction_map": pred_damage_map,
        "gt_damage_map": gt_damage_map,
        "gt_localization_map": np.asarray(target_payload["localization_target_map"], dtype=np.uint8),
        "tile_scores": tile_scores,
    }
