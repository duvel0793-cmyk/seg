from __future__ import annotations

from typing import Any

import numpy as np


PIXEL_CLASS_NAMES = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}


def rasterize_instances_to_damage_map(
    instances: list[dict[str, Any]],
    height: int,
    width: int,
    *,
    label_offset: int = 1,
) -> np.ndarray:
    label_map = np.zeros((int(height), int(width)), dtype=np.uint8)
    score_map = np.full((int(height), int(width)), fill_value=-1.0, dtype=np.float32)
    for instance in instances:
        mask = np.asarray(instance["mask"], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(f"Mask shape mismatch: expected {(height, width)}, got {mask.shape}.")
        score = float(instance.get("score", 1.0))
        replace = mask & (score >= score_map)
        label_map[replace] = int(instance["label"]) + int(label_offset)
        score_map[replace] = score
    return label_map


def _confusion_matrix(pred_map: np.ndarray, gt_map: np.ndarray, num_classes: int) -> np.ndarray:
    if pred_map.shape != gt_map.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred_map.shape} vs {gt_map.shape}.")
    flat = (gt_map.astype(np.int64).reshape(-1) * num_classes) + pred_map.astype(np.int64).reshape(-1)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict[str, Any]:
    num_classes = 5
    tp_loc = float(cm[1:, 1:].sum())
    fp_loc = float(cm[0, 1:].sum())
    fn_loc = float(cm[1:, 0].sum())
    loc_precision = _safe_divide(tp_loc, tp_loc + fp_loc)
    loc_recall = _safe_divide(tp_loc, tp_loc + fn_loc)
    localization_f1 = _f1(loc_precision, loc_recall)

    damage_per_class: dict[str, Any] = {}
    damage_f1_values: list[float] = []
    for class_id in range(1, num_classes):
        tp = float(cm[class_id, class_id])
        fp = float(cm[:, class_id].sum() - cm[class_id, class_id])
        fn = float(cm[class_id, :].sum() - cm[class_id, class_id])
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1_value = _f1(precision, recall)
        damage_per_class[PIXEL_CLASS_NAMES[class_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_value,
            "support": int(cm[class_id, :].sum()),
        }
        damage_f1_values.append(max(f1_value, 1e-8))
    hmean = 0.0 if not damage_f1_values else float(len(damage_f1_values) / sum(1.0 / value for value in damage_f1_values))
    macro_f1 = float(np.mean([entry["f1"] for entry in damage_per_class.values()])) if damage_per_class else 0.0
    return {
        "confusion_matrix_5x5": cm.tolist(),
        "localization_f1": localization_f1,
        "damage_f1_hmean": hmean,
        "damage_macro_f1": macro_f1,
        "xview2_overall_score": (0.3 * localization_f1) + (0.7 * hmean),
        "damage_per_class": damage_per_class,
    }


def compute_pixel_bridge_metrics(pred_map: np.ndarray, gt_map: np.ndarray) -> dict[str, Any]:
    cm = _confusion_matrix(pred_map, gt_map, num_classes=5)
    return metrics_from_confusion_matrix(cm)
