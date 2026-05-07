from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from utils.misc import CLASS_NAMES


def per_class_f1_aliases(metrics: dict[str, Any]) -> dict[str, float]:
    mapping = {
        "no-damage": "no_damage",
        "minor-damage": "minor_damage",
        "major-damage": "major_damage",
        "destroyed": "destroyed",
    }
    per_class = metrics.get("per_class", {})
    return {alias: float((per_class.get(class_name) or {}).get("f1", 0.0)) for class_name, alias in mapping.items()}


def compute_classification_metrics(y_true: list[int], y_pred: list[int], probabilities: np.ndarray | None = None) -> dict[str, Any]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, class_name in enumerate(CLASS_NAMES)
    }
    distances = np.abs(y_true_np - y_pred_np)
    binary_true = (y_true_np > 0).astype(np.int64)
    binary_pred = (y_pred_np > 0).astype(np.int64)
    binary_f1 = f1_score(binary_true, binary_pred, zero_division=0)
    binary_acc = accuracy_score(binary_true, binary_pred)
    damaged_mask = y_true_np > 0
    damaged_severity_macro_f1 = (
        f1_score(y_true_np[damaged_mask] - 1, y_pred_np[damaged_mask] - 1, average="macro", zero_division=0)
        if np.any(damaged_mask)
        else 0.0
    )
    qwk = cohen_kappa_score(y_true_np, y_pred_np, weights="quadratic")
    if not np.isfinite(qwk):
        qwk = 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "QWK": float(qwk),
        "adjacent_error": float(np.mean(distances == 1)) if distances.size else 0.0,
        "far_error": float(np.mean(distances >= 2)) if distances.size else 0.0,
        "per_class": per_class,
        "binary_damage_accuracy": float(binary_acc),
        "binary_damage_f1": float(binary_f1),
        "damaged_severity_macro_f1": float(damaged_severity_macro_f1),
        "no_f1": per_class["no-damage"]["f1"],
        "minor_precision": per_class["minor-damage"]["precision"],
        "minor_recall": per_class["minor-damage"]["recall"],
        "minor_f1": per_class["minor-damage"]["f1"],
        "major_precision": per_class["major-damage"]["precision"],
        "major_recall": per_class["major-damage"]["recall"],
        "major_f1": per_class["major-damage"]["f1"],
        "destroyed_f1": per_class["destroyed"]["f1"],
        "gt_minor_pred_no_rate": float(np.mean(y_pred_np[y_true_np == 1] == 0)) if np.any(y_true_np == 1) else 0.0,
        "gt_minor_pred_major_rate": float(np.mean(y_pred_np[y_true_np == 1] == 2)) if np.any(y_true_np == 1) else 0.0,
        "gt_major_pred_minor_rate": float(np.mean(y_pred_np[y_true_np == 2] == 1)) if np.any(y_true_np == 2) else 0.0,
        "gt_major_pred_no_rate": float(np.mean(y_pred_np[y_true_np == 2] == 0)) if np.any(y_true_np == 2) else 0.0,
        "gt_destroyed_pred_major_rate": float(np.mean(y_pred_np[y_true_np == 3] == 2)) if np.any(y_true_np == 3) else 0.0,
        "prediction_distribution": {CLASS_NAMES[idx]: int(np.sum(y_pred_np == idx)) for idx in range(len(CLASS_NAMES))},
        "gt_distribution": {CLASS_NAMES[idx]: int(np.sum(y_true_np == idx)) for idx in range(len(CLASS_NAMES))},
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np, labels=np.arange(len(CLASS_NAMES))).tolist(),
    }
    if probabilities is not None and probabilities.size > 0:
        metrics["mean_probability_by_class"] = {
            CLASS_NAMES[idx]: float(np.mean(probabilities[:, idx])) for idx in range(probabilities.shape[1])
        }
    return metrics


def build_classification_report(y_true: list[int], y_pred: list[int]) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )


def decode_predictions_with_damage_threshold(
    damage_probabilities: np.ndarray,
    severity_probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if damage_probabilities.size == 0:
        return np.zeros((0,), dtype=np.int64)
    severity_pred = np.argmax(severity_probabilities, axis=1) + 1
    pred = np.where(damage_probabilities >= float(threshold), severity_pred, 0)
    return pred.astype(np.int64)


def run_damage_threshold_sweep(
    y_true: list[int],
    damage_probabilities: np.ndarray,
    severity_probabilities: np.ndarray,
    class_probabilities: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    sweep_results = []
    best_macro = None
    best_qwk = None
    y_true_list = [int(v) for v in y_true]
    for threshold in thresholds:
        pred = decode_predictions_with_damage_threshold(damage_probabilities, severity_probabilities, threshold).tolist()
        metrics = compute_classification_metrics(y_true_list, pred, class_probabilities)
        record = {"threshold": float(threshold), **metrics}
        sweep_results.append(record)
        if best_macro is None or record["macro_f1"] > best_macro["macro_f1"]:
            best_macro = record
        if best_qwk is None or record["QWK"] > best_qwk["QWK"]:
            best_qwk = record
    return {
        "results": sweep_results,
        "best_threshold_by_macro_f1": None if best_macro is None else {"threshold": best_macro["threshold"], "macro_f1": best_macro["macro_f1"]},
        "best_threshold_by_qwk": None if best_qwk is None else {"threshold": best_qwk["threshold"], "QWK": best_qwk["QWK"]},
    }


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _aggregate_evidence_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "scales": {}}
    output: dict[str, Any] = {"count": len(records), "scales": {}}
    for scale_name in ("tight", "context", "neighborhood"):
        sev_mean = []
        sev_topk = []
        high_damage_ratio = []
        for record in records:
            scale_stats = (record.get("evidence_stats") or {}).get(scale_name)
            if not scale_stats or len(scale_stats) < 17:
                continue
            sev_mean.append(float(scale_stats[12]))
            sev_topk.append(float(scale_stats[14]))
            high_damage_ratio.append(float(scale_stats[16]))
        if sev_mean or sev_topk or high_damage_ratio:
            output["scales"][scale_name] = {
                "severity_mean": _mean_or_none(sev_mean),
                "severity_topk": _mean_or_none(sev_topk),
                "high_damage_ratio": _mean_or_none(high_damage_ratio),
            }
    return output


def build_diagnostics(
    prediction_records: list[dict[str, Any]],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y_true = [int(record["gt_label"]) for record in prediction_records]
    y_pred = [int(record["pred_label"]) for record in prediction_records]
    diagnostics = {
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "qwk": metrics.get("QWK", 0.0),
        "far_error": metrics.get("far_error", 0.0),
        "pred_label_distribution": metrics.get("prediction_distribution", {}),
        "gt_label_distribution": metrics.get("gt_distribution", {}),
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "per_class": metrics.get("per_class", {}),
        "binary_damage_f1": metrics.get("binary_damage_f1", 0.0),
        "damaged_severity_macro_f1": metrics.get("damaged_severity_macro_f1", 0.0),
        "gt_minor_pred_no_rate": metrics.get("gt_minor_pred_no_rate", 0.0),
        "gt_minor_pred_major_rate": metrics.get("gt_minor_pred_major_rate", 0.0),
        "gt_major_pred_minor_rate": metrics.get("gt_major_pred_minor_rate", 0.0),
        "gt_major_pred_no_rate": metrics.get("gt_major_pred_no_rate", 0.0),
        "gt_destroyed_pred_major_rate": metrics.get("gt_destroyed_pred_major_rate", 0.0),
        "mean_binary_damaged_probability_by_gt_class": {},
        "mean_severity_probabilities_by_gt_class": {},
        "mean_class_probabilities_by_gt_class": {},
        "error_group_evidence_stats": {},
    }
    evidence_available = False
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_records = [record for record in prediction_records if int(record["gt_label"]) == class_id]
        binary_probs = [float(record["binary_damaged_probability"]) for record in class_records if record.get("binary_damaged_probability") is not None]
        severity_probs = [record.get("severity_probabilities") for record in class_records if record.get("severity_probabilities")]
        class_probs = [record.get("class_probabilities") for record in class_records if record.get("class_probabilities")]
        diagnostics["mean_binary_damaged_probability_by_gt_class"][class_name] = _mean_or_none(binary_probs)
        if severity_probs:
            arr = np.asarray(severity_probs, dtype=np.float64)
            diagnostics["mean_severity_probabilities_by_gt_class"][class_name] = arr.mean(axis=0).tolist()
        else:
            diagnostics["mean_severity_probabilities_by_gt_class"][class_name] = None
        if class_probs:
            diagnostics["mean_class_probabilities_by_gt_class"][class_name] = np.asarray(class_probs, dtype=np.float64).mean(axis=0).tolist()
        else:
            diagnostics["mean_class_probabilities_by_gt_class"][class_name] = None
        if any(
            isinstance(record.get("evidence_stats"), dict)
            and any(value is not None for value in record["evidence_stats"].values())
            for record in class_records
        ):
            evidence_available = True

    grouped = {
        "gt_minor_pred_no": [record for record in prediction_records if int(record["gt_label"]) == 1 and int(record["pred_label"]) == 0],
        "gt_minor_pred_major": [record for record in prediction_records if int(record["gt_label"]) == 1 and int(record["pred_label"]) == 2],
        "gt_major_pred_minor": [record for record in prediction_records if int(record["gt_label"]) == 2 and int(record["pred_label"]) == 1],
        "gt_destroyed_pred_major": [record for record in prediction_records if int(record["gt_label"]) == 3 and int(record["pred_label"]) == 2],
    }
    for name, records in grouped.items():
        diagnostics["error_group_evidence_stats"][name] = _aggregate_evidence_group(records) if evidence_available else None

    if not evidence_available:
        diagnostics["error_group_evidence_stats"] = None

    diagnostics["num_records"] = len(prediction_records)
    if extra:
        diagnostics.update(extra)
    return diagnostics
