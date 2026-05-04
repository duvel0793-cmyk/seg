from __future__ import annotations

from typing import Any

import numpy as np

from metrics.localization_metrics import build_iou_matrix, greedy_match_by_iou
from utils.misc import CLASS_NAMES


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)


def _classification_summary(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if not y_true:
        per_class = {
            class_name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            for class_name in CLASS_NAMES
        }
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": per_class}
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    per_class: dict[str, Any] = {}
    f1_values = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        tp = float(np.logical_and(y_true_np == class_index, y_pred_np == class_index).sum())
        fp = float(np.logical_and(y_true_np != class_index, y_pred_np == class_index).sum())
        fn = float(np.logical_and(y_true_np == class_index, y_pred_np != class_index).sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1_value = _f1(precision, recall)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_value,
            "support": int((y_true_np == class_index).sum()),
        }
        f1_values.append(f1_value)
    accuracy = float((y_true_np == y_pred_np).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def compute_matched_damage_metrics(prediction_groups: list[dict[str, Any]], *, iou_threshold: float = 0.5) -> dict[str, Any]:
    matched_true: list[int] = []
    matched_pred: list[int] = []
    for group in prediction_groups:
        pred_masks = [instance["mask"] for instance in group["pred_instances"]]
        gt_masks = [instance["mask"] for instance in group["gt_instances"]]
        iou_matrix = build_iou_matrix(pred_masks, gt_masks)
        matches = greedy_match_by_iou(iou_matrix, threshold=iou_threshold)
        for pred_index, gt_index, _ in matches:
            matched_true.append(int(group["gt_instances"][gt_index]["label"]))
            matched_pred.append(int(group["pred_instances"][pred_index]["label"]))
    summary = _classification_summary(matched_true, matched_pred)
    summary["num_matched"] = len(matched_true)
    return summary


def compute_end_to_end_damage_metrics(prediction_groups: list[dict[str, Any]], *, iou_threshold: float = 0.5) -> dict[str, Any]:
    tp = np.zeros((len(CLASS_NAMES),), dtype=np.float64)
    fp = np.zeros((len(CLASS_NAMES),), dtype=np.float64)
    fn = np.zeros((len(CLASS_NAMES),), dtype=np.float64)

    for group in prediction_groups:
        pred_masks = [instance["mask"] for instance in group["pred_instances"]]
        gt_masks = [instance["mask"] for instance in group["gt_instances"]]
        iou_matrix = build_iou_matrix(pred_masks, gt_masks)
        matches = greedy_match_by_iou(iou_matrix, threshold=iou_threshold)
        matched_pred = {pred_index for pred_index, _, _ in matches}
        matched_gt = {gt_index for _, gt_index, _ in matches}

        for pred_index, gt_index, _ in matches:
            pred_label = int(group["pred_instances"][pred_index]["label"])
            gt_label = int(group["gt_instances"][gt_index]["label"])
            if pred_label == gt_label:
                tp[gt_label] += 1.0
            else:
                fp[pred_label] += 1.0
                fn[gt_label] += 1.0

        for pred_index, pred_instance in enumerate(group["pred_instances"]):
            if pred_index not in matched_pred:
                fp[int(pred_instance["label"])] += 1.0
        for gt_index, gt_instance in enumerate(group["gt_instances"]):
            if gt_index not in matched_gt:
                fn[int(gt_instance["label"])] += 1.0

    per_class: dict[str, Any] = {}
    f1_values = []
    supports = []
    weighted_f1_numerator = 0.0
    for class_index, class_name in enumerate(CLASS_NAMES):
        precision = _safe_divide(tp[class_index], tp[class_index] + fp[class_index])
        recall = _safe_divide(tp[class_index], tp[class_index] + fn[class_index])
        f1_value = _f1(precision, recall)
        support = int(tp[class_index] + fn[class_index])
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_value,
            "support": support,
            "tp": float(tp[class_index]),
            "fp": float(fp[class_index]),
            "fn": float(fn[class_index]),
        }
        f1_values.append(f1_value)
        supports.append(support)
        weighted_f1_numerator += f1_value * support

    total_support = int(sum(supports))
    return {
        "per_class": per_class,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_f1": weighted_f1_numerator / max(float(total_support), 1e-8),
        "total_support": total_support,
    }
