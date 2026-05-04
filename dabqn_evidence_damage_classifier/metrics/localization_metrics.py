from __future__ import annotations

from typing import Any

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def build_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_index, pred_mask in enumerate(pred_masks):
        for gt_index, gt_mask in enumerate(gt_masks):
            matrix[pred_index, gt_index] = mask_iou(pred_mask, gt_mask)
    return matrix


def greedy_match_by_iou(iou_matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    if iou_matrix.size == 0:
        return []
    candidates: list[tuple[float, int, int]] = []
    for pred_index in range(iou_matrix.shape[0]):
        for gt_index in range(iou_matrix.shape[1]):
            iou = float(iou_matrix[pred_index, gt_index])
            if iou >= float(threshold):
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, iou))
    return matches


def compute_localization_metrics(
    prediction_groups: list[dict[str, Any]],
    *,
    iou_thresholds: tuple[float, float] = (0.5, 0.75),
) -> dict[str, Any]:
    threshold_records: dict[float, dict[str, float]] = {
        threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0}
        for threshold in iou_thresholds
    }
    matched_ious: list[float] = []

    for group in prediction_groups:
        pred_masks = [instance["mask"] for instance in group["pred_instances"]]
        gt_masks = [instance["mask"] for instance in group["gt_instances"]]
        iou_matrix = build_iou_matrix(pred_masks, gt_masks)
        match_05 = greedy_match_by_iou(iou_matrix, threshold=0.5)
        matched_ious.extend([match[2] for match in match_05])
        for threshold in iou_thresholds:
            matches = greedy_match_by_iou(iou_matrix, threshold=threshold)
            tp = float(len(matches))
            fp = float(max(len(pred_masks) - len(matches), 0))
            fn = float(max(len(gt_masks) - len(matches), 0))
            threshold_records[threshold]["tp"] += tp
            threshold_records[threshold]["fp"] += fp
            threshold_records[threshold]["fn"] += fn

    metrics: dict[str, Any] = {}
    for threshold in iou_thresholds:
        tp = threshold_records[threshold]["tp"]
        fp = threshold_records[threshold]["fp"]
        fn = threshold_records[threshold]["fn"]
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
        suffix = int(round(threshold * 100))
        metrics[f"precision_iou{suffix}"] = precision
        metrics[f"recall_iou{suffix}"] = recall
        metrics[f"f1_iou{suffix}"] = f1
    metrics["matched_mean_iou"] = float(np.mean(matched_ious)) if matched_ious else 0.0
    metrics["localization_f1"] = metrics["f1_iou50"]
    metrics["localization_precision"] = metrics["precision_iou50"]
    metrics["localization_recall"] = metrics["recall_iou50"]
    return metrics
