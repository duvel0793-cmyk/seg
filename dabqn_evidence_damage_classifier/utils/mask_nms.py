from __future__ import annotations

import math
from typing import Any

import numpy as np


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _box_center(box: list[float]) -> tuple[float, float]:
    return float((box[0] + box[2]) * 0.5), float((box[1] + box[3]) * 0.5)


def _box_diagonal(box: list[float]) -> float:
    width = max(float(box[2] - box[0]), 1e-6)
    height = max(float(box[3] - box[1]), 1e-6)
    return float(math.hypot(width, height))


def _point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def compute_patch_center_weight(
    *,
    local_box_xyxy: list[float],
    patch_width: int,
    patch_height: int,
    use_patch_center_score: bool,
    patch_border_penalty: float,
    border_margin: int,
) -> tuple[float, bool, float]:
    if not use_patch_center_score:
        return 1.0, False, 0.0
    x1, y1, x2, y2 = [float(value) for value in local_box_xyxy]
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    patch_center_x = float(patch_width) * 0.5
    patch_center_y = float(patch_height) * 0.5
    center_distance = math.hypot(center_x - patch_center_x, center_y - patch_center_y)
    max_distance = max(math.hypot(float(patch_width) * 0.5, float(patch_height) * 0.5), 1.0)
    distance_ratio = min(center_distance / max_distance, 1.0)
    weight = max(0.7, 1.0 - (0.15 * distance_ratio))
    touches_border = (
        x1 <= float(border_margin)
        or y1 <= float(border_margin)
        or x2 >= float(patch_width - border_margin)
        or y2 >= float(patch_height - border_margin)
    )
    if touches_border:
        weight *= float(patch_border_penalty)
    return float(weight), bool(touches_border), float(distance_ratio)


def _overlap_window(pred_a: dict[str, Any], pred_b: dict[str, Any]) -> tuple[int, int, int, int] | None:
    ax1, ay1, ax2, ay2 = [int(value) for value in pred_a["patch_box"]]
    bx1, by1, bx2, by2 = [int(value) for value in pred_b["patch_box"]]
    overlap_x1 = max(ax1, bx1)
    overlap_y1 = max(ay1, by1)
    overlap_x2 = min(ax2, bx2)
    overlap_y2 = min(ay2, by2)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return None
    return overlap_x1, overlap_y1, overlap_x2, overlap_y2


def _mask_intersection(pred_a: dict[str, Any], pred_b: dict[str, Any]) -> int:
    overlap = _overlap_window(pred_a, pred_b)
    if overlap is None:
        return 0
    overlap_x1, overlap_y1, overlap_x2, overlap_y2 = overlap
    ax1, ay1, _, _ = [int(value) for value in pred_a["patch_box"]]
    bx1, by1, _, _ = [int(value) for value in pred_b["patch_box"]]
    mask_a = pred_a["mask"][overlap_y1 - ay1 : overlap_y2 - ay1, overlap_x1 - ax1 : overlap_x2 - ax1]
    mask_b = pred_b["mask"][overlap_y1 - by1 : overlap_y2 - by1, overlap_x1 - bx1 : overlap_x2 - bx1]
    return int(np.logical_and(mask_a, mask_b).sum())


def duplicate_metrics(pred_a: dict[str, Any], pred_b: dict[str, Any]) -> dict[str, float | bool]:
    intersection = float(_mask_intersection(pred_a, pred_b))
    area_a = float(pred_a["mask_area"])
    area_b = float(pred_b["mask_area"])
    union = area_a + area_b - intersection
    mask_iou = 0.0 if union <= 0.0 else intersection / union
    containment = 0.0 if min(area_a, area_b) <= 0.0 else intersection / min(area_a, area_b)
    box_iou_value = box_iou(pred_a["box_xyxy"], pred_b["box_xyxy"])
    center_a = _box_center(pred_a["box_xyxy"])
    center_b = _box_center(pred_b["box_xyxy"])
    center_distance = float(math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    diagonal_scale = max(min(_box_diagonal(pred_a["box_xyxy"]), _box_diagonal(pred_b["box_xyxy"])), 1.0)
    center_distance_ratio = center_distance / diagonal_scale
    center_inside_other = _point_in_box(center_a, pred_b["box_xyxy"]) or _point_in_box(center_b, pred_a["box_xyxy"])
    return {
        "mask_iou": float(mask_iou),
        "box_iou": float(box_iou_value),
        "containment": float(containment),
        "center_distance": float(center_distance),
        "center_distance_ratio": float(center_distance_ratio),
        "center_inside_other": bool(center_inside_other),
    }


def _duplicate_reason(
    metrics: dict[str, float | bool],
    *,
    mask_nms_iou: float,
    box_nms_iou: float,
    containment_nms_thr: float,
    center_distance_ratio: float,
) -> str | None:
    if float(metrics["mask_iou"]) >= float(mask_nms_iou):
        return "mask_iou"
    if float(metrics["containment"]) >= float(containment_nms_thr):
        return "containment"
    if float(metrics["box_iou"]) >= float(box_nms_iou) and float(metrics["center_distance_ratio"]) <= float(center_distance_ratio):
        return "box_iou"
    if bool(metrics["center_inside_other"]) and float(metrics["center_distance_ratio"]) <= float(center_distance_ratio) and float(metrics["box_iou"]) >= max(float(box_nms_iou) * 0.5, 0.1):
        return "center_distance"
    return None


def _prediction_rank(prediction: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(prediction.get("adjusted_score", prediction.get("score", 0.0))),
        float(prediction.get("center_prior_weight", 1.0)),
        0.0 if bool(prediction.get("touches_patch_border", False)) else 1.0,
        float(prediction.get("score", 0.0)),
        float(prediction.get("mask_area", 0.0)),
    )


def suppress_duplicate_predictions(
    predictions: list[dict[str, Any]],
    *,
    mask_nms_iou: float,
    box_nms_iou: float,
    containment_nms_thr: float,
    center_distance_ratio: float,
    max_predictions_per_image: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not predictions:
        return [], {
            "predictions_before_nms": 0,
            "predictions_after_nms": 0,
            "suppressed_pairs_count": 0,
            "suppressed_by_mask_iou": 0,
            "suppressed_by_box_iou": 0,
            "suppressed_by_containment": 0,
            "suppressed_by_center_distance": 0,
        }, []

    ordered = sorted(predictions, key=_prediction_rank, reverse=True)[: max(int(max_predictions_per_image) * 4, int(max_predictions_per_image))]
    kept: list[dict[str, Any]] = []
    suppressed_pairs: list[dict[str, Any]] = []
    stats = {
        "predictions_before_nms": int(len(predictions)),
        "predictions_after_nms": 0,
        "suppressed_pairs_count": 0,
        "suppressed_by_mask_iou": 0,
        "suppressed_by_box_iou": 0,
        "suppressed_by_containment": 0,
        "suppressed_by_center_distance": 0,
    }

    for candidate in ordered:
        duplicate_of: dict[str, Any] | None = None
        duplicate_reason: str | None = None
        duplicate_detail: dict[str, Any] | None = None
        for existing in kept:
            metrics = duplicate_metrics(candidate, existing)
            reason = _duplicate_reason(
                metrics,
                mask_nms_iou=mask_nms_iou,
                box_nms_iou=box_nms_iou,
                containment_nms_thr=containment_nms_thr,
                center_distance_ratio=center_distance_ratio,
            )
            if reason is not None:
                duplicate_of = existing
                duplicate_reason = reason
                duplicate_detail = metrics
                break

        if duplicate_of is not None and duplicate_reason is not None and duplicate_detail is not None:
            stats["suppressed_pairs_count"] += 1
            stats[f"suppressed_by_{duplicate_reason}"] += 1
            suppressed_pairs.append(
                {
                    "suppressed_prediction": candidate,
                    "kept_prediction": duplicate_of,
                    "reason": duplicate_reason,
                    "metrics": duplicate_detail,
                }
            )
            continue

        kept.append(candidate)
        if len(kept) >= int(max_predictions_per_image):
            break

    stats["predictions_after_nms"] = int(len(kept))
    return kept, stats, suppressed_pairs
