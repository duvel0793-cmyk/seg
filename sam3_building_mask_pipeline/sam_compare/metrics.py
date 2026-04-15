"""Metric utilities for binary segmentation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


SIZE_BUCKET_THRESHOLDS = (
    ("small", 0.0, 0.01),
    ("medium", 0.01, 0.05),
    ("large", 0.05, None),
)


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def to_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )


def _safe_divide(numerator: float, denominator: float, *, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return empty_value
    return numerator / denominator


def confusion_counts_from_masks(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> ConfusionCounts:
    """Compute confusion counts from binary masks."""
    pred = pred_mask.to(dtype=torch.bool).flatten()
    gt = gt_mask.to(dtype=torch.bool).flatten()

    tp = int(torch.logical_and(pred, gt).sum().item())
    fp = int(torch.logical_and(pred, torch.logical_not(gt)).sum().item())
    fn = int(torch.logical_and(torch.logical_not(pred), gt).sum().item())
    tn = int(torch.logical_and(torch.logical_not(pred), torch.logical_not(gt)).sum().item())
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def metrics_from_counts(counts: ConfusionCounts) -> dict[str, float | int]:
    """Convert confusion counts to binary segmentation metrics."""
    pred_positive = counts.tp + counts.fp
    gt_positive = counts.tp + counts.fn
    union = counts.tp + counts.fp + counts.fn

    empty_prediction = pred_positive == 0
    empty_ground_truth = gt_positive == 0
    empty_both = empty_prediction and empty_ground_truth

    precision = (
        1.0 if empty_both else _safe_divide(counts.tp, pred_positive, empty_value=0.0)
    )
    recall = (
        1.0 if empty_both else _safe_divide(counts.tp, gt_positive, empty_value=0.0)
    )
    iou = 1.0 if empty_both else _safe_divide(counts.tp, union, empty_value=0.0)
    f1 = (
        1.0
        if empty_both
        else _safe_divide(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn, empty_value=0.0)
    )
    pixel_accuracy = _safe_divide(counts.tp + counts.tn, counts.total, empty_value=1.0)

    return {
        **counts.to_dict(),
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_accuracy),
        "total_pixels": counts.total,
    }


def aggregate_confusion_counts(records: Iterable[dict[str, float | int]]) -> ConfusionCounts:
    total = ConfusionCounts()
    for record in records:
        total += ConfusionCounts(
            tp=int(record["tp"]),
            fp=int(record["fp"]),
            fn=int(record["fn"]),
            tn=int(record["tn"]),
        )
    return total


def _mean_metric(records: list[dict[str, float | int]], name: str) -> float:
    if not records:
        return 0.0
    return float(sum(float(record[name]) for record in records) / len(records))


def _metric_quantiles(records: list[dict[str, float | int]], metric_name: str) -> dict[str, float]:
    if not records:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    values = np.asarray([float(record[metric_name]) for record in records], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def _size_bucket_for_ratio(foreground_ratio: float) -> str:
    for bucket_name, lower, upper in SIZE_BUCKET_THRESHOLDS:
        if foreground_ratio >= lower and (upper is None or foreground_ratio < upper):
            return bucket_name
    return "large"


def build_metric_reports(
    records: list[dict[str, float | int]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build the public summary plus a richer breakdown for later analysis."""
    overall_counts = aggregate_confusion_counts(records)
    overall = metrics_from_counts(overall_counts)

    if not records:
        summary = {"num_images": 0, "overall": overall, "mean_per_image": {}}
        breakdown = {
            "counts": {
                "num_images": 0,
                "gt_non_empty_count": 0,
                "gt_empty_count": 0,
                "pred_non_empty_count": 0,
                "pred_empty_count": 0,
                "gt_and_pred_empty_count": 0,
            },
            "non_empty_only": {},
            "empty_image_false_positive_rate": 0.0,
            "size_buckets": {},
            "per_image_quantiles": {},
        }
        return summary, breakdown

    metric_names = ("iou", "f1", "precision", "recall", "pixel_accuracy")
    mean_per_image = {
        name: _mean_metric(records, name)
        for name in metric_names
    }

    gt_non_empty_records = [
        record for record in records if float(record.get("gt_foreground_ratio", 0.0)) > 0.0
    ]
    gt_empty_records = [
        record for record in records if float(record.get("gt_foreground_ratio", 0.0)) <= 0.0
    ]
    pred_non_empty_records = [
        record for record in records if float(record.get("pred_foreground_ratio", 0.0)) > 0.0
    ]
    empty_both_count = sum(
        1
        for record in records
        if float(record.get("gt_foreground_ratio", 0.0)) <= 0.0
        and float(record.get("pred_foreground_ratio", 0.0)) <= 0.0
    )
    empty_image_fp_rate = (
        float(
            sum(
                1
                for record in gt_empty_records
                if float(record.get("pred_foreground_ratio", 0.0)) > 0.0
            )
            / len(gt_empty_records)
        )
        if gt_empty_records
        else 0.0
    )

    non_empty_only = {
        name: _mean_metric(gt_non_empty_records, name)
        for name in ("iou", "f1", "precision", "recall")
    }
    non_empty_only["num_images"] = len(gt_non_empty_records)

    size_buckets: dict[str, dict[str, float | int]] = {}
    for bucket_name, _, _ in SIZE_BUCKET_THRESHOLDS:
        bucket_records = [
            record
            for record in gt_non_empty_records
            if str(record.get("size_bucket", "")) == bucket_name
        ]
        size_buckets[bucket_name] = {
            "num_images": len(bucket_records),
            "iou": _mean_metric(bucket_records, "iou"),
            "f1": _mean_metric(bucket_records, "f1"),
            "precision": _mean_metric(bucket_records, "precision"),
            "recall": _mean_metric(bucket_records, "recall"),
        }

    per_image_quantiles = {
        "iou": _metric_quantiles(records, "iou"),
        "f1": _metric_quantiles(records, "f1"),
    }

    summary = {
        "num_images": len(records),
        "overall": overall,
        "mean_per_image": mean_per_image,
    }
    breakdown = {
        "counts": {
            "num_images": len(records),
            "gt_non_empty_count": len(gt_non_empty_records),
            "gt_empty_count": len(gt_empty_records),
            "pred_non_empty_count": len(pred_non_empty_records),
            "pred_empty_count": len(records) - len(pred_non_empty_records),
            "gt_and_pred_empty_count": empty_both_count,
        },
        "non_empty_only": non_empty_only,
        "empty_image_false_positive_rate": empty_image_fp_rate,
        "size_buckets": size_buckets,
        "per_image_quantiles": per_image_quantiles,
    }
    return summary, breakdown


def summarize_metric_records(records: list[dict[str, float | int]]) -> dict[str, object]:
    """Build overall and mean-per-image summaries."""
    summary, _ = build_metric_reports(records)
    return summary


def enrich_record_with_foreground_stats(record: dict[str, float | int]) -> dict[str, float | int]:
    """Attach empty/non-empty and size-bucket annotations to a per-image metric record."""
    gt_foreground_ratio = float(record.get("gt_foreground_ratio", 0.0))
    pred_foreground_ratio = float(record.get("pred_foreground_ratio", 0.0))
    enriched = dict(record)
    enriched["gt_empty"] = int(gt_foreground_ratio <= 0.0)
    enriched["pred_empty"] = int(pred_foreground_ratio <= 0.0)
    enriched["size_bucket"] = _size_bucket_for_ratio(gt_foreground_ratio) if gt_foreground_ratio > 0.0 else "empty"
    return enriched
