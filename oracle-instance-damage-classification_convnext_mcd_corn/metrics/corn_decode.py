from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import torch

from losses.corn_loss import decode_threshold_count, logits_to_threshold_probabilities
from metrics.classification import compute_classification_metrics


def decode_corn_logits_with_thresholds(logits: torch.Tensor, thresholds: Sequence[float] | None = None) -> torch.Tensor:
    if thresholds is None:
        return decode_threshold_count(logits)
    return decode_threshold_count(logits, list(thresholds))


def search_best_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidates: Sequence[float],
    class_names: list[str],
    metric_name: str = "macro_f1",
) -> dict[str, Any]:
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape [N, K-1].")
    if logits.shape[0] == 0:
        raise ValueError("Cannot calibrate thresholds on an empty set.")
    if not candidates:
        raise ValueError("Threshold candidates must not be empty.")

    labels = labels.long().view(-1).cpu()
    best_thresholds = [0.5] * logits.shape[1]
    best_score = float("-inf")
    best_metrics: dict[str, Any] | None = None
    best_distance = float("inf")

    for thresholds in product(candidates, repeat=logits.shape[1]):
        preds = decode_corn_logits_with_thresholds(logits, thresholds).cpu().tolist()
        metrics = compute_classification_metrics(labels.tolist(), preds, class_names)
        score = float(metrics.get(metric_name, float("-inf")))
        distance = float(sum(abs(float(value) - 0.5) for value in thresholds))
        if score > best_score or (abs(score - best_score) <= 1e-12 and distance < best_distance):
            best_score = score
            best_distance = distance
            best_thresholds = [float(v) for v in thresholds]
            best_metrics = metrics

    threshold_probs = logits_to_threshold_probabilities(logits)
    return {
        "best_thresholds": best_thresholds,
        "best_metric_name": metric_name,
        "best_metric_value": float(best_score),
        "best_metrics": best_metrics or {},
        "num_candidates": int(len(candidates) ** logits.shape[1]),
        "mean_threshold_probability": threshold_probs.mean(dim=0).cpu().tolist(),
    }

