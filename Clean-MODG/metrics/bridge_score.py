"""Placeholder bridge score implementation for project compatibility."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from metrics.classification import macro_f1_score_np


def compute_bridge_score(preds: Sequence[int], targets: Sequence[int], areas: Sequence[float] | None = None) -> float:
    """Project-level placeholder for the bridge evaluation score.

    Replace this function with the user's official evaluator when available.
    The current version falls back to macro F1 and optionally blends in area-weighted accuracy.
    """

    base = macro_f1_score_np(preds, targets)
    if areas is None:
        return float(base)
    areas = np.asarray(areas, dtype=np.float64)
    if areas.size == 0 or np.allclose(areas.sum(), 0.0):
        return float(base)
    weights = areas / areas.sum()
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    weighted_accuracy = float(np.sum(weights * (preds == targets).astype(np.float64)))
    return float(0.5 * base + 0.5 * weighted_accuracy)
