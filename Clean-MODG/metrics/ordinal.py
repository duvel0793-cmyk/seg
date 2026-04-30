"""Ordinal metrics for instance-level damage classification."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def ordinal_mae(preds: Sequence[int], targets: Sequence[int]) -> float:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float(np.mean(np.abs(preds - targets)))


def ordinal_rmse(preds: Sequence[int], targets: Sequence[int]) -> float:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def severe_error_rate(preds: Sequence[int], targets: Sequence[int], threshold: int = 2) -> float:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float(np.mean(np.abs(preds - targets) >= threshold))
