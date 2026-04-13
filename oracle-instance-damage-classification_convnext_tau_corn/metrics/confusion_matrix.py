"""Confusion matrix helpers."""

from __future__ import annotations

import numpy as np


def compute_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute a dense confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, preds):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            matrix[int(target), int(pred)] += 1
    return matrix

