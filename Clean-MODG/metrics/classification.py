"""Classification metrics for fixed xBD damage classes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from utils.common import CLASS_NAMES


def accuracy_score_np(preds: Sequence[int], targets: Sequence[int]) -> float:
    return float(accuracy_score(targets, preds))


def macro_f1_score_np(preds: Sequence[int], targets: Sequence[int]) -> float:
    return float(f1_score(targets, preds, average="macro", labels=list(range(len(CLASS_NAMES))), zero_division=0))


def per_class_f1_score_np(preds: Sequence[int], targets: Sequence[int]) -> list[float]:
    scores = f1_score(targets, preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    return [float(score) for score in scores]


def precision_recall_np(preds: Sequence[int], targets: Sequence[int]) -> tuple[list[float], list[float]]:
    precision = precision_score(targets, preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    recall = recall_score(targets, preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    return [float(x) for x in precision], [float(x) for x in recall]


def classification_report_dict(preds: Sequence[int], targets: Sequence[int]) -> dict:
    return classification_report(
        targets,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
