from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_classification_metrics(targets: list[int] | np.ndarray, preds: list[int] | np.ndarray, class_names: list[str]) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }
    for idx, class_name in enumerate(class_names):
        metrics["per_class"][class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return metrics

