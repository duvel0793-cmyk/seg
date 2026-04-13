"""Classification metrics without sklearn dependency."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from metrics.confusion_matrix import compute_confusion_matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def compute_classification_metrics(
    targets: List[int] | np.ndarray,
    preds: List[int] | np.ndarray,
    num_classes: int,
    class_names: List[str] | None = None,
) -> Dict:
    """Compute accuracy, macro/weighted F1, per-class metrics, and confusion matrix."""
    targets_np = np.asarray(targets, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    confusion = compute_confusion_matrix(targets_np, preds_np, num_classes=num_classes)

    per_class = []
    supports = confusion.sum(axis=1)
    total = confusion.sum()
    correct = np.trace(confusion)
    accuracy = _safe_divide(correct, total)

    f1_values = []
    weighted_f1_sum = 0.0
    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        support = int(supports[class_idx])
        weighted_f1_sum += f1 * support
        f1_values.append(f1)
        per_class.append(
            {
                "class_idx": class_idx,
                "class_name": class_names[class_idx] if class_names else str(class_idx),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "num_samples": int(total),
    }

