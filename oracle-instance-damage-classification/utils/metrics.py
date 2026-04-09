from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_matrix_heatmap_plot(
    matrix: np.ndarray | list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    save_path: str | Path,
    title: str,
    cmap: str = "viridis",
    value_format: str = ".2f",
    xlabel: str = "Columns",
    ylabel: str = "Rows",
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    values = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(col_labels)),
        yticks=np.arange(len(row_labels)),
        xticklabels=col_labels,
        yticklabels=row_labels,
        ylabel=ylabel,
        xlabel=xlabel,
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    threshold = float(values.max()) / 2.0 if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = format(int(value), value_format) if value_format.endswith("d") else format(float(value), value_format)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_ordinal_error_profile(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    distances = np.abs(y_true - y_pred)
    total = int(len(y_true))
    error_mask = distances > 0
    num_errors = int(error_mask.sum())

    histogram = {
        str(distance): int((distances == distance).sum())
        for distance in range(len(class_names))
    }
    per_class: dict[str, dict[str, Any]] = {}
    for class_idx, class_name in enumerate(class_names):
        class_mask = y_true == class_idx
        class_distances = distances[class_mask]
        per_class[class_name] = {
            "support": int(class_mask.sum()),
            "adjacent_error_rate": float(np.mean(class_distances == 1)) if class_distances.size else 0.0,
            "far_error_rate": float(np.mean(class_distances >= 2)) if class_distances.size else 0.0,
            "mean_absolute_distance": float(np.mean(class_distances)) if class_distances.size else 0.0,
        }

    return {
        "num_instances": total,
        "num_errors": num_errors,
        "exact_match_rate": float(np.mean(distances == 0)) if total > 0 else 0.0,
        "adjacent_error_rate": float(np.mean(distances == 1)) if total > 0 else 0.0,
        "far_error_rate": float(np.mean(distances >= 2)) if total > 0 else 0.0,
        "adjacent_error_share_among_errors": float(np.mean(distances[error_mask] == 1)) if num_errors > 0 else 0.0,
        "far_error_share_among_errors": float(np.mean(distances[error_mask] >= 2)) if num_errors > 0 else 0.0,
        "mean_absolute_class_distance": float(np.mean(distances)) if total > 0 else 0.0,
        "distance_histogram": histogram,
        "per_class": per_class,
    }


def compute_emd_from_probabilities(
    probabilities: np.ndarray,
    y_true: list[int] | np.ndarray,
    positions: list[float] | np.ndarray,
) -> tuple[float, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, K].")
    if probabilities.shape[1] != positions.shape[0]:
        raise ValueError("probabilities and positions must agree on the class dimension.")

    target = np.zeros_like(probabilities)
    target[np.arange(len(y_true)), y_true] = 1.0
    cdf_pred = np.cumsum(probabilities, axis=1)
    cdf_target = np.cumsum(target, axis=1)
    deltas = np.diff(positions)
    sample_emd = np.sum(np.abs(cdf_pred[:, :-1] - cdf_target[:, :-1]) * deltas[None, :], axis=1)
    return float(sample_emd.mean()) if sample_emd.size else 0.0, sample_emd


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, Any], str]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    true_counts = np.bincount(y_true, minlength=len(class_names))
    ordinal_profile = compute_ordinal_error_profile(y_true, y_pred, class_names)

    metrics = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
                labels=np.arange(len(class_names)),
                weights="quadratic",
            )
        ),
        "adjacent_error_rate": float(ordinal_profile["adjacent_error_rate"]),
        "far_error_rate": float(ordinal_profile["far_error_rate"]),
        "per_class": {},
        "class_counts": {name: int(count) for name, count in zip(class_names, true_counts)},
        "prediction_distribution": {name: int(count) for name, count in zip(class_names, pred_counts)},
        "confusion_matrix": cm.tolist(),
        "ordinal_error_profile": ordinal_profile,
    }

    for idx, class_name in enumerate(class_names):
        metrics["per_class"][class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    return metrics, report


def save_confusion_matrix_plot(
    confusion: np.ndarray | list[list[int]],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    save_matrix_heatmap_plot(
        matrix=np.asarray(confusion),
        row_labels=class_names,
        col_labels=class_names,
        save_path=save_path,
        title="Confusion Matrix",
        cmap="Blues",
        value_format="d",
        xlabel="Prediction",
        ylabel="Ground Truth",
    )
