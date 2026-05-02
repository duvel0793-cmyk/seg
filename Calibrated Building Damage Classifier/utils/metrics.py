from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import warnings
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

    target = np.zeros_like(probabilities)
    target[np.arange(len(y_true)), y_true] = 1.0
    cdf_pred = np.cumsum(probabilities, axis=1)
    cdf_target = np.cumsum(target, axis=1)
    deltas = np.diff(positions)
    sample_emd = np.sum(np.abs(cdf_pred[:, :-1] - cdf_target[:, :-1]) * deltas[None, :], axis=1)
    return float(sample_emd.mean()) if sample_emd.size else 0.0, sample_emd


def compute_expected_severity_error(
    probabilities: np.ndarray,
    y_true: list[int] | np.ndarray,
    positions: list[float] | np.ndarray,
) -> tuple[float, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    expected = (probabilities * positions[None, :]).sum(axis=1)
    target = positions[y_true]
    sample_error = np.abs(expected - target)
    return float(sample_error.mean()) if sample_error.size else 0.0, sample_error


def compute_expected_severity_bias(
    probabilities: np.ndarray,
    y_true: list[int] | np.ndarray,
    positions: list[float] | np.ndarray,
) -> tuple[float, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    expected = (probabilities * positions[None, :]).sum(axis=1)
    target = positions[y_true]
    sample_bias = expected - target
    return float(sample_bias.mean()) if sample_bias.size else 0.0, sample_bias


def compute_composite_score(
    *,
    macro_f1: float,
    qwk: float,
    far_error_rate: float,
    alpha_qwk: float,
    beta_far_error: float,
) -> float:
    return float(macro_f1) + (float(alpha_qwk) * float(qwk)) - (float(beta_far_error) * float(far_error_rate))


def summarize_mean_predicted_distribution_by_true_class(
    probabilities: np.ndarray,
    y_true: list[int] | np.ndarray,
    class_names: list[str],
    positions: list[float] | np.ndarray,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)

    summary: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = y_true == class_index
        if not np.any(class_mask):
            summary[class_name] = {
                "support": 0,
                "mean_predicted_distribution": {name: 0.0 for name in class_names},
                "expected_severity_mean": 0.0,
                "expected_severity_bias": 0.0,
            }
            continue
        class_probabilities = probabilities[class_mask]
        mean_distribution = class_probabilities.mean(axis=0)
        expected_mean = float((class_probabilities * positions[None, :]).sum(axis=1).mean())
        summary[class_name] = {
            "support": int(class_mask.sum()),
            "mean_predicted_distribution": {
                pred_name: float(mean_distribution[pred_index])
                for pred_index, pred_name in enumerate(class_names)
            },
            "expected_severity_mean": expected_mean,
            "expected_severity_bias": float(expected_mean - positions[class_index]),
        }
    return summary


def compute_adjacent_far_confusion_stats(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    distances = np.abs(y_true - y_pred)
    error_mask = distances > 0
    num_errors = int(error_mask.sum())

    per_class: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = y_true == class_index
        class_distances = distances[class_mask]
        per_class[class_name] = {
            "support": int(class_mask.sum()),
            "adjacent_errors": int(np.sum(class_distances == 1)),
            "far_errors": int(np.sum(class_distances >= 2)),
            "adjacent_error_rate": float(np.mean(class_distances == 1)) if class_distances.size else 0.0,
            "far_error_rate": float(np.mean(class_distances >= 2)) if class_distances.size else 0.0,
        }

    return {
        "total_errors": num_errors,
        "adjacent_errors": int(np.sum(distances == 1)),
        "far_errors": int(np.sum(distances >= 2)),
        "adjacent_share_among_errors": float(np.mean(distances[error_mask] == 1)) if num_errors > 0 else 0.0,
        "far_share_among_errors": float(np.mean(distances[error_mask] >= 2)) if num_errors > 0 else 0.0,
        "per_true_class": per_class,
    }


def compute_minor_major_bidirectional_analysis(
    probabilities: np.ndarray,
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
    positions: list[float] | np.ndarray,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    minor_index = class_names.index("minor-damage")
    major_index = class_names.index("major-damage")
    expected = (probabilities * positions[None, :]).sum(axis=1)

    def _summarize(mask: np.ndarray, target_index: int, prediction_index: int) -> dict[str, Any]:
        count = int(mask.sum())
        target_support = int(np.sum(y_true == target_index))
        if count == 0:
            return {
                "count": 0,
                "support": target_support,
                "rate_within_true_class": 0.0,
                "mean_target_probability": 0.0,
                "mean_prediction_probability": 0.0,
                "mean_expected_severity": 0.0,
            }
        return {
            "count": count,
            "support": target_support,
            "rate_within_true_class": float(count / max(target_support, 1)),
            "mean_target_probability": float(probabilities[mask, target_index].mean()),
            "mean_prediction_probability": float(probabilities[mask, prediction_index].mean()),
            "mean_expected_severity": float(expected[mask].mean()),
        }

    minor_to_major = (y_true == minor_index) & (y_pred == major_index)
    major_to_minor = (y_true == major_index) & (y_pred == minor_index)
    return {
        "gt_minor_pred_major": _summarize(minor_to_major, minor_index, major_index),
        "gt_major_pred_minor": _summarize(major_to_minor, major_index, minor_index),
    }


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, Any], str]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    num_classes = len(class_names)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"compute_classification_metrics expects matched lengths, got "
            f"len(y_true)={y_true.shape[0]} and len(y_pred)={y_pred.shape[0]}."
        )

    if y_true.size == 0:
        zero_counts = {name: 0 for name in class_names}
        zero_per_class = {
            name: {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }
            for name in class_names
        }
        empty_metrics = {
            "overall_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_accuracy": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "QWK": 0.0,
            "adjacent_error_rate": 0.0,
            "far_error_rate": 0.0,
            "per_class": zero_per_class,
            "class_counts": zero_counts,
            "prediction_distribution": zero_counts.copy(),
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64).tolist(),
            "ordinal_error_profile": compute_ordinal_error_profile(y_true, y_pred, class_names),
        }
        report = (
            "No valid predictions were collected for this split. "
            "All classification metrics were set to 0.0."
        )
        return empty_metrics, report

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    true_counts = np.bincount(y_true, minlength=num_classes)
    ordinal_profile = compute_ordinal_error_profile(y_true, y_pred, class_names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        balanced_accuracy = float(np.nan_to_num(balanced_accuracy_score(y_true, y_pred), nan=0.0))
        quadratic_weighted_kappa = float(
            np.nan_to_num(
                cohen_kappa_score(
                    y_true,
                    y_pred,
                    labels=np.arange(num_classes),
                    weights="quadratic",
                ),
                nan=0.0,
            )
        )

    metrics = {
        "overall_accuracy": float(np.nan_to_num(accuracy_score(y_true, y_pred), nan=0.0)),
        "macro_f1": float(np.nan_to_num(f1_score(y_true, y_pred, average="macro", zero_division=0), nan=0.0)),
        "weighted_f1": float(
            np.nan_to_num(f1_score(y_true, y_pred, average="weighted", zero_division=0), nan=0.0)
        ),
        "balanced_accuracy": balanced_accuracy,
        "quadratic_weighted_kappa": quadratic_weighted_kappa,
        "adjacent_error_rate": float(ordinal_profile["adjacent_error_rate"]),
        "far_error_rate": float(ordinal_profile["far_error_rate"]),
        "per_class": {},
        "class_counts": {name: int(count) for name, count in zip(class_names, true_counts)},
        "prediction_distribution": {name: int(count) for name, count in zip(class_names, pred_counts)},
        "confusion_matrix": cm.tolist(),
        "ordinal_error_profile": ordinal_profile,
    }
    metrics["QWK"] = metrics["quadratic_weighted_kappa"]

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
        labels=np.arange(num_classes),
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
