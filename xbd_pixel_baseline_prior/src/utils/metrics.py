from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch


CLASS_NAMES = ["background", "no_damage", "minor_damage", "major_damage", "destroyed"]


def _to_numpy(array: np.ndarray | torch.Tensor | Sequence[int]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def compute_confusion_matrix(
    prediction: np.ndarray | torch.Tensor | Sequence[int],
    target: np.ndarray | torch.Tensor | Sequence[int],
    num_classes: int,
) -> np.ndarray:
    prediction_array = _to_numpy(prediction).astype(np.int64, copy=False)
    target_array = _to_numpy(target).astype(np.int64, copy=False)

    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction_array.shape} vs {target_array.shape}"
        )

    valid_mask = (
        (target_array >= 0)
        & (target_array < num_classes)
        & (prediction_array >= 0)
        & (prediction_array < num_classes)
    )
    flat_index = (target_array[valid_mask] * num_classes) + prediction_array[valid_mask]
    histogram = np.bincount(flat_index.ravel(), minlength=num_classes * num_classes)
    return histogram.reshape(num_classes, num_classes)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator != 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def summarize_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    ignore_background_in_macro: bool = True,
) -> dict[str, Any]:
    confusion_matrix = confusion_matrix.astype(np.float64, copy=False)
    true_positive = np.diag(confusion_matrix)
    gt_pixels = confusion_matrix.sum(axis=1)
    pred_pixels = confusion_matrix.sum(axis=0)

    precision = _safe_divide(true_positive, pred_pixels)
    recall = _safe_divide(true_positive, gt_pixels)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    iou = _safe_divide(true_positive, gt_pixels + pred_pixels - true_positive)

    total = confusion_matrix.sum()
    overall_pixel_accuracy = float(true_positive.sum() / total) if total > 0 else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    for index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "iou": float(iou[index]),
            "support": int(gt_pixels[index]),
        }

    macro_indices = list(range(1, len(class_names))) if ignore_background_in_macro else list(
        range(len(class_names))
    )
    damage_macro_f1 = float(np.mean(f1[macro_indices])) if macro_indices else 0.0

    return {
        "overall_pixel_accuracy": overall_pixel_accuracy,
        "per_class": per_class,
        "damage_macro_f1": damage_macro_f1,
        "no_damage_f1": float(f1[1]) if len(f1) > 1 else 0.0,
        "minor_damage_f1": float(f1[2]) if len(f1) > 2 else 0.0,
        "major_damage_f1": float(f1[3]) if len(f1) > 3 else 0.0,
        "destroyed_f1": float(f1[4]) if len(f1) > 4 else 0.0,
        "confusion_matrix": confusion_matrix.astype(np.int64).tolist(),
    }


def summarize_binary_localization(confusion_matrix: np.ndarray) -> dict[str, float | list[list[int]]]:
    confusion_matrix = confusion_matrix.astype(np.float64, copy=False)
    true_positive = confusion_matrix[1, 1]
    gt_positive = confusion_matrix[1, :].sum()
    pred_positive = confusion_matrix[:, 1].sum()

    precision = float(true_positive / pred_positive) if pred_positive > 0 else 0.0
    recall = float(true_positive / gt_positive) if gt_positive > 0 else 0.0
    denominator = precision + recall
    f1 = float((2.0 * precision * recall) / denominator) if denominator > 0 else 0.0
    iou_denominator = gt_positive + pred_positive - true_positive
    iou = float(true_positive / iou_denominator) if iou_denominator > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "confusion_matrix": confusion_matrix.astype(np.int64).tolist(),
    }


@dataclass
class RunningSegmentationMetrics:
    num_classes: int = 5
    class_names: Sequence[str] = tuple(CLASS_NAMES)

    def __post_init__(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.localization_confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction_np = _to_numpy(prediction).astype(np.int64, copy=False)
        target_np = _to_numpy(target).astype(np.int64, copy=False)

        self.confusion_matrix += compute_confusion_matrix(
            prediction_np,
            target_np,
            num_classes=self.num_classes,
        )
        self.localization_confusion_matrix += compute_confusion_matrix(
            (prediction_np > 0).astype(np.int64, copy=False),
            (target_np > 0).astype(np.int64, copy=False),
            num_classes=2,
        )

    def summarize(self, ignore_background_in_macro: bool = True) -> dict[str, Any]:
        summary = summarize_confusion_matrix(
            self.confusion_matrix,
            class_names=self.class_names,
            ignore_background_in_macro=ignore_background_in_macro,
        )
        summary["building_localization"] = summarize_binary_localization(
            self.localization_confusion_matrix
        )
        summary["building_localization_f1"] = summary["building_localization"]["f1"]
        return summary


def format_confusion_matrix(confusion_matrix: Iterable[Iterable[int]]) -> str:
    matrix = np.asarray(list(confusion_matrix), dtype=np.int64)
    return np.array2string(matrix, separator=", ")
