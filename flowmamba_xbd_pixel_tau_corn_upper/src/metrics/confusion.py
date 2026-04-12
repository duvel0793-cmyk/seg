"""Small confusion-matrix helper."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int | None = None, class_names: List[str] | None = None) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.class_names = class_names or [str(idx) for idx in range(self.num_classes)]
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, target, pred) -> None:
        target_np = self._to_numpy(target).astype(np.int64).reshape(-1)
        pred_np = self._to_numpy(pred).astype(np.int64).reshape(-1)

        mask = (target_np >= 0) & (target_np < self.num_classes)
        if self.ignore_index is not None:
            mask &= target_np != self.ignore_index
        mask &= (pred_np >= 0) & (pred_np < self.num_classes)
        if not np.any(mask):
            return

        indices = self.num_classes * target_np[mask] + pred_np[mask]
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    @staticmethod
    def _to_numpy(array):
        if torch.is_tensor(array):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def precision_per_class(self) -> np.ndarray:
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        return tp / np.clip(tp + fp, 1, None)

    def recall_per_class(self) -> np.ndarray:
        tp = np.diag(self.matrix)
        fn = self.matrix.sum(axis=1) - tp
        return tp / np.clip(tp + fn, 1, None)

    def f1_per_class(self) -> np.ndarray:
        precision = self.precision_per_class()
        recall = self.recall_per_class()
        return 2.0 * precision * recall / np.clip(precision + recall, 1e-12, None)

    def support_per_class(self) -> np.ndarray:
        return self.matrix.sum(axis=1)

    def predicted_area_per_class(self) -> np.ndarray:
        return self.matrix.sum(axis=0)

    def summary(self) -> Dict[str, object]:
        precision = self.precision_per_class()
        recall = self.recall_per_class()
        f1 = self.f1_per_class()
        support = self.support_per_class()
        predicted = self.predicted_area_per_class()
        per_class = []
        for idx, name in enumerate(self.class_names):
            per_class.append(
                {
                    "class_id": idx,
                    "class_name": name,
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "target_area": int(support[idx]),
                    "pred_area": int(predicted[idx]),
                }
            )
        return {
            "matrix": self.matrix.tolist(),
            "per_class": per_class,
        }

    def reset(self) -> None:
        self.matrix.fill(0)

