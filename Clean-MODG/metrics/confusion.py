"""Confusion matrix helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from utils.common import CLASS_NAMES


def confusion_matrix(preds: Sequence[int], targets: Sequence[int]) -> np.ndarray:
    return sk_confusion_matrix(targets, preds, labels=list(range(len(CLASS_NAMES))))


def save_confusion_matrix_png(matrix: np.ndarray, path: str | Path, class_names: Sequence[str] | None = None) -> Path:
    class_names = list(class_names or CLASS_NAMES)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
