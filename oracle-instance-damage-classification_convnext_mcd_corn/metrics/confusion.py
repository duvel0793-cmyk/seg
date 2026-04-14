from __future__ import annotations

import numpy as np


def as_numpy_confusion_matrix(confusion_matrix: list[list[int]] | np.ndarray) -> np.ndarray:
    return np.asarray(confusion_matrix, dtype=np.int64)
