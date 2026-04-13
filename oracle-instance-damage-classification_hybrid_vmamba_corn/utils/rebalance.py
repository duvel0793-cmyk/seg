from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler


def build_stage2_sampler(
    dataset: Any,
    *,
    mode: str = "class_balanced",
    focus_minor_major_weight: float = 1.25,
) -> WeightedRandomSampler | None:
    sampler_mode = str(mode).lower()
    if sampler_mode in {"none", "disabled", "off"}:
        return None

    class_counts = getattr(dataset, "class_counts", None)
    samples = getattr(dataset, "samples", None)
    if not class_counts or samples is None:
        return None

    counts = torch.tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
    class_weights = counts.sum() / counts
    class_weights = class_weights / class_weights.mean()
    class_weights[1] *= float(focus_minor_major_weight)
    class_weights[2] *= float(focus_minor_major_weight)

    sample_weights = torch.tensor(
        [float(class_weights[int(sample["label"])].item()) for sample in samples],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
