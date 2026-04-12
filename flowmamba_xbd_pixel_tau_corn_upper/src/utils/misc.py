"""Miscellaneous helpers shared by scripts and engine code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .vis_predictions import save_prediction_bundle


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


@dataclass
class AverageMeter:
    """Track scalar means for logging."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def save_debug_visualization(
    save_path: str | Path,
    sample: Dict[str, Any],
    outputs: Dict[str, Any],
) -> None:
    """Save a simple visual strip for smoke-test debugging."""
    save_path = Path(save_path).expanduser().resolve()
    save_prediction_bundle(
        save_dir=save_path.parent,
        sample=sample,
        outputs=outputs,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        stem=save_path.stem,
    )
