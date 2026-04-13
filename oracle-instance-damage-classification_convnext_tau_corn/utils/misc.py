"""Miscellaneous helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch


class AverageMeter:
    """Track running averages."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += int(n)


def ensure_dir(path: str | Path) -> str:
    """Create a directory and return its string path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def make_run_dir(base_dir: str | Path, run_name: str | None = None) -> str:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "run"
    run_dir = Path(base_dir) / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor fields to a target device."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
