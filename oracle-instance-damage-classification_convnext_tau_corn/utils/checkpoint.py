"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Save a checkpoint dictionary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dictionary."""
    return torch.load(str(path), map_location=map_location)

