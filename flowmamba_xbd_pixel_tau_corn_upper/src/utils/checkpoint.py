"""Checkpoint helpers with metadata compatibility checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from .backend_info import compare_checkpoint_metadata


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str = "cpu",
    expected_metadata: Dict[str, Any] | None = None,
    strict_metadata: bool = True,
    logger: Any | None = None,
    load_weights_only: bool = False,
) -> Dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_meta = state.get("metadata") or {}
    errors, warnings = compare_checkpoint_metadata(checkpoint_meta, expected_metadata or {})
    for warning in warnings:
        if logger is not None:
            logger.warning("Checkpoint compatibility warning: %s", warning)
    if errors and strict_metadata:
        message = " | ".join(errors)
        raise RuntimeError(f"Incompatible checkpoint metadata for {checkpoint_path}: {message}")
    if errors and logger is not None and not strict_metadata:
        for error in errors:
            logger.warning("Checkpoint compatibility relaxed error: %s", error)

    model.load_state_dict(state["model"], strict=False)
    if not load_weights_only and optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if not load_weights_only and scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if not load_weights_only and scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state
