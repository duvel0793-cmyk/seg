from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from utils.misc import ensure_dir


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(state, tmp_path)
        # Fail fast if the just-written checkpoint is unreadable.
        torch.load(tmp_path, map_location="cpu")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {path}") from exc


def resolve_checkpoint_model_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    best_model_source = str(checkpoint.get("best_model_source", "")).lower()
    if best_model_source == "raw":
        state_dict = checkpoint.get("raw_model_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("ema_state_dict")
    elif best_model_source == "ema":
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("raw_model_state_dict")
    else:
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("raw_model_state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain any model state dict.")
    return state_dict
