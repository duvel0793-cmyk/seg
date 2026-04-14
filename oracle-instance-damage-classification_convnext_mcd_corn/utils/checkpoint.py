from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    return torch.load(checkpoint_path, map_location=map_location)


def save_json(payload: dict[str, Any] | list[Any], path: str | Path) -> None:
    json_path = Path(path).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

