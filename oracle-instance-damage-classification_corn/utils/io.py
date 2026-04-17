from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def load_state_dict_with_fallback(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    strict: bool = True,
    description: str = "module",
) -> dict[str, Any]:
    def _load_with_shape_filter(error_message: str) -> dict[str, Any]:
        current_state_dict = module.state_dict()
        filtered_state_dict: dict[str, Any] = {}
        skipped_shape_mismatch: list[str] = []
        unexpected_from_filter: list[str] = []

        for key, value in state_dict.items():
            if key not in current_state_dict:
                unexpected_from_filter.append(key)
                continue
            current_value = current_state_dict[key]
            if (
                isinstance(value, torch.Tensor)
                and isinstance(current_value, torch.Tensor)
                and tuple(value.shape) != tuple(current_value.shape)
            ):
                skipped_shape_mismatch.append(key)
                continue
            filtered_state_dict[key] = value

        incompatible = module.load_state_dict(filtered_state_dict, strict=False)
        unexpected_keys = list(incompatible.unexpected_keys)
        if unexpected_from_filter:
            unexpected_keys.extend(unexpected_from_filter)
            unexpected_keys = sorted(set(unexpected_keys))
        return {
            "strict": False,
            "fallback_used": True,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": unexpected_keys,
            "shape_mismatch_keys": skipped_shape_mismatch,
            "error": error_message,
        }

    try:
        incompatible = module.load_state_dict(state_dict, strict=strict)
        return {
            "strict": strict,
            "fallback_used": False,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "shape_mismatch_keys": [],
            "error": None,
        }
    except RuntimeError as exc:
        if strict:
            return _load_with_shape_filter(f"{description} strict load failed: {exc}")
        return _load_with_shape_filter(f"{description} non-strict load encountered shape mismatch: {exc}")
