from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_path"] = str(config_path)
    return config


def merge_config(config: dict[str, Any], updates: dict[str, Any] | None = None) -> dict[str, Any]:
    if not updates:
        return copy.deepcopy(config)
    return _deep_merge(config, updates)


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(config)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)

