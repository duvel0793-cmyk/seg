"""YAML config loading with lightweight default-file composition."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def _load_with_defaults(path: Path) -> Dict[str, Any]:
    raw = _load_yaml(path)
    defaults = raw.pop("defaults", [])
    merged: Dict[str, Any] = {}
    for default_name in defaults:
        default_path = (path.parent / default_name).resolve()
        merged = _merge_dicts(merged, _load_with_defaults(default_path))
    return _merge_dicts(merged, raw)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config and recursively merge its defaults."""

    path = Path(config_path).expanduser().resolve()
    cfg = _load_with_defaults(path)
    cfg["config_path"] = str(path)
    return cfg


def dump_config(config: Dict[str, Any], save_path: str | Path) -> None:
    path = Path(save_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
