"""YAML config loading and merging."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base and return a new dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a config file and recursively expand its includes field."""
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    merged: Dict[str, Any] = {}

    for include_path in config.pop("includes", []):
        include_full_path = (config_path.parent / include_path).resolve()
        include_config = load_config(include_full_path)
        merged = deep_merge_dicts(merged, include_config)

    merged = deep_merge_dicts(merged, config)
    merged["_config_path"] = str(config_path)
    return merged


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """Persist a merged config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = deepcopy(config)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)

