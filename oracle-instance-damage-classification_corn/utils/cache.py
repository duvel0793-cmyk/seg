from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from utils.io import ensure_dir


def _normalize_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def compute_cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha1(_normalize_payload(payload).encode("utf-8")).hexdigest()[:16]


def make_cache_path(cache_dir: str | Path, prefix: str, payload: dict[str, Any], suffix: str = ".pkl") -> Path:
    cache_dir = ensure_dir(cache_dir)
    cache_key = compute_cache_key(payload)
    return cache_dir / f"{prefix}_{cache_key}{suffix}"


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_pickle(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(data, f)
