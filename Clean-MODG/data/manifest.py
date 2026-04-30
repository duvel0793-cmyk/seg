"""Manifest loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "pre_image",
    "post_image",
    "label",
    "polygon",
    "bbox",
    "building_id",
    "disaster_id",
    "split",
]


def load_manifest_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest file does not exist: {path}. "
            "Please create it with data/build_xbd_manifest.py or supply a valid manifest path."
        )
    return pd.read_csv(path)


def validate_manifest_dataframe(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    value = str(value).strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON field: {value}") from exc


def resolve_path(path_str: str, image_root: str | Path = "") -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    image_root = Path(image_root) if image_root else Path(".")
    return (image_root / path).expanduser().resolve()


def filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" not in df.columns:
        raise ValueError("Manifest has no 'split' column.")
    return df[df["split"].astype(str) == str(split)].reset_index(drop=True)


def preview_rows(df: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    if len(df) == 0:
        return []
    return df.head(n).to_dict(orient="records")


def count_missing_geometry(df: pd.DataFrame) -> int:
    count = 0
    for _, row in df.iterrows():
        polygon = parse_json_field(row.get("polygon"))
        bbox = parse_json_field(row.get("bbox"))
        if polygon is None and bbox is None:
            count += 1
    return count


def iter_manifest_records(df: pd.DataFrame) -> Iterable[dict[str, Any]]:
    for record in df.to_dict(orient="records"):
        yield record
