"""Manifest utilities for the local xBD layout."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SPLIT_CANDIDATES = ("train", "test", "hold", "tier3")


def _dedupe(values: List[Optional[str]]) -> List[Optional[str]]:
    seen = set()
    ordered: List[Optional[str]] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _sample_paths(base_dir: Path, image_id: str) -> Dict[str, Path]:
    return {
        "pre_image": base_dir / "images" / f"{image_id}_pre_disaster.png",
        "post_image": base_dir / "images" / f"{image_id}_post_disaster.png",
        "post_target": base_dir / "targets" / f"{image_id}_post_disaster_target.png",
        "post_json": base_dir / "labels" / f"{image_id}_post_disaster.json",
    }


def resolve_sample_paths(
    data_root: str | Path,
    image_id: str,
    preferred_split: str | None = None,
) -> Tuple[Dict[str, str] | None, Dict[str, object]]:
    root = Path(data_root).expanduser().resolve()
    search_order = _dedupe([preferred_split, *SPLIT_CANDIDATES, None])

    for split in search_order:
        base_dir = root / split if split else root
        paths = _sample_paths(base_dir, image_id)
        exists = {name: path.exists() for name, path in paths.items()}
        if any(exists.values()):
            missing = [name for name, ok in exists.items() if not ok]
            if not missing:
                item = {"image_id": image_id, "split": split or "root"}
                item.update({name: str(path) for name, path in paths.items()})
                return item, {"missing": []}
            return None, {"missing": missing, "base_dir": str(base_dir)}

    return None, {"missing": list(paths.keys()), "base_dir": "not_found"}


def load_ids_from_list(list_path: str | Path) -> List[str]:
    path = Path(list_path).expanduser().resolve()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def scan_ids_from_split(data_root: str | Path, split: str) -> List[str]:
    split_root = Path(data_root).expanduser().resolve() / split / "images"
    if not split_root.exists():
        return []
    ids = []
    for image_path in sorted(split_root.glob("*_post_disaster.png")):
        ids.append(image_path.name.replace("_post_disaster.png", ""))
    return ids


def build_manifest(
    data_root: str | Path,
    preferred_split: str,
    list_path: str | Path | None = None,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    image_ids = load_ids_from_list(list_path) if list_path else []
    source = "list"
    if not image_ids:
        image_ids = scan_ids_from_split(data_root, preferred_split)
        source = "scan"

    manifest: List[Dict[str, str]] = []
    missing_records: List[Dict[str, object]] = []
    missing_counter: Counter[str] = Counter()

    for image_id in image_ids:
        item, info = resolve_sample_paths(data_root, image_id, preferred_split=preferred_split)
        if item is not None:
            manifest.append(item)
            continue
        missing_counter.update(info.get("missing", []))
        missing_records.append({"image_id": image_id, **info})

    stats = {
        "source": source,
        "preferred_split": preferred_split,
        "num_ids": len(image_ids),
        "num_valid": len(manifest),
        "num_missing": len(missing_records),
        "missing_by_type": dict(missing_counter),
        "missing_records": missing_records,
    }
    return manifest, stats


def save_manifest(path: str | Path, manifest: List[Dict[str, str]], stats: Dict[str, object]) -> None:
    payload = {"manifest": manifest, "stats": stats}
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
