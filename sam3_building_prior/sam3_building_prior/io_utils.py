"""Shared filesystem, logging, and serialization helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence
import json
import logging

from PIL import Image

from .config import IMAGE_EXTENSIONS, OUTPUT_SUBDIR_NAMES


def is_pre_disaster_image(path: Path) -> bool:
    """Return True when the path is a supported pre-disaster image."""
    suffix = path.suffix.lower()
    return (
        path.is_file()
        and suffix in IMAGE_EXTENSIONS
        and path.name.lower().endswith(f"_pre_disaster{suffix}")
    )


def scan_input_dir(input_dir: Path, num_workers: int = 1) -> list[Path]:
    """Collect supported pre-disaster images from a directory or a single file."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir.is_file():
        if not is_pre_disaster_image(input_dir):
            raise ValueError(
                f"Input file is not a supported pre-disaster image: {input_dir}"
            )
        logging.info("Using single pre-disaster image: %s", input_dir)
        return [input_dir]

    candidates = [path for path in input_dir.rglob("*") if path.is_file()]
    if num_workers > 1 and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            keep_flags = list(executor.map(is_pre_disaster_image, candidates))
        results = [path for path, keep in zip(candidates, keep_flags) if keep]
    else:
        results = [path for path in candidates if is_pre_disaster_image(path)]

    results = sorted(results)
    logging.info("Found %d pre-disaster images in %s", len(results), input_dir)
    return results


def slice_paths(paths: list[Path], start_index: int = 0, limit: int = 0) -> list[Path]:
    """Slice an ordered list of paths using a start index and optional limit."""
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if limit < 0:
        raise ValueError("limit must be >= 0")

    sliced = paths[start_index:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def ensure_dir(path: Path) -> Path:
    """Create a directory when it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return the standard output subdirectories for a run."""
    directories = {"root": output_dir}
    for name in OUTPUT_SUBDIR_NAMES:
        directories[name] = output_dir / name
    for path in directories.values():
        ensure_dir(path)
    return directories


def configure_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    """Configure console and optional file logging for the current process."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        ensure_dir(log_file.parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def write_json(path: Path, payload: Any) -> None:
    """Serialize JSON payloads with UTF-8 and readable indentation."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_binary_mask(path: Path, mask: Any) -> None:
    """Save a binary mask as a single-channel 0/255 image."""
    ensure_dir(path.parent)
    Image.fromarray((mask.astype("uint8") * 255)).save(path)


def resolve_matching_image_path(
    search_root: Path,
    candidate_names: Sequence[str] | None = None,
    candidate_stems: Sequence[str] | None = None,
    suffixes: Sequence[str] = IMAGE_EXTENSIONS,
) -> Path | None:
    """Resolve the first existing image path that matches the provided names or stems."""
    if search_root.is_file():
        return search_root

    seen_names: set[str] = set()
    ordered_names: list[str] = []
    for name in candidate_names or ():
        if name and name not in seen_names:
            seen_names.add(name)
            ordered_names.append(name)
    for stem in candidate_stems or ():
        if not stem:
            continue
        for suffix in suffixes:
            candidate_name = f"{stem}{suffix}"
            if candidate_name in seen_names:
                continue
            seen_names.add(candidate_name)
            ordered_names.append(candidate_name)

    for name in ordered_names:
        candidate = search_root / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def read_image(path: Path) -> Image.Image:
    """Read an input image as RGB."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logging.exception("Failed to read image %s: %s", path, exc)
        raise
