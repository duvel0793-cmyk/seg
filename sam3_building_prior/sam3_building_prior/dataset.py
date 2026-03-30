"""Dataset scanning helpers for xBD-style pre-disaster image batches."""

from pathlib import Path

from .io_utils import scan_input_dir, slice_paths
from .types import ImageRecord


def make_image_records(
    input_dir: Path,
    start_index: int = 0,
    limit: int = 0,
    num_workers: int = 1,
) -> list[ImageRecord]:
    """Scan and slice image records for a batch run."""
    paths = scan_input_dir(input_dir=input_dir, num_workers=num_workers)
    selected_paths = slice_paths(paths=paths, start_index=start_index, limit=limit)
    return [ImageRecord(path=path, stem=path.stem) for path in selected_paths]
