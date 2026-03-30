"""Project defaults and path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_XBD_ROOT = Path("/home/lky/data/xBD")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "sam3.pt"
DEFAULT_TEXT_PROMPT = "building"
DEFAULT_MIN_AREA = 64
DEFAULT_PROMPT_MIN_AREA = 64
DEFAULT_NUM_WORKERS = 1
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SUPPORTED_SPLITS = ("train", "test", "hold", "tier3")
SUPPORTED_PROMPT_MODES = ("text", "box", "point", "box_point")
OUTPUT_SUBDIR_NAMES = ("masks", "instances", "visualizations", "prompts", "logs")
RUN_SUMMARY_JSON_NAME = "summary.json"
METRICS_SUMMARY_JSON_NAME = "metrics_summary.json"
PER_IMAGE_METRICS_CSV_NAME = "per_image_metrics.csv"
EXPERIMENT_SUMMARY_JSON_NAME = "experiment_summary.json"
DEFAULT_LOG_FILENAME = "run.log"


def resolve_split_input_dir(split: str, xbd_root: Path = DEFAULT_XBD_ROOT) -> Path:
    """Resolve the input image directory for an xBD split."""
    return xbd_root / split / "images"


def resolve_split_gt_dir(split: str, xbd_root: Path = DEFAULT_XBD_ROOT) -> Path:
    """Resolve the ground-truth mask directory for an xBD split."""
    return xbd_root / split / "targets"


def build_default_output_dir(
    split: Optional[str],
    prompt_mode: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Build a conventional output directory name for batch experiments."""
    split_name = split or "custom"
    mode_name = "text" if prompt_mode == "text" else f"refine_{prompt_mode}"
    return output_root / f"xbd_{split_name}_{mode_name}"


def resolve_default_log_file(output_dir: Path) -> Path:
    """Return the default log file path for a run."""
    return output_dir / "logs" / DEFAULT_LOG_FILENAME


def resolve_run_summary_path(output_dir: Path) -> Path:
    """Return the default run summary path for an inference batch."""
    return output_dir / RUN_SUMMARY_JSON_NAME


def resolve_metrics_base_dir(pred_path: Path) -> Path:
    """Return the canonical experiment directory for prediction outputs."""
    if pred_path.is_dir() and pred_path.name == "masks":
        return pred_path.parent
    return pred_path.parent if pred_path.is_file() else pred_path


def resolve_metrics_summary_path(base_dir: Path) -> Path:
    """Return the default evaluation summary JSON path."""
    return base_dir / METRICS_SUMMARY_JSON_NAME


def resolve_per_image_metrics_csv_path(base_dir: Path) -> Path:
    """Return the default per-image CSV path."""
    return base_dir / PER_IMAGE_METRICS_CSV_NAME


def resolve_experiment_summary_path(output_dir: Path) -> Path:
    """Return the default full-experiment summary path."""
    return output_dir / EXPERIMENT_SUMMARY_JSON_NAME
