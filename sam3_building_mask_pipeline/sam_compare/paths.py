"""Project path utilities and default path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "sam_compare"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPSTREAM_SAM3_DIR = PROJECT_ROOT / "sam3"
DEFAULT_XBD_ROOT = Path("/home/lky/data/xBD")
CHECKPOINT_ENV_VARS = ("SAM3_CHECKPOINT", "SAM_COMPARE_SAM3_CHECKPOINT")


def default_checkpoint_candidates() -> list[Path]:
    """Return common local checkpoint locations to probe."""
    return [
        PROJECT_ROOT / "sam3.pt",
        PROJECT_ROOT / "weight" / "sam3.pt",
        PROJECT_ROOT / "weights" / "sam3.pt",
        PROJECT_ROOT / "checkpoints" / "sam3.pt",
        Path.cwd() / "sam3.pt",
        Path.cwd() / "weight" / "sam3.pt",
        Path.cwd() / "weights" / "sam3.pt",
    ]


def resolve_checkpoint_path(
    explicit_path: Optional[str | os.PathLike[str]],
    *,
    must_exist: bool = False,
) -> Optional[Path]:
    """Resolve a checkpoint path from CLI/config/env/default candidates."""
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    for env_name in CHECKPOINT_ENV_VARS:
        env_value = os.environ.get(env_name)
        if env_value:
            candidate = Path(env_value).expanduser().resolve()
            if not must_exist or candidate.exists():
                return candidate

    for candidate in default_checkpoint_candidates():
        if candidate.exists():
            return candidate.resolve()

    return None


def ensure_output_root(path: Optional[str | os.PathLike[str]]) -> Path:
    """Resolve the output root directory."""
    if path is None:
        return OUTPUTS_DIR.resolve()
    return Path(path).expanduser().resolve()
