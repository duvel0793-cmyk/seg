"""Sequential ablation runner for Clean-MODG."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/clean_dual_scale.yaml",
    "configs/hier_dual_scale.yaml",
    "configs/ablate_tight_only.yaml",
    "configs/ablate_context_only.yaml",
    "configs/ablate_local_window_attention.yaml",
    "configs/ablate_mask_pooling.yaml",
    "configs/ablate_ce_loss.yaml",
    "configs/ablate_focal_loss.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequence of Clean-MODG ablations.")
    parser.add_argument("--configs", type=str, nargs="*", default=DEFAULT_CONFIGS, help="List of YAML configs.")
    parser.add_argument("--print-only", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config in args.configs:
        cmd = [sys.executable, "train.py", "--config", config]
        print(" ".join(cmd))
        if args.print_only:
            continue
        subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
