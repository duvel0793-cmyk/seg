from __future__ import annotations

import argparse

from engine.trainer import run_training
from utils.misc import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, choices=["localization", "damage", "joint"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.stage is not None:
        config.setdefault("training", {})["stage"] = args.stage
    if args.output_dir is not None:
        config.setdefault("project", {})["output_dir"] = args.output_dir
    run_training(config)


if __name__ == "__main__":
    main()
