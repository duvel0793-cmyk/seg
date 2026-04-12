#!/usr/bin/env python
"""Validate local xBD layout and build project-side manifest files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import build_manifest, save_manifest
from src.utils import dump_config, ensure_dir, load_config, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare xBD manifests for the standalone project.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg["output_dir"])
    log_dir = ensure_dir(output_dir / "logs")
    logger = setup_logger("prepare_xbd_lists", log_dir / "prepare_xbd_lists.log")

    data_root = Path(cfg["data"]["data_root"]).expanduser().resolve()
    checks = [
        data_root,
        data_root / "train" / "images",
        data_root / "test" / "images",
        data_root / "targets",
        data_root / "labels",
        Path(cfg["data"]["train_list"]).expanduser().resolve(),
        Path(cfg["data"]["val_list"]).expanduser().resolve(),
    ]
    logger.info("Checking local xBD paths")
    for path in checks:
        logger.info("[%s] %s", "OK" if path.exists() else "MISS", path)

    train_manifest, train_stats = build_manifest(
        data_root=data_root,
        preferred_split=cfg["data"]["train_split"],
        list_path=cfg["data"]["train_list"],
    )
    val_manifest, val_stats = build_manifest(
        data_root=data_root,
        preferred_split=cfg["data"]["val_split"],
        list_path=cfg["data"]["val_list"],
    )

    save_manifest(log_dir / "train_manifest.json", train_manifest, train_stats)
    save_manifest(log_dir / "val_manifest.json", val_manifest, val_stats)
    dump_config(cfg, log_dir / "resolved_config.yaml")

    for split_name, stats in (("train", train_stats), ("val", val_stats)):
        logger.info(
            "%s manifest source=%s valid=%d missing=%d missing_by_type=%s",
            split_name,
            stats["source"],
            stats["num_valid"],
            stats["num_missing"],
            stats["missing_by_type"],
        )
        if stats["missing_records"]:
            preview = stats["missing_records"][:5]
            logger.info("%s missing preview=%s", split_name, preview)


if __name__ == "__main__":
    main()
