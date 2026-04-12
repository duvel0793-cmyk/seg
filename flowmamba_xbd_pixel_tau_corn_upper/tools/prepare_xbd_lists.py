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


def log_check(logger, status: str, path: Path, note: str = "") -> None:
    msg = f"[{status}] {path}"
    if note:
        msg += f" | {note}"
    logger.info(msg)


def check_xbd_layout(logger, data_root: Path, train_list: Path, val_list: Path) -> None:
    """
    Health check for xBD layout.

    Supported layouts:
    1) split layout:
       data_root/train/{images,targets,labels}
       data_root/test/{images,targets,labels}

    2) legacy flat layout:
       data_root/{images,targets,labels}
    """
    logger.info("Checking local xBD paths")

    # required root
    log_check(logger, "OK" if data_root.exists() else "MISS", data_root, "data_root")

    split_checks = [
        (data_root / "train" / "images", "train images"),
        (data_root / "train" / "targets", "train targets"),
        (data_root / "train" / "labels", "train labels"),
        (data_root / "test" / "images", "test images"),
        (data_root / "test" / "targets", "test targets"),
        (data_root / "test" / "labels", "test labels"),
    ]

    flat_checks = [
        (data_root / "images", "flat images"),
        (data_root / "targets", "flat targets"),
        (data_root / "labels", "flat labels"),
    ]

    split_ok = all(path.exists() for path, _ in split_checks)
    flat_ok = all(path.exists() for path, _ in flat_checks)

    if split_ok:
        logger.info("Detected xBD layout: split-style")
        for path, note in split_checks:
            log_check(logger, "OK", path, note)
        for path, note in flat_checks:
            log_check(logger, "INFO", path, f"{note}; not required for current split-style layout")
    elif flat_ok:
        logger.info("Detected xBD layout: flat-style")
        for path, note in flat_checks:
            log_check(logger, "OK", path, note)
        for path, note in split_checks:
            log_check(logger, "INFO", path, f"{note}; not required for current flat-style layout")
    else:
        logger.warning("Could not fully match split-style or flat-style xBD layout.")
        logger.warning("Detailed path status below:")
        for path, note in split_checks:
            log_check(logger, "OK" if path.exists() else "MISS", path, note)
        for path, note in flat_checks:
            log_check(logger, "OK" if path.exists() else "MISS", path, note)

    # list files are optional if manifest builder supports fallback scan
    if train_list.exists():
        log_check(logger, "OK", train_list, "train list")
    else:
        log_check(logger, "WARN", train_list, "train list missing; will fallback to directory scan")

    if val_list.exists():
        log_check(logger, "OK", val_list, "val list")
    else:
        log_check(logger, "WARN", val_list, "val list missing; will fallback to directory scan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare xBD manifests for the standalone project.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg["output_dir"])
    log_dir = ensure_dir(output_dir / "logs")
    logger = setup_logger("prepare_xbd_lists", log_dir / "prepare_xbd_lists.log")

    data_root = Path(cfg["data"]["data_root"]).expanduser().resolve()
    train_list = Path(cfg["data"]["train_list"]).expanduser().resolve()
    val_list = Path(cfg["data"]["val_list"]).expanduser().resolve()

    check_xbd_layout(logger, data_root, train_list, val_list)

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