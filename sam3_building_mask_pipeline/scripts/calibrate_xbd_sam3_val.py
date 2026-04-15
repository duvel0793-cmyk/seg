#!/usr/bin/env python3
"""Search validation threshold and postprocess settings for xBD SAM3 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_compare.config import load_experiment_config
from sam_compare.infer import calibrate_validation_postprocess
from sam_compare.paths import CONFIGS_DIR
from sam_compare.utils import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Optional extra YAML config to merge.")
    parser.add_argument("--xbd-root", type=str, default=None, help="Path to xBD root.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Experiment checkpoint or SAM3 backbone checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Calibration output directory.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--image-size", type=int, default=None, help="Resized square image size fed to SAM3.")
    parser.add_argument(
        "--enable-tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable conservative test-time augmentation during calibration.",
    )
    parser.add_argument(
        "--tta-hflip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use horizontal-flip TTA.",
    )
    parser.add_argument(
        "--tta-vflip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use vertical-flip TTA.",
    )
    parser.add_argument(
        "--tta-rot90",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use 90-degree rotation TTA.",
    )
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=None,
        help="Threshold candidates searched on the validation split.",
    )
    parser.add_argument(
        "--min-component-area-candidates",
        type=int,
        nargs="+",
        default=None,
        help="Minimum-component-area candidates searched on the validation split.",
    )
    parser.add_argument(
        "--max-hole-area-candidates",
        type=int,
        nargs="+",
        default=None,
        help="Maximum-hole-area candidates searched on the validation split.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_paths = [CONFIGS_DIR / "default.yaml", CONFIGS_DIR / "eval.yaml"]
    if args.config:
        config_paths.append(Path(args.config))
    config = load_experiment_config(config_paths)

    if args.xbd_root is not None:
        config.paths.xbd_root = Path(args.xbd_root).expanduser().resolve()
    if args.output_dir is None:
        output_dir = config.paths.output_root / "xbd_sam3_val_calibration"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    if args.device is not None:
        config.system.device = args.device
    if args.batch_size is not None:
        config.eval.batch_size = args.batch_size
    if args.num_workers is not None:
        config.system.num_workers = args.num_workers
    if args.image_size is not None:
        config.data.image_size = args.image_size
    if args.enable_tta is not None:
        config.eval.enable_tta = args.enable_tta
    if args.tta_hflip is not None:
        config.eval.tta_hflip = args.tta_hflip
    if args.tta_vflip is not None:
        config.eval.tta_vflip = args.tta_vflip
    if args.tta_rot90 is not None:
        config.eval.tta_rot90 = args.tta_rot90

    artifacts = calibrate_validation_postprocess(
        config,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint,
        threshold_candidates=list(args.threshold_candidates) if args.threshold_candidates is not None else None,
        min_component_area_candidates=list(args.min_component_area_candidates) if args.min_component_area_candidates is not None else None,
        max_hole_area_candidates=list(args.max_hole_area_candidates) if args.max_hole_area_candidates is not None else None,
    )
    summary = {
        "output_dir": str(artifacts.output_dir),
        "calibration_json": str(artifacts.calibration_json_path),
        "selected_threshold_json": str(artifacts.selected_threshold_path),
        "selected_postprocess_json": str(artifacts.selected_postprocess_path),
        "checkpoint_path": str(artifacts.checkpoint_path) if artifacts.checkpoint_path else None,
        "best_threshold": artifacts.best_threshold,
        "best_min_component_area": artifacts.best_min_component_area,
        "best_max_hole_area": artifacts.best_max_hole_area,
    }
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
