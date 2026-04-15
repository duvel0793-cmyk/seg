#!/usr/bin/env python3
"""Run xBD val/test split inference and evaluation for the standalone SAM3 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_compare.config import load_experiment_config
from sam_compare.infer import evaluate_model
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
    parser.add_argument("--output-dir", type=str, default=None, help="Evaluation output directory.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--image-size", type=int, default=None, help="Resized square image size fed to SAM3.")
    parser.add_argument("--eval-split", type=str, default=None, help="Evaluation split: test or val.")
    parser.add_argument("--threshold", type=float, default=None, help="Binary mask threshold.")
    parser.add_argument(
        "--save-pred-masks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export predicted PNG masks.",
    )
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export prediction-vs-label comparisons and prediction overlays on the original image.",
    )
    parser.add_argument(
        "--enable-tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable conservative test-time augmentation.",
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
        "--enable-threshold-sweep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run a threshold sweep on the reproducible validation subset.",
    )
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=None,
        help="Threshold candidates used when sweep is enabled.",
    )
    parser.add_argument(
        "--threshold-json",
        type=str,
        default=None,
        help="Path to a previously saved threshold_sweep.json file.",
    )
    parser.add_argument(
        "--postprocess-json",
        type=str,
        default=None,
        help="Path to a JSON file containing min_component_area and max_hole_area.",
    )
    parser.add_argument(
        "--enable-postprocess",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable conservative small-component removal and hole filling.",
    )
    parser.add_argument("--min-component-area", type=int, default=None, help="Minimum isolated component area to keep.")
    parser.add_argument("--max-hole-area", type=int, default=None, help="Maximum enclosed hole area to fill.")
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
        output_dir = config.paths.output_root / "xbd_sam3_eval"
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
    if args.eval_split is not None:
        config.eval.eval_split = args.eval_split
    if args.threshold is not None:
        config.eval.threshold = args.threshold
    if args.save_pred_masks is not None:
        config.eval.save_pred_masks = args.save_pred_masks
    if args.save_visualizations is not None:
        config.eval.save_visualizations = args.save_visualizations
    if args.enable_tta is not None:
        config.eval.enable_tta = args.enable_tta
    if args.tta_hflip is not None:
        config.eval.tta_hflip = args.tta_hflip
    if args.tta_vflip is not None:
        config.eval.tta_vflip = args.tta_vflip
    if args.tta_rot90 is not None:
        config.eval.tta_rot90 = args.tta_rot90
    if args.enable_threshold_sweep is not None:
        config.eval.enable_threshold_sweep = args.enable_threshold_sweep
    if args.threshold_candidates is not None:
        config.eval.threshold_candidates = list(args.threshold_candidates)
    if args.threshold_json is not None:
        config.eval.threshold_json = Path(args.threshold_json).expanduser().resolve()
    if args.postprocess_json is not None:
        config.eval.postprocess_json = Path(args.postprocess_json).expanduser().resolve()
    if args.enable_postprocess is not None:
        config.eval.enable_postprocess = args.enable_postprocess
    if args.min_component_area is not None:
        config.eval.min_component_area = args.min_component_area
    if args.max_hole_area is not None:
        config.eval.max_hole_area = args.max_hole_area

    artifacts = evaluate_model(config, output_dir=output_dir, checkpoint_path=args.checkpoint)
    summary = {
        "output_dir": str(artifacts.output_dir),
        "metrics_json": str(artifacts.metrics_json_path),
        "metrics_breakdown_json": str(artifacts.metrics_breakdown_path),
        "per_image_metrics_csv": str(artifacts.per_image_csv_path),
        "eval_config_used_json": str(artifacts.eval_config_used_path),
        "postprocess_config_used_json": str(artifacts.postprocess_config_used_path),
        "pred_masks_dir": str(artifacts.pred_masks_dir) if artifacts.pred_masks_dir else None,
        "comparison_vis_dir": str(artifacts.comparison_vis_dir) if artifacts.comparison_vis_dir else None,
        "overlay_vis_dir": str(artifacts.overlay_vis_dir) if artifacts.overlay_vis_dir else None,
        "checkpoint_path": str(artifacts.checkpoint_path) if artifacts.checkpoint_path else None,
        "threshold_sweep_json": str(artifacts.threshold_sweep_path) if artifacts.threshold_sweep_path else None,
        "tta_used_json": str(artifacts.tta_used_path) if artifacts.tta_used_path else None,
        "num_images": artifacts.num_images,
        "used_threshold": artifacts.used_threshold,
        "eval_split": artifacts.eval_split,
    }
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
