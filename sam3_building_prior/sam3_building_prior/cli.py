"""Batch inference entrypoint for SAM3 building prior generation."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
import logging
import sys
import time

import numpy as np

from .config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MIN_AREA,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROMPT_MIN_AREA,
    DEFAULT_TEXT_PROMPT,
    DEFAULT_XBD_ROOT,
    SUPPORTED_PROMPT_MODES,
    SUPPORTED_SPLITS,
    build_default_output_dir,
    resolve_default_log_file,
    resolve_run_summary_path,
    resolve_split_input_dir,
)
from .dataset import make_image_records
from .io_utils import (
    build_output_dirs,
    configure_logging,
    read_image,
    save_binary_mask,
    write_json,
)
from .postprocess import (
    extract_instances_from_masks,
    extract_largest_instance_from_mask,
    merge_instances_to_mask,
)
from .prompting import (
    extract_prompt_regions,
    prompt_regions_to_json,
    read_binary_mask,
    resolve_mask_path,
)
from .sam3_adapter import SAM3Adapter
from .visualize import overlay_mask_on_image


def add_run_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Attach batch inference arguments to a parser."""
    parser.add_argument(
        "--split",
        type=str,
        choices=SUPPORTED_SPLITS,
        default=None,
        help="Convenience split name. When provided without --input-dir, uses <xbd-root>/<split>/images.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input image directory or a single pre-disaster image. Overrides --split.",
    )
    parser.add_argument(
        "--xbd-root",
        type=Path,
        default=DEFAULT_XBD_ROOT,
        help="xBD dataset root used together with --split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/xbd_<split|custom>_<mode>.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base output root used when --output-dir is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device. Use auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help="Text prompt used in text-only baseline mode.",
    )
    parser.add_argument(
        "--coarse-mask-dir",
        type=Path,
        default=None,
        help="Directory containing external coarse masks for refine mode.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="text",
        choices=SUPPORTED_PROMPT_MODES,
        help="Inference prompt mode. Use text for baseline, or box/point/box_point for refine.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Minimum predicted connected-component area to keep.",
    )
    parser.add_argument(
        "--prompt-min-area",
        type=int,
        default=DEFAULT_PROMPT_MIN_AREA,
        help="Minimum coarse connected-component area used to generate prompts.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N sorted inputs before running inference.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N inputs. Use 0 for all discovered inputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Used only for filesystem scanning / IO preparation, not model parallelism.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs for each selected image before re-running it.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose requested output files already exist.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save RGB visualization overlays under output_dir/visualizations.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save per-image instance metadata JSON under output_dir/instances.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save generated prompt metadata JSON under output_dir/prompts.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path. Defaults to output_dir/logs/run.log.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_parser(add_help: bool = True) -> ArgumentParser:
    """Build the batch inference parser."""
    parser = ArgumentParser(
        add_help=add_help,
        description="Run SAM3 building-prior inference on xBD pre-disaster images.",
    )
    return add_run_arguments(parser)


def parse_args(argv=None) -> Namespace:
    """Parse CLI arguments for the batch inference entrypoint."""
    return build_parser().parse_args(argv)


def default_vscode_argv() -> list[str]:
    """Return a safe default preset for bare VSCode file runs."""
    return [
        "--split",
        "train",
        "--device",
        "cuda",
        "--prompt-mode",
        "text",
        "--skip-existing",
        "--save-json",
        "--save-vis",
        "--save-prompts",
    ]


def prepare_run_args(args: Namespace) -> Namespace:
    """Resolve derived runtime paths and validate argument combinations."""
    if getattr(args, "_sam3_run_prepared", False):
        return args
    if args.input_dir is None and args.split is None:
        raise ValueError("Either --input-dir or --split must be provided")
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite cannot be used together")
    if args.start_index < 0:
        raise ValueError("start_index must be >= 0")
    if args.limit < 0:
        raise ValueError("limit must be >= 0")
    if args.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if args.prompt_mode != "text" and args.coarse_mask_dir is None:
        raise ValueError(f"Prompt mode '{args.prompt_mode}' requires --coarse-mask-dir")

    args.xbd_root = Path(args.xbd_root).resolve()
    args.output_root = Path(args.output_root).resolve()
    args.checkpoint = Path(args.checkpoint).resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    args.input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir is not None
        else resolve_split_input_dir(args.split, args.xbd_root).resolve()
    )
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_dir}")
    if args.output_dir is None:
        args.output_dir = build_default_output_dir(
            split=args.split,
            prompt_mode=args.prompt_mode,
            output_root=args.output_root,
        )
    args.output_dir = Path(args.output_dir).resolve()
    if args.log_file is None:
        args.log_file = resolve_default_log_file(args.output_dir)
    else:
        args.log_file = Path(args.log_file).resolve()
    if args.coarse_mask_dir is not None:
        args.coarse_mask_dir = Path(args.coarse_mask_dir).resolve()
        if not args.coarse_mask_dir.exists():
            raise FileNotFoundError(
                f"Coarse mask directory does not exist: {args.coarse_mask_dir}"
            )
    args._sam3_run_prepared = True
    return args


def _save_instance_json(path: Path, instances) -> None:
    """Write per-image instance predictions as JSON."""
    payload = []
    for inst in instances:
        payload.append(
            {
                "instance_id": inst.instance_id,
                "bbox_xyxy": inst.bbox_xyxy,
                "area": inst.area,
                "score": inst.score,
                "prompt_type": inst.prompt_type,
                "prompt_id": inst.prompt_id,
                "prompt_bbox_xyxy": inst.prompt_bbox_xyxy,
                "prompt_point_xy": inst.prompt_point_xy,
                "coarse_area": inst.coarse_area,
            }
        )
    write_json(path, payload)


def _build_record_output_paths(record, output_dirs: dict[str, Path]) -> dict[str, Path]:
    """Resolve all standard output file paths for a single image record."""
    return {
        "mask": output_dirs["masks"] / record.path.name,
        "instance_json": output_dirs["instances"] / f"{record.stem}.json",
        "visualization": output_dirs["visualizations"] / record.path.name,
        "prompt_json": output_dirs["prompts"] / f"{record.stem}.json",
    }


def _can_skip_record(paths: dict[str, Path], args: Namespace) -> bool:
    """Return True when all requested outputs for a record already exist."""
    required_paths = [paths["mask"]]
    if args.save_json:
        required_paths.append(paths["instance_json"])
    if args.save_vis:
        required_paths.append(paths["visualization"])
    if args.save_prompts:
        required_paths.append(paths["prompt_json"])
    return all(path.exists() for path in required_paths)


def _clear_record_outputs(paths: dict[str, Path]) -> None:
    """Delete stale per-record outputs before an overwrite run."""
    for path in paths.values():
        if path.exists():
            path.unlink()


def _build_text_prompts_payload(record, args: Namespace) -> dict:
    """Build prompt metadata payload for text-only runs."""
    return prompt_regions_to_json(
        image_path=record.path,
        prompt_mode=args.prompt_mode,
        text_prompt=args.text_prompt,
        coarse_mask_path=None,
        prompt_min_area=args.prompt_min_area,
        prompt_regions=[],
    )


def _run_text_baseline(adapter: SAM3Adapter, image, args: Namespace):
    """Run SAM3 with a text prompt and post-process predicted masks."""
    masks, scores, _ = adapter.predict_with_text(image, args.text_prompt)
    return extract_instances_from_masks(
        masks=np.asarray(masks),
        min_area=args.min_area,
        prompt_type="text",
        scores=np.asarray(scores),
    )


def _run_prompt_refine(adapter: SAM3Adapter, image, record, args: Namespace):
    """Run SAM3 refinement using prompts extracted from a coarse mask."""
    coarse_mask_path = resolve_mask_path(record.path, args.coarse_mask_dir)
    if coarse_mask_path is None:
        raise FileNotFoundError(
            f"Missing coarse mask for {record.path.name} under {args.coarse_mask_dir}"
        )

    coarse_mask = read_binary_mask(coarse_mask_path)
    prompt_regions = extract_prompt_regions(coarse_mask, min_area=args.prompt_min_area)
    prompts_payload = prompt_regions_to_json(
        image_path=record.path,
        prompt_mode=args.prompt_mode,
        text_prompt=args.text_prompt,
        coarse_mask_path=coarse_mask_path,
        prompt_min_area=args.prompt_min_area,
        prompt_regions=prompt_regions,
    )
    refined_results = adapter.refine_with_prompts(image, prompt_regions, args.prompt_mode)

    instances = []
    next_instance_id = 1
    for result in refined_results:
        prompt = result["prompt"]
        instance = extract_largest_instance_from_mask(
            mask=result["mask"],
            instance_id=next_instance_id,
            min_area=args.min_area,
            score=result["score"],
            prompt_type=result["prompt_type"],
            prompt_id=prompt.prompt_id,
            prompt_bbox_xyxy=prompt.bbox_xyxy,
            prompt_point_xy=prompt.center_point_xy,
            coarse_area=prompt.area,
        )
        if instance is None:
            continue
        instances.append(instance)
        next_instance_id += 1

    return instances, prompts_payload


def run_pipeline(args: Namespace) -> dict:
    """Run batch inference and return a structured summary dictionary."""
    args = prepare_run_args(args)
    output_dirs = build_output_dirs(args.output_dir)

    records = make_image_records(
        input_dir=args.input_dir,
        start_index=args.start_index,
        limit=args.limit,
        num_workers=args.num_workers,
    )
    if not records:
        raise RuntimeError(f"No pre-disaster images found under: {args.input_dir}")

    device = (
        "cuda"
        if args.device == "auto" and __import__("torch").cuda.is_available()
        else args.device
    )
    start_time = time.time()
    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "device": device,
        "mode": "text" if args.prompt_mode == "text" else "refine",
        "prompt_mode": args.prompt_mode,
        "text_prompt": args.text_prompt,
        "coarse_mask_dir": str(args.coarse_mask_dir) if args.coarse_mask_dir else None,
        "start_index": args.start_index,
        "limit": args.limit,
        "num_workers": args.num_workers,
        "save_vis": args.save_vis,
        "save_json": args.save_json,
        "save_prompts": args.save_prompts,
        "overwrite": args.overwrite,
        "skip_existing": args.skip_existing,
        "num_records": len(records),
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "successful_files": [],
        "failed_files": [],
        "skipped_files": [],
    }

    pending_records = []
    for record in records:
        output_paths = _build_record_output_paths(record, output_dirs)
        if args.skip_existing and _can_skip_record(output_paths, args):
            summary["skipped_count"] += 1
            summary["skipped_files"].append(record.path.name)
            continue
        pending_records.append((record, output_paths))

    if args.prompt_mode == "text" and args.coarse_mask_dir is not None:
        logging.info(
            "Ignoring coarse mask directory for text baseline: %s",
            args.coarse_mask_dir,
        )

    adapter = None
    if pending_records:
        adapter = SAM3Adapter(checkpoint=str(args.checkpoint), device=device)
        adapter.load()
    else:
        logging.info("No pending images to process after applying skip rules.")

    total_pending = len(pending_records)
    for index, (record, output_paths) in enumerate(pending_records, start=1):
        try:
            if args.overwrite:
                _clear_record_outputs(output_paths)
            if index == 1 or index == total_pending or index % 25 == 0:
                logging.info(
                    "Processing %d/%d: %s",
                    index,
                    total_pending,
                    record.path.name,
                )
            image = read_image(record.path)
            width, height = image.size
            if args.prompt_mode == "text":
                instances = _run_text_baseline(adapter, image, args)
                prompts_payload = _build_text_prompts_payload(record, args)
            else:
                instances, prompts_payload = _run_prompt_refine(adapter, image, record, args)

            merged_mask = merge_instances_to_mask(instances, height, width)

            save_binary_mask(output_paths["mask"], merged_mask)
            if args.save_json:
                _save_instance_json(output_paths["instance_json"], instances)
            if args.save_vis:
                overlay_mask_on_image(image, merged_mask).save(output_paths["visualization"])
            if args.save_prompts:
                write_json(output_paths["prompt_json"], prompts_payload)

            summary["success_count"] += 1
            summary["successful_files"].append(record.path.name)
        except Exception as exc:
            logging.exception("Failed processing %s: %s", record.path, exc)
            summary["failed_count"] += 1
            summary["failed_files"].append(
                {
                    "image": record.path.name,
                    "error": str(exc),
                }
            )

    summary["elapsed_seconds"] = round(time.time() - start_time, 2)
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(resolve_run_summary_path(output_dirs["root"]), summary)
    return summary


def run(args: Namespace) -> int:
    """Run batch inference and return a CLI-friendly exit code."""
    try:
        summary = run_pipeline(args)
    except Exception as exc:
        logging.exception("Batch inference failed before processing completed: %s", exc)
        return 1

    logging.info(
        "Total: %d | success: %d | failed: %d | skipped: %d | elapsed: %.1fs",
        summary["num_records"],
        summary["success_count"],
        summary["failed_count"],
        summary["skipped_count"],
        summary["elapsed_seconds"],
    )
    return 0 if summary["failed_count"] == 0 else 1


def main(argv=None) -> int:
    """CLI main entrypoint."""
    if argv is None and len(sys.argv) == 1:
        argv = default_vscode_argv()
        print(
            "No CLI arguments provided; using VSCode default preset: "
            "--split train --prompt-mode text --skip-existing"
        )
    try:
        args = prepare_run_args(parse_args(argv))
    except Exception as exc:
        print(f"Argument error: {exc}")
        return 1
    configure_logging(args.log_file, verbose=args.verbose)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
