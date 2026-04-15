"""Inference and evaluation utilities for xBD building segmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import XBDPreDisasterDataset, scan_xbd_split, split_train_val_samples
from .export import (
    save_mask_png,
    save_prediction_label_comparison,
    save_prediction_overlay,
    write_metrics_json,
    write_per_image_metrics_csv,
)
from .metrics import (
    build_metric_reports,
    confusion_counts_from_masks,
    enrich_record_with_foreground_stats,
    metrics_from_counts,
)
from .model_adapter import (
    SAM3BinarySegmentationModel,
    apply_experiment_checkpoint_config,
    inspect_checkpoint,
    load_experiment_state,
    normalize_model_output,
)
from .paths import resolve_checkpoint_path
from .postprocess import (
    binary_mask_to_uint8,
    logits_to_probabilities,
    postprocess_binary_mask,
    probabilities_to_binary_mask,
    resize_probabilities,
)
from .utils import ensure_dir, resolve_device, set_seed, setup_logger


@dataclass
class EvaluationArtifacts:
    output_dir: Path
    metrics_json_path: Path
    metrics_breakdown_path: Path
    per_image_csv_path: Path
    eval_config_used_path: Path
    postprocess_config_used_path: Path
    pred_masks_dir: Optional[Path]
    comparison_vis_dir: Optional[Path]
    overlay_vis_dir: Optional[Path]
    checkpoint_path: Optional[Path]
    threshold_sweep_path: Optional[Path]
    tta_used_path: Optional[Path]
    num_images: int
    used_threshold: float
    eval_split: str


@dataclass
class CalibrationArtifacts:
    output_dir: Path
    calibration_json_path: Path
    selected_threshold_path: Path
    selected_postprocess_path: Path
    checkpoint_path: Optional[Path]
    best_threshold: float
    best_min_component_area: int
    best_max_hole_area: int


def build_eval_dataloader(
    config: ExperimentConfig,
    *,
    eval_split: Optional[str] = None,
) -> tuple[DataLoader, dict[str, Any]]:
    split_name = (eval_split or config.eval.eval_split).lower()
    if split_name == "test":
        dataset = XBDPreDisasterDataset(
            config.paths.xbd_root,
            config.data.test_split,
            image_size=config.data.image_size,
            use_list_files=config.data.use_list_files,
            enable_augment=False,
        )
        split_info = {
            "eval_split": "test",
            "source_split": config.data.test_split,
            "dataset_size": len(dataset),
        }
    elif split_name == "val":
        all_train_samples = scan_xbd_split(
            config.paths.xbd_root,
            config.data.train_split,
            use_list_file=config.data.use_list_files,
        )
        _, val_samples = split_train_val_samples(
            all_train_samples,
            val_ratio=config.train.val_ratio,
            split_seed=config.train.split_seed,
        )
        dataset = XBDPreDisasterDataset(
            config.paths.xbd_root,
            config.data.train_split,
            image_size=config.data.image_size,
            use_list_files=config.data.use_list_files,
            samples=val_samples,
            enable_augment=False,
        )
        if len(dataset) == 0:
            raise ValueError(
                "Validation subset is empty. Check train.val_ratio and the available xBD train samples."
            )
        split_info = {
            "eval_split": "val",
            "source_split": config.data.train_split,
            "dataset_size": len(dataset),
            "val_ratio": config.train.val_ratio,
            "split_seed": config.train.split_seed,
        }
    else:
        raise ValueError(f"Unsupported eval split: {eval_split}")

    loader = DataLoader(
        dataset,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, split_info


def _apply_tta_transform(tensor: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "identity":
        return tensor
    if transform_name == "hflip":
        return torch.flip(tensor, dims=(-1,))
    if transform_name == "vflip":
        return torch.flip(tensor, dims=(-2,))
    if transform_name == "rot90":
        return torch.rot90(tensor, k=1, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA transform: {transform_name}")


def _invert_tta_transform(tensor: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "identity":
        return tensor
    if transform_name == "hflip":
        return torch.flip(tensor, dims=(-1,))
    if transform_name == "vflip":
        return torch.flip(tensor, dims=(-2,))
    if transform_name == "rot90":
        return torch.rot90(tensor, k=3, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA transform: {transform_name}")


def get_tta_transforms(config: ExperimentConfig) -> list[str]:
    transforms = ["identity"]
    if not config.eval.enable_tta:
        return transforms
    if config.eval.tta_hflip:
        transforms.append("hflip")
    if config.eval.tta_vflip:
        transforms.append("vflip")
    if config.eval.tta_rot90:
        transforms.append("rot90")
    return transforms


def predict_probabilities(
    model: SAM3BinarySegmentationModel,
    images: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Keep probability maps until the final threshold/postprocess step."""
    tta_transforms = get_tta_transforms(config)
    probability_sum: Optional[torch.Tensor] = None

    for transform_name in tta_transforms:
        augmented_images = _apply_tta_transform(images, transform_name)
        model_output = normalize_model_output(model(augmented_images))
        probabilities = logits_to_probabilities(model_output.mask_logits)
        probabilities = _invert_tta_transform(probabilities, transform_name)
        if probability_sum is None:
            probability_sum = probabilities
        else:
            probability_sum = probability_sum + probabilities

    if probability_sum is None:
        raise RuntimeError("TTA probability accumulation unexpectedly produced no outputs.")
    return probability_sum / float(len(tta_transforms))


def _mask_from_probabilities(
    probabilities: torch.Tensor,
    *,
    threshold: float,
    enable_postprocess: bool,
    min_component_area: int,
    max_hole_area: int,
) -> torch.Tensor:
    pred_mask = probabilities_to_binary_mask(probabilities, threshold=threshold)
    if pred_mask.ndim == 4:
        pred_mask = pred_mask[0]
    return postprocess_binary_mask(
        pred_mask,
        enable_postprocess=enable_postprocess,
        min_component_area=min_component_area,
        max_hole_area=max_hole_area,
    )


def _build_per_image_record(
    *,
    image_id: str,
    image_path: str,
    target_path: str,
    pred_mask_path: str,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_foreground_ratio: float,
    comparison_vis_path: str = "",
    overlay_vis_path: str = "",
) -> dict[str, float | int | str]:
    counts = confusion_counts_from_masks(pred_mask.cpu(), gt_mask.cpu())
    record = {
        **metrics_from_counts(counts),
        "image_id": image_id,
        "gt_foreground_ratio": float(gt_foreground_ratio),
        "pred_foreground_ratio": float(pred_mask.float().mean().item()),
        "image_path": image_path,
        "target_path": target_path,
        "pred_mask_path": pred_mask_path,
        "comparison_vis_path": comparison_vis_path,
        "overlay_vis_path": overlay_vis_path,
    }
    return enrich_record_with_foreground_stats(record)


def _load_threshold_from_json(threshold_json_path: Path) -> float:
    with threshold_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "best_threshold" not in payload:
        raise ValueError(f"Threshold JSON missing best_threshold: {threshold_json_path}")
    return float(payload["best_threshold"])


def _load_postprocess_from_json(postprocess_json_path: Path) -> dict[str, Any]:
    with postprocess_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "min_component_area" not in payload or "max_hole_area" not in payload:
        raise ValueError(
            "Postprocess JSON must contain min_component_area and max_hole_area: "
            f"{postprocess_json_path}"
        )
    payload.setdefault("enable_postprocess", True)
    return payload


def _prepare_model_for_evaluation(
    config: ExperimentConfig,
    *,
    checkpoint_path: Optional[str | Path],
    device: torch.device,
    logger,
) -> tuple[SAM3BinarySegmentationModel, Optional[Path], Optional[Path], str]:
    requested_checkpoint = checkpoint_path if checkpoint_path is not None else config.paths.checkpoint
    resolved_checkpoint = resolve_checkpoint_path(requested_checkpoint, must_exist=False)
    if requested_checkpoint is not None and resolved_checkpoint is not None and not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

    experiment_checkpoint_payload = None
    checkpoint_kind = "none"
    backbone_checkpoint = resolve_checkpoint_path(config.paths.checkpoint, must_exist=False)
    if resolved_checkpoint is not None and resolved_checkpoint.exists():
        inspection = inspect_checkpoint(resolved_checkpoint)
        checkpoint_kind = inspection.checkpoint_type
        if inspection.checkpoint_type == "experiment":
            experiment_checkpoint_payload = inspection.payload
            restore_hints = apply_experiment_checkpoint_config(config, experiment_checkpoint_payload)
            if restore_hints.backbone_checkpoint is not None:
                backbone_checkpoint = restore_hints.backbone_checkpoint
            if (
                restore_hints.backbone_checkpoint is None
                and backbone_checkpoint is not None
                and backbone_checkpoint == resolved_checkpoint
            ):
                backbone_checkpoint = None
        else:
            backbone_checkpoint = resolved_checkpoint

    model = SAM3BinarySegmentationModel(
        sam3_repo=config.paths.sam3_repo,
        backbone_checkpoint=backbone_checkpoint,
        decoder_channels=config.model.decoder_channels,
        freeze_backbone=config.model.freeze_backbone,
        device=device,
        use_amp=config.model.use_amp,
        amp_dtype=config.model.amp_dtype,
        decoder_dropout=config.model.decoder_dropout,
        use_resaspp=config.model.use_resaspp,
        use_attention_fusion=config.model.use_attention_fusion,
        use_presence_head=config.model.use_presence_head,
        use_boundary_head=config.model.use_boundary_head,
    )
    if experiment_checkpoint_payload is not None:
        load_experiment_state(model, experiment_checkpoint_payload)
        logger.info("Loaded experiment checkpoint: %s", resolved_checkpoint)
    else:
        logger.warning(
            "Evaluating with a backbone-only checkpoint. The local decoder is randomly initialized, "
            "so the resulting segmentation metrics are usually not meaningful for comparison."
        )
        logger.info("Using backbone checkpoint only: %s", backbone_checkpoint)

    model.eval()
    return model, resolved_checkpoint, backbone_checkpoint, checkpoint_kind


def run_threshold_sweep(
    model: SAM3BinarySegmentationModel,
    config: ExperimentConfig,
    *,
    device: torch.device,
    logger,
) -> dict[str, Any]:
    """Sweep a small threshold set on the reproducible validation subset."""
    val_loader, split_info = build_eval_dataloader(config, eval_split="val")
    threshold_candidates = [float(value) for value in config.eval.threshold_candidates]
    records_by_threshold: dict[float, list[dict[str, float | int | str]]] = {
        threshold: [] for threshold in threshold_candidates
    }

    logger.info(
        "Running threshold sweep on val split: %s | candidates=%s",
        split_info,
        threshold_candidates,
    )
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            probabilities = predict_probabilities(model, images, config)

            for index in range(images.shape[0]):
                gt_mask = batch["mask_original"][index].to(dtype=torch.uint8).cpu()
                original_height = int(batch["original_height"][index])
                original_width = int(batch["original_width"][index])
                resized_probabilities = resize_probabilities(
                    probabilities[index : index + 1],
                    output_shape=(original_height, original_width),
                )
                for threshold in threshold_candidates:
                    pred_mask = _mask_from_probabilities(
                        resized_probabilities,
                        threshold=threshold,
                        enable_postprocess=config.eval.enable_postprocess,
                        min_component_area=config.eval.min_component_area,
                        max_hole_area=config.eval.max_hole_area,
                    )
                    records_by_threshold[threshold].append(
                        _build_per_image_record(
                            image_id=batch["image_id"][index],
                            image_path=batch["image_path"][index],
                            target_path=batch["target_path"][index],
                            pred_mask_path="",
                            pred_mask=pred_mask.cpu(),
                            gt_mask=gt_mask,
                            gt_foreground_ratio=float(batch["foreground_ratio"][index]),
                        )
                    )

    best_threshold = threshold_candidates[0]
    best_iou = float("-inf")
    best_f1 = float("-inf")
    reports: dict[str, Any] = {}
    for threshold in threshold_candidates:
        summary, breakdown = build_metric_reports(records_by_threshold[threshold])
        threshold_key = f"{threshold:.2f}"
        reports[threshold_key] = {
            "summary": summary,
            "breakdown": breakdown,
        }
        current_iou = float(summary["overall"]["iou"])
        current_f1 = float(summary["overall"]["f1"])
        if current_iou > best_iou or (current_iou == best_iou and current_f1 > best_f1):
            best_threshold = threshold
            best_iou = current_iou
            best_f1 = current_f1

    return {
        "selection_split": split_info,
        "candidates": threshold_candidates,
        "best_threshold": best_threshold,
        "best_overall_iou": best_iou,
        "best_overall_f1": best_f1,
        "results": reports,
    }


def _is_better_calibration_candidate(
    candidate: dict[str, float | int],
    best: Optional[dict[str, float | int]],
) -> bool:
    if best is None:
        return True

    candidate_iou = float(candidate["overall_iou"])
    best_iou = float(best["overall_iou"])
    if candidate_iou != best_iou:
        return candidate_iou > best_iou

    candidate_f1 = float(candidate["overall_f1"])
    best_f1 = float(best["overall_f1"])
    if candidate_f1 != best_f1:
        return candidate_f1 > best_f1

    candidate_empty_fp = float(candidate["empty_image_false_positive_rate"])
    best_empty_fp = float(best["empty_image_false_positive_rate"])
    if candidate_empty_fp != best_empty_fp:
        return candidate_empty_fp < best_empty_fp

    candidate_small_iou = float(candidate["small_bucket_iou"])
    best_small_iou = float(best["small_bucket_iou"])
    return candidate_small_iou > best_small_iou


def calibrate_validation_postprocess(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    checkpoint_path: Optional[str | Path],
    threshold_candidates: Optional[list[float]] = None,
    min_component_area_candidates: Optional[list[int]] = None,
    max_hole_area_candidates: Optional[list[int]] = None,
) -> CalibrationArtifacts:
    """Search threshold + postprocess settings on the reproducible validation subset."""
    output_dir = ensure_dir(output_dir)
    logs_dir = ensure_dir(output_dir / "logs")
    logger = setup_logger("sam_compare.calibrate", logs_dir / "val_calibration.log")

    set_seed(config.system.seed)
    device = resolve_device(config.system.device)

    thresholds = threshold_candidates or [0.50, 0.55, 0.60, 0.65, 0.70]
    min_areas = min_component_area_candidates or [16, 32, 48, 64]
    max_holes = max_hole_area_candidates or [8, 16, 32]

    model, resolved_checkpoint, _, checkpoint_kind = _prepare_model_for_evaluation(
        config,
        checkpoint_path=checkpoint_path,
        device=device,
        logger=logger,
    )
    val_loader, split_info = build_eval_dataloader(config, eval_split="val")

    records_by_candidate: dict[tuple[float, int, int], list[dict[str, float | int | str]]] = {
        (float(threshold), int(min_area), int(max_hole_area)): []
        for threshold in thresholds
        for min_area in min_areas
        for max_hole_area in max_holes
    }

    logger.info(
        "Running val calibration on split=%s | checkpoint_kind=%s | thresholds=%s | min_component_areas=%s | max_hole_areas=%s",
        split_info,
        checkpoint_kind,
        thresholds,
        min_areas,
        max_holes,
    )

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            probabilities = predict_probabilities(model, images, config)

            for index in range(images.shape[0]):
                gt_mask = batch["mask_original"][index].to(dtype=torch.uint8).cpu()
                original_height = int(batch["original_height"][index])
                original_width = int(batch["original_width"][index])
                resized_probabilities = resize_probabilities(
                    probabilities[index : index + 1],
                    output_shape=(original_height, original_width),
                )
                threshold_masks = {
                    float(threshold): probabilities_to_binary_mask(
                        resized_probabilities,
                        threshold=float(threshold),
                    )[0]
                    for threshold in thresholds
                }

                for threshold, threshold_mask in threshold_masks.items():
                    for min_area in min_areas:
                        for max_hole_area in max_holes:
                            candidate_key = (float(threshold), int(min_area), int(max_hole_area))
                            pred_mask = postprocess_binary_mask(
                                threshold_mask,
                                enable_postprocess=True,
                                min_component_area=int(min_area),
                                max_hole_area=int(max_hole_area),
                            )
                            records_by_candidate[candidate_key].append(
                                _build_per_image_record(
                                    image_id=batch["image_id"][index],
                                    image_path=batch["image_path"][index],
                                    target_path=batch["target_path"][index],
                                    pred_mask_path="",
                                    pred_mask=pred_mask.cpu(),
                                    gt_mask=gt_mask,
                                    gt_foreground_ratio=float(batch["foreground_ratio"][index]),
                                )
                            )

    best_candidate: Optional[dict[str, float | int]] = None
    candidate_reports: list[dict[str, Any]] = []
    for threshold in thresholds:
        for min_area in min_areas:
            for max_hole_area in max_holes:
                candidate_key = (float(threshold), int(min_area), int(max_hole_area))
                summary, breakdown = build_metric_reports(records_by_candidate[candidate_key])
                candidate_report = {
                    "threshold": float(threshold),
                    "min_component_area": int(min_area),
                    "max_hole_area": int(max_hole_area),
                    "overall_iou": float(summary["overall"]["iou"]),
                    "overall_f1": float(summary["overall"]["f1"]),
                    "empty_image_false_positive_rate": float(
                        breakdown["empty_image_false_positive_rate"]
                    ),
                    "small_bucket_iou": float(
                        breakdown["size_buckets"].get("small", {}).get("iou", 0.0)
                    ),
                    "summary": summary,
                    "breakdown": breakdown,
                }
                candidate_reports.append(candidate_report)
                if _is_better_calibration_candidate(candidate_report, best_candidate):
                    best_candidate = candidate_report

    if best_candidate is None:
        raise RuntimeError("Validation calibration produced no candidate results.")

    calibration_payload = {
        "selection_split": split_info,
        "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint is not None else None,
        "checkpoint_kind": checkpoint_kind,
        "candidates": {
            "thresholds": [float(value) for value in thresholds],
            "min_component_areas": [int(value) for value in min_areas],
            "max_hole_areas": [int(value) for value in max_holes],
        },
        "selection_priority": [
            "overall_iou",
            "overall_f1",
            "empty_image_false_positive_rate_lower_is_better",
            "small_bucket_iou",
        ],
        "best_candidate": best_candidate,
        "results": candidate_reports,
    }
    calibration_json_path = write_metrics_json(calibration_payload, output_dir / "val_calibration.json")
    selected_threshold_path = write_metrics_json(
        {
            "selection_split": split_info,
            "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint is not None else None,
            "best_threshold": float(best_candidate["threshold"]),
            "selection_metrics": {
                "overall_iou": float(best_candidate["overall_iou"]),
                "overall_f1": float(best_candidate["overall_f1"]),
                "empty_image_false_positive_rate": float(best_candidate["empty_image_false_positive_rate"]),
                "small_bucket_iou": float(best_candidate["small_bucket_iou"]),
            },
        },
        output_dir / "selected_threshold.json",
    )
    selected_postprocess_path = write_metrics_json(
        {
            "selection_split": split_info,
            "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint is not None else None,
            "enable_postprocess": True,
            "min_component_area": int(best_candidate["min_component_area"]),
            "max_hole_area": int(best_candidate["max_hole_area"]),
            "selection_metrics": {
                "overall_iou": float(best_candidate["overall_iou"]),
                "overall_f1": float(best_candidate["overall_f1"]),
                "empty_image_false_positive_rate": float(best_candidate["empty_image_false_positive_rate"]),
                "small_bucket_iou": float(best_candidate["small_bucket_iou"]),
            },
        },
        output_dir / "selected_postprocess.json",
    )

    logger.info(
        "Validation calibration selected threshold=%.2f | min_component_area=%d | max_hole_area=%d | overall_iou=%.6f",
        float(best_candidate["threshold"]),
        int(best_candidate["min_component_area"]),
        int(best_candidate["max_hole_area"]),
        float(best_candidate["overall_iou"]),
    )
    return CalibrationArtifacts(
        output_dir=Path(output_dir),
        calibration_json_path=calibration_json_path,
        selected_threshold_path=selected_threshold_path,
        selected_postprocess_path=selected_postprocess_path,
        checkpoint_path=resolved_checkpoint,
        best_threshold=float(best_candidate["threshold"]),
        best_min_component_area=int(best_candidate["min_component_area"]),
        best_max_hole_area=int(best_candidate["max_hole_area"]),
    )


def evaluate_model(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    checkpoint_path: Optional[str | Path],
) -> EvaluationArtifacts:
    """Run inference on xBD val/test split and export masks plus metrics."""
    output_dir = ensure_dir(output_dir)
    logs_dir = ensure_dir(output_dir / "logs")
    logger = setup_logger("sam_compare.eval", logs_dir / "eval.log")

    set_seed(config.system.seed)
    device = resolve_device(config.system.device)

    postprocess_source = "config"
    if config.eval.postprocess_json is not None:
        postprocess_payload = _load_postprocess_from_json(
            Path(config.eval.postprocess_json).expanduser().resolve()
        )
        config.eval.enable_postprocess = bool(postprocess_payload.get("enable_postprocess", True))
        config.eval.min_component_area = int(postprocess_payload["min_component_area"])
        config.eval.max_hole_area = int(postprocess_payload["max_hole_area"])
        postprocess_source = "postprocess_json"
        logger.info(
            "Loaded postprocess config from %s | min_component_area=%d | max_hole_area=%d",
            config.eval.postprocess_json,
            config.eval.min_component_area,
            config.eval.max_hole_area,
        )

    model, resolved_checkpoint, backbone_checkpoint, checkpoint_kind = _prepare_model_for_evaluation(
        config,
        checkpoint_path=checkpoint_path,
        device=device,
        logger=logger,
    )
    eval_loader, eval_split_info = build_eval_dataloader(config)

    pred_masks_dir = ensure_dir(output_dir / "pred_masks") if config.eval.save_pred_masks else None
    comparison_vis_dir = (
        ensure_dir(output_dir / "visualizations" / "comparison")
        if config.eval.save_visualizations
        else None
    )
    overlay_vis_dir = (
        ensure_dir(output_dir / "visualizations" / "pred_overlay")
        if config.eval.save_visualizations
        else None
    )
    threshold_sweep_path: Optional[Path] = None
    tta_used_path: Optional[Path] = None
    used_threshold = float(config.eval.threshold)
    threshold_source = "config"

    if config.eval.threshold_json is not None:
        used_threshold = _load_threshold_from_json(
            Path(config.eval.threshold_json).expanduser().resolve()
        )
        threshold_source = "threshold_json"
        logger.info("Loaded threshold %.3f from %s", used_threshold, config.eval.threshold_json)
    elif config.eval.enable_threshold_sweep:
        threshold_sweep = run_threshold_sweep(model, config, device=device, logger=logger)
        threshold_sweep_path = write_metrics_json(threshold_sweep, output_dir / "threshold_sweep.json")
        used_threshold = float(threshold_sweep["best_threshold"])
        threshold_source = "threshold_sweep"
        logger.info(
            "Threshold sweep selected %.3f on val split | best_overall_iou=%.6f",
            used_threshold,
            float(threshold_sweep["best_overall_iou"]),
        )

    if config.eval.enable_tta:
        tta_used_path = write_metrics_json(
            {
                "enabled": True,
                "transforms": get_tta_transforms(config),
            },
            output_dir / "tta_used.json",
        )

    per_image_records: list[dict[str, float | int | str]] = []

    logger.info("Evaluating on device: %s", device)
    logger.info("Eval split info: %s", eval_split_info)
    logger.info(
        "Evaluation threshold=%.3f (%s) | TTA=%s | postprocess=%s (%s)",
        used_threshold,
        threshold_source,
        config.eval.enable_tta,
        config.eval.enable_postprocess,
        postprocess_source,
    )

    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"].to(device, non_blocking=True)
            probabilities = predict_probabilities(model, images, config)

            for index in range(images.shape[0]):
                image_id = batch["image_id"][index]
                image_path = batch["image_path"][index]
                target_path = batch["target_path"][index]
                gt_mask = batch["mask_original"][index].to(dtype=torch.uint8).cpu()
                original_height = int(batch["original_height"][index])
                original_width = int(batch["original_width"][index])

                resized_probabilities = resize_probabilities(
                    probabilities[index : index + 1],
                    output_shape=(original_height, original_width),
                )
                pred_mask = _mask_from_probabilities(
                    resized_probabilities,
                    threshold=used_threshold,
                    enable_postprocess=config.eval.enable_postprocess,
                    min_component_area=config.eval.min_component_area,
                    max_hole_area=config.eval.max_hole_area,
                )

                pred_mask_path = ""
                comparison_vis_path = ""
                overlay_vis_path = ""
                pred_uint8 = binary_mask_to_uint8(pred_mask.squeeze(0))
                gt_uint8 = binary_mask_to_uint8(gt_mask.squeeze(0))
                if pred_masks_dir is not None:
                    pred_mask_path = str(save_mask_png(pred_uint8, pred_masks_dir / f"{image_id}.png"))
                if comparison_vis_dir is not None and overlay_vis_dir is not None:
                    comparison_vis_path = str(
                        save_prediction_label_comparison(
                            image_path,
                            pred_uint8,
                            gt_uint8,
                            comparison_vis_dir / f"{image_id}.png",
                        )
                    )
                    overlay_vis_path = str(
                        save_prediction_overlay(
                            image_path,
                            pred_uint8,
                            overlay_vis_dir / f"{image_id}.png",
                        )
                    )

                per_image_records.append(
                    _build_per_image_record(
                        image_id=image_id,
                        image_path=image_path,
                        target_path=target_path,
                        pred_mask_path=pred_mask_path,
                        comparison_vis_path=comparison_vis_path,
                        overlay_vis_path=overlay_vis_path,
                        pred_mask=pred_mask.cpu(),
                        gt_mask=gt_mask,
                        gt_foreground_ratio=float(batch["foreground_ratio"][index]),
                    )
                )

    summary, breakdown = build_metric_reports(per_image_records)
    summary["checkpoint_path"] = str(resolved_checkpoint) if resolved_checkpoint is not None else None
    summary["backbone_checkpoint"] = str(backbone_checkpoint) if backbone_checkpoint is not None else None
    summary["used_threshold"] = used_threshold
    summary["threshold_source"] = threshold_source
    summary["eval_split"] = eval_split_info["eval_split"]
    summary["checkpoint_kind"] = checkpoint_kind

    metrics_json_path = write_metrics_json(summary, output_dir / "metrics.json")
    metrics_breakdown_path = write_metrics_json(breakdown, output_dir / "metrics_breakdown.json")
    per_image_csv_path = write_per_image_metrics_csv(per_image_records, output_dir / "per_image_metrics.csv")
    eval_config_used_path = write_metrics_json(
        {
            "config": config.to_dict(),
            "resolved_checkpoint": str(resolved_checkpoint) if resolved_checkpoint else None,
            "resolved_backbone_checkpoint": str(backbone_checkpoint) if backbone_checkpoint else None,
            "checkpoint_kind": checkpoint_kind,
            "eval_split_info": eval_split_info,
            "used_threshold": used_threshold,
            "threshold_source": threshold_source,
            "postprocess_source": postprocess_source,
            "save_visualizations": config.eval.save_visualizations,
        },
        output_dir / "eval_config_used.json",
    )
    postprocess_config_used_path = write_metrics_json(
        {
            "enabled": config.eval.enable_postprocess,
            "source": postprocess_source,
            "min_component_area": config.eval.min_component_area,
            "max_hole_area": config.eval.max_hole_area,
        },
        output_dir / "postprocess_config_used.json",
    )

    logger.info("Evaluation complete. Overall IoU: %.6f", float(summary["overall"]["iou"]))
    logger.info("Metrics JSON: %s", metrics_json_path)
    logger.info("Metrics breakdown JSON: %s", metrics_breakdown_path)
    logger.info("Per-image CSV: %s", per_image_csv_path)
    if comparison_vis_dir is not None:
        logger.info("Comparison visualizations: %s", comparison_vis_dir)
    if overlay_vis_dir is not None:
        logger.info("Prediction overlay visualizations: %s", overlay_vis_dir)

    return EvaluationArtifacts(
        output_dir=Path(output_dir),
        metrics_json_path=metrics_json_path,
        metrics_breakdown_path=metrics_breakdown_path,
        per_image_csv_path=per_image_csv_path,
        eval_config_used_path=eval_config_used_path,
        postprocess_config_used_path=postprocess_config_used_path,
        pred_masks_dir=pred_masks_dir,
        comparison_vis_dir=comparison_vis_dir,
        overlay_vis_dir=overlay_vis_dir,
        checkpoint_path=resolved_checkpoint,
        threshold_sweep_path=threshold_sweep_path,
        tta_used_path=tta_used_path,
        num_images=len(per_image_records),
        used_threshold=used_threshold,
        eval_split=eval_split_info["eval_split"],
    )
