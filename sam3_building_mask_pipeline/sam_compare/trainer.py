"""Training loop for the standalone xBD vs. SAM3 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import ExperimentConfig
from .dataset import (
    XBDPreDisasterDataset,
    compute_sample_foreground_ratios,
    scan_xbd_split,
    split_train_val_samples,
)
from .metrics import (
    confusion_counts_from_masks,
    enrich_record_with_foreground_stats,
    metrics_from_counts,
    summarize_metric_records,
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
    logits_to_probabilities,
    postprocess_binary_mask,
    probabilities_to_binary_mask,
    resize_probabilities,
)
from .utils import (
    count_trainable_parameters,
    ensure_dir,
    resolve_device,
    set_seed,
    setup_logger,
    write_json,
)


@dataclass
class BestCheckpointState:
    best_epoch: int = -1
    best_val_iou: float = float("-inf")
    best_val_f1: float = float("-inf")
    best_val_loss: float = float("inf")


@dataclass
class EarlyStoppingState:
    best_metric: float = float("-inf")
    no_improve_epochs: int = 0
    stopped_early: bool = False
    stop_epoch: Optional[int] = None


@dataclass
class TrainingArtifacts:
    output_dir: Path
    best_model_path: Path
    last_model_path: Path
    checkpoints_dir: Path
    train_config_path: Path
    training_summary_path: Path
    backbone_checkpoint: Optional[Path]
    epochs_completed: int
    best_epoch: int
    best_val_iou: float


def build_boundary_target(
    targets: torch.Tensor,
    *,
    boundary_kernel_size: int,
) -> torch.Tensor:
    """Approximate a thin binary boundary band with max-pool dilation/erosion."""
    kernel_size = max(3, int(boundary_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    dilation = F.max_pool2d(
        targets,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    erosion = 1.0 - F.max_pool2d(
        1.0 - targets,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return (dilation - erosion).clamp(min=0.0, max=1.0)


def build_boundary_weight_map(
    targets: torch.Tensor,
    *,
    boundary_weight: float,
    boundary_kernel_size: int,
) -> torch.Tensor:
    """Highlight a thin GT boundary band so BCE pays extra attention to edges."""
    if boundary_weight <= 0.0:
        return torch.ones_like(targets)
    boundary_band = build_boundary_target(
        targets,
        boundary_kernel_size=boundary_kernel_size,
    )
    return 1.0 + boundary_weight * boundary_band


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float,
    boundary_weight_map: torch.Tensor,
) -> torch.Tensor:
    """Foreground-weighted BCE with an extra multiplicative boundary weight map."""
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=logits.new_tensor(pos_weight),
    )
    weighted_loss = pixel_loss * boundary_weight_map
    normalizer = boundary_weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return weighted_loss.sum(dim=(1, 2, 3)) / normalizer


def soft_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Tversky loss with heavier FN penalty for better recall."""
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss for sparse auxiliary boundary supervision."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any().item()):
        return values[mask].mean()
    return values.new_tensor(0.0)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _build_foreground_masks(
    foreground_ratio: torch.Tensor,
    *,
    small_fg_ratio_thr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty_mask = foreground_ratio <= 0.0
    small_fg_mask = torch.logical_and(
        foreground_ratio > 0.0,
        foreground_ratio < small_fg_ratio_thr,
    )
    non_empty_mask = torch.logical_not(empty_mask)
    return empty_mask, small_fg_mask, non_empty_mask


def _foreground_bucket_name(
    foreground_ratio_resized: float,
    *,
    small_fg_ratio_thr: float,
) -> str:
    # Reuse the project's empty/small/medium/large bucket semantics so weighted
    # sampling can bias train batches without changing the deterministic split.
    if foreground_ratio_resized <= 0.0:
        return "empty"
    if foreground_ratio_resized < small_fg_ratio_thr:
        return "small"
    if foreground_ratio_resized < 0.05:
        return "medium"
    return "large"


def _sampler_bucket_weights(config: ExperimentConfig) -> dict[str, float]:
    return {
        "empty": float(config.train.sample_weight_empty),
        "small": float(config.train.sample_weight_small),
        "medium": float(config.train.sample_weight_medium),
        "large": float(config.train.sample_weight_large),
    }


def _build_weighted_train_sampler(
    train_dataset: XBDPreDisasterDataset,
    config: ExperimentConfig,
) -> tuple[Optional[WeightedRandomSampler], dict[str, Any]]:
    bucket_counts = {bucket_name: 0 for bucket_name in ("empty", "small", "medium", "large")}
    bucket_weights = _sampler_bucket_weights(config)
    sample_weights: list[float] = []

    for sample in train_dataset.samples:
        _, foreground_ratio_resized = compute_sample_foreground_ratios(
            sample,
            image_size=config.data.image_size,
        )
        bucket_name = _foreground_bucket_name(
            foreground_ratio_resized,
            small_fg_ratio_thr=config.train.small_fg_ratio_thr,
        )
        bucket_counts[bucket_name] += 1
        sample_weights.append(bucket_weights[bucket_name])

    sampler_info = {
        "enabled": bool(config.train.enable_weighted_sampler),
        "bucket_counts": bucket_counts,
        "bucket_weights": bucket_weights,
        "num_samples": len(sample_weights),
        "replacement": True,
    }
    if not config.train.enable_weighted_sampler:
        return None, sampler_info

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(config.train.split_seed),
    )
    return sampler, sampler_info


def compute_training_loss(
    model_output: torch.Tensor | dict[str, torch.Tensor | None],
    targets: torch.Tensor,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Combine mask loss with boundary/presence auxiliary supervision."""
    outputs = normalize_model_output(model_output)
    mask_logits = outputs.mask_logits

    boundary_weight_map = build_boundary_weight_map(
        targets,
        boundary_weight=config.train.boundary_weight,
        boundary_kernel_size=config.train.boundary_kernel_size,
    )
    weighted_bce = weighted_bce_with_logits(
        mask_logits,
        targets,
        pos_weight=config.train.pos_weight,
        boundary_weight_map=boundary_weight_map,
    )
    tversky = soft_tversky_loss(
        mask_logits,
        targets,
        alpha=config.train.tversky_alpha,
        beta=config.train.tversky_beta,
    )
    foreground_ratio = targets.mean(dim=(1, 2, 3))
    empty_mask, small_fg_mask, non_empty_mask = _build_foreground_masks(
        foreground_ratio,
        small_fg_ratio_thr=config.train.small_fg_ratio_thr,
    )
    # Empty images get a distinct boost from tiny-foreground images so we can
    # reduce false positives without collapsing the current small-object focus.
    sample_boost = torch.where(
        empty_mask,
        mask_logits.new_full(foreground_ratio.shape, config.train.empty_fg_boost),
        torch.where(
            small_fg_mask,
            mask_logits.new_full(foreground_ratio.shape, config.train.small_fg_boost),
            mask_logits.new_ones(foreground_ratio.shape),
        ),
    )
    mask_loss_per_sample = (weighted_bce + tversky) * sample_boost
    mask_loss = mask_loss_per_sample.mean()

    boundary_loss = mask_logits.new_tensor(0.0)
    boundary_bce = mask_logits.new_tensor(0.0)
    boundary_dice = mask_logits.new_tensor(0.0)
    if outputs.boundary_logits is not None:
        boundary_targets = build_boundary_target(
            targets,
            boundary_kernel_size=config.train.boundary_kernel_size,
        )
        boundary_bce_per_sample = F.binary_cross_entropy_with_logits(
            outputs.boundary_logits,
            boundary_targets,
            reduction="none",
        ).mean(dim=(1, 2, 3))
        boundary_dice_per_sample = soft_dice_loss(
            outputs.boundary_logits,
            boundary_targets,
        )
        boundary_bce = boundary_bce_per_sample.mean()
        boundary_dice = boundary_dice_per_sample.mean()
        boundary_loss = boundary_bce + boundary_dice

    presence_loss = mask_logits.new_tensor(0.0)
    presence_acc = mask_logits.new_tensor(0.0)
    presence_acc_empty = mask_logits.new_tensor(0.0)
    presence_acc_non_empty = mask_logits.new_tensor(0.0)
    presence_correct_count = 0.0
    presence_empty_correct_count = 0.0
    presence_non_empty_correct_count = 0.0
    presence_count = 0
    presence_empty_count = 0
    presence_non_empty_count = 0
    if outputs.presence_logit is not None:
        presence_targets = (
            targets.sum(dim=(1, 2, 3)).clamp(min=0.0) > 0.0
        ).to(dtype=mask_logits.dtype)
        presence_logits = outputs.presence_logit.reshape(-1)
        # Keep the current head/interface intact, but weight BCE per image so
        # empty images and tiny-foreground images constrain false positives more.
        presence_weights = torch.where(
            empty_mask,
            presence_logits.new_full(presence_targets.shape, config.train.presence_empty_weight),
            torch.where(
                small_fg_mask,
                presence_logits.new_full(presence_targets.shape, config.train.presence_small_weight),
                presence_logits.new_ones(presence_targets.shape),
            ),
        )
        presence_loss_per_sample = F.binary_cross_entropy_with_logits(
            presence_logits,
            presence_targets,
            reduction="none",
        )
        presence_loss = (presence_loss_per_sample * presence_weights).mean()
        presence_predictions = torch.sigmoid(presence_logits) >= 0.5
        presence_correct = (
            presence_predictions == presence_targets.to(dtype=torch.bool)
        ).float()
        presence_acc = presence_correct.mean()
        presence_acc_empty = _safe_masked_mean(presence_correct, empty_mask)
        presence_acc_non_empty = _safe_masked_mean(presence_correct, non_empty_mask)
        presence_correct_count = float(presence_correct.sum().item())
        presence_empty_correct_count = float(presence_correct[empty_mask].sum().item())
        presence_non_empty_correct_count = float(
            presence_correct[non_empty_mask].sum().item()
        )
        presence_count = int(presence_correct.numel())
        presence_empty_count = int(empty_mask.sum().item())
        presence_non_empty_count = int(non_empty_mask.sum().item())

    total_loss = (
        mask_loss
        + config.train.boundary_aux_weight * boundary_loss
        + config.train.presence_aux_weight * presence_loss
    )
    return {
        "loss": total_loss,
        "mask_loss": mask_loss,
        "weighted_bce": weighted_bce.mean(),
        "tversky_loss": tversky.mean(),
        "boundary_loss": boundary_loss,
        "boundary_bce": boundary_bce,
        "boundary_dice_loss": boundary_dice,
        "presence_loss": presence_loss,
        "presence_acc": presence_acc,
        "presence_acc_empty": presence_acc_empty,
        "presence_acc_non_empty": presence_acc_non_empty,
        "foreground_ratio": foreground_ratio.mean(),
        "small_fg_fraction": small_fg_mask.float().mean(),
        "empty_fraction": empty_mask.float().mean(),
        "sample_boost_mean": sample_boost.mean(),
        "batch_size": int(targets.shape[0]),
        "empty_sample_count": int(empty_mask.sum().item()),
        "non_empty_sample_count": int(non_empty_mask.sum().item()),
        "presence_count": presence_count,
        "presence_correct_count": presence_correct_count,
        "presence_empty_count": presence_empty_count,
        "presence_empty_correct_count": presence_empty_correct_count,
        "presence_non_empty_count": presence_non_empty_count,
        "presence_non_empty_correct_count": presence_non_empty_correct_count,
    }


def build_train_val_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    all_train_samples = scan_xbd_split(
        config.paths.xbd_root,
        config.data.train_split,
        use_list_file=config.data.use_list_files,
    )
    train_samples, val_samples = split_train_val_samples(
        all_train_samples,
        val_ratio=config.train.val_ratio,
        split_seed=config.train.split_seed,
    )
    if not train_samples or not val_samples:
        raise ValueError(
            "The current training pipeline requires a non-empty train/val split. "
            f"Got train={len(train_samples)}, val={len(val_samples)}."
        )

    train_dataset = XBDPreDisasterDataset(
        config.paths.xbd_root,
        config.data.train_split,
        image_size=config.data.image_size,
        use_list_files=config.data.use_list_files,
        samples=train_samples,
        enable_augment=config.data.enable_train_aug,
        aug_hflip=config.data.aug_hflip,
        aug_vflip=config.data.aug_vflip,
        aug_rot90=config.data.aug_rot90,
        aug_brightness_contrast=config.data.aug_brightness_contrast,
    )
    val_dataset = XBDPreDisasterDataset(
        config.paths.xbd_root,
        config.data.train_split,
        image_size=config.data.image_size,
        use_list_files=config.data.use_list_files,
        samples=val_samples,
        enable_augment=False,
    )

    train_sampler, sampler_info = _build_weighted_train_sampler(train_dataset, config)
    generator = torch.Generator().manual_seed(config.system.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        num_workers=config.system.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    split_info = {
        "full_train_size": len(all_train_samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "train_image_ids": [sample.image_id for sample in train_samples],
        "val_image_ids": [sample.image_id for sample in val_samples],
        "train_sampler": sampler_info,
    }
    return train_loader, val_loader, split_info


def _build_optimizer(
    model: SAM3BinarySegmentationModel,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    backbone_parameters = list(model.backbone_parameters())
    decoder_parameters = list(model.decoder_parameters())
    param_groups = []

    if backbone_parameters:
        param_groups.append(
            {
                "params": backbone_parameters,
                "lr": config.train.lr * config.train.backbone_lr_scale,
                "weight_decay": config.train.weight_decay,
                "name": "backbone",
            }
        )
    if decoder_parameters:
        param_groups.append(
            {
                "params": decoder_parameters,
                "lr": config.train.lr,
                "weight_decay": config.train.weight_decay,
                "name": "decoder",
            }
        )
    return torch.optim.AdamW(param_groups, lr=config.train.lr, weight_decay=config.train.weight_decay)


def _format_optimizer_lrs(optimizer: torch.optim.Optimizer) -> str:
    formatted = []
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{index}")
        formatted.append(f"{group_name}={float(group['lr']):.6g}")
    return ", ".join(formatted)


def _get_current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    current_lrs: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{index}")
        current_lrs[group_name] = float(group["lr"])
    return current_lrs


def _scheduler_min_lrs(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> list[float]:
    min_lrs: list[float] = []
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{index}")
        if group_name == "backbone":
            min_lrs.append(float(config.train.scheduler_min_lr_backbone))
        elif group_name == "decoder":
            min_lrs.append(float(config.train.scheduler_min_lr_decoder))
        else:
            min_lrs.append(
                min(
                    float(config.train.scheduler_min_lr_backbone),
                    float(config.train.scheduler_min_lr_decoder),
                )
            )
    return min_lrs


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> Optional[ReduceLROnPlateau]:
    if not config.train.enable_scheduler:
        return None
    scheduler_type = config.train.scheduler_type.strip().lower()
    if scheduler_type != "plateau":
        raise ValueError(
            f"Unsupported scheduler_type={config.train.scheduler_type!r}. "
            "The current training pipeline supports only 'plateau'."
        )
    return ReduceLROnPlateau(
        optimizer,
        mode=config.train.scheduler_mode,
        factor=config.train.scheduler_factor,
        patience=config.train.scheduler_patience,
        threshold=config.train.scheduler_threshold,
        threshold_mode="abs",
        cooldown=0,
        min_lr=_scheduler_min_lrs(optimizer, config),
    )


def _scheduler_step_metric(
    config: ExperimentConfig,
    *,
    val_loss: float,
    val_iou: float,
    val_f1: float,
) -> float:
    monitor_name = config.train.scheduler_monitor.strip().lower()
    if monitor_name == "val_iou":
        return float(val_iou)
    if monitor_name == "val_f1":
        return float(val_f1)
    if monitor_name == "val_loss":
        return float(val_loss)
    raise ValueError(
        f"Unsupported scheduler_monitor={config.train.scheduler_monitor!r}. "
        "Expected one of: val_iou, val_f1, val_loss."
    )


def _init_epoch_stats() -> dict[str, float | int]:
    return {
        "steps": 0,
        "sample_count": 0,
        "loss_sum": 0.0,
        "weighted_bce_sum": 0.0,
        "tversky_sum": 0.0,
        "boundary_loss_sum": 0.0,
        "presence_loss_sum": 0.0,
        "foreground_ratio_sum": 0.0,
        "small_fg_fraction_sum": 0.0,
        "empty_fraction_sum": 0.0,
        "sample_boost_mean_sum": 0.0,
        "presence_correct_count": 0.0,
        "presence_count": 0,
        "presence_empty_correct_count": 0.0,
        "presence_empty_count": 0,
        "presence_non_empty_correct_count": 0.0,
        "presence_non_empty_count": 0,
    }


def _accumulate_epoch_stats(
    epoch_stats: dict[str, float | int],
    loss_terms: dict[str, Any],
) -> None:
    batch_size = int(loss_terms["batch_size"])
    epoch_stats["steps"] += 1
    epoch_stats["sample_count"] += batch_size
    epoch_stats["loss_sum"] += float(loss_terms["loss"].item())
    epoch_stats["weighted_bce_sum"] += float(loss_terms["weighted_bce"].item())
    epoch_stats["tversky_sum"] += float(loss_terms["tversky_loss"].item())
    epoch_stats["boundary_loss_sum"] += float(loss_terms["boundary_loss"].item())
    epoch_stats["presence_loss_sum"] += float(loss_terms["presence_loss"].item())
    epoch_stats["foreground_ratio_sum"] += float(loss_terms["foreground_ratio"].item()) * batch_size
    epoch_stats["small_fg_fraction_sum"] += float(loss_terms["small_fg_fraction"].item()) * batch_size
    epoch_stats["empty_fraction_sum"] += float(loss_terms["empty_fraction"].item()) * batch_size
    epoch_stats["sample_boost_mean_sum"] += float(loss_terms["sample_boost_mean"].item()) * batch_size
    epoch_stats["presence_correct_count"] += float(loss_terms["presence_correct_count"])
    epoch_stats["presence_count"] += int(loss_terms["presence_count"])
    epoch_stats["presence_empty_correct_count"] += float(
        loss_terms["presence_empty_correct_count"]
    )
    epoch_stats["presence_empty_count"] += int(loss_terms["presence_empty_count"])
    epoch_stats["presence_non_empty_correct_count"] += float(
        loss_terms["presence_non_empty_correct_count"]
    )
    epoch_stats["presence_non_empty_count"] += int(loss_terms["presence_non_empty_count"])


def _finalize_epoch_stats(epoch_stats: dict[str, float | int]) -> dict[str, float]:
    steps = int(epoch_stats["steps"])
    sample_count = int(epoch_stats["sample_count"])
    return {
        "loss": _safe_divide(float(epoch_stats["loss_sum"]), max(steps, 1)),
        "weighted_bce": _safe_divide(float(epoch_stats["weighted_bce_sum"]), max(steps, 1)),
        "tversky_loss": _safe_divide(float(epoch_stats["tversky_sum"]), max(steps, 1)),
        "boundary_loss": _safe_divide(float(epoch_stats["boundary_loss_sum"]), max(steps, 1)),
        "presence_loss": _safe_divide(float(epoch_stats["presence_loss_sum"]), max(steps, 1)),
        "presence_acc": _safe_divide(
            float(epoch_stats["presence_correct_count"]),
            int(epoch_stats["presence_count"]),
        ),
        "foreground_ratio": _safe_divide(
            float(epoch_stats["foreground_ratio_sum"]),
            sample_count,
        ),
        "small_fg_fraction": _safe_divide(
            float(epoch_stats["small_fg_fraction_sum"]),
            sample_count,
        ),
        "empty_fraction": _safe_divide(
            float(epoch_stats["empty_fraction_sum"]),
            sample_count,
        ),
        "sample_boost_mean": _safe_divide(
            float(epoch_stats["sample_boost_mean_sum"]),
            sample_count,
        ),
        "presence_acc_empty": _safe_divide(
            float(epoch_stats["presence_empty_correct_count"]),
            int(epoch_stats["presence_empty_count"]),
        ),
        "presence_acc_non_empty": _safe_divide(
            float(epoch_stats["presence_non_empty_correct_count"]),
            int(epoch_stats["presence_non_empty_count"]),
        ),
        "steps": float(steps),
        "sample_count": float(sample_count),
    }


def _update_early_stopping_state(
    early_stopping_state: EarlyStoppingState,
    *,
    candidate_metric: float,
    epoch: int,
    config: ExperimentConfig,
) -> bool:
    if not config.train.enable_early_stopping:
        return False

    current_epoch = epoch + 1
    start_epoch = max(1, int(config.train.early_stopping_start_epoch))
    if current_epoch < start_epoch:
        early_stopping_state.best_metric = max(
            early_stopping_state.best_metric,
            candidate_metric,
        )
        early_stopping_state.no_improve_epochs = 0
        return False

    if candidate_metric > early_stopping_state.best_metric + config.train.early_stopping_min_delta:
        early_stopping_state.best_metric = candidate_metric
        early_stopping_state.no_improve_epochs = 0
        return False

    early_stopping_state.no_improve_epochs += 1
    if early_stopping_state.no_improve_epochs >= config.train.early_stopping_patience:
        early_stopping_state.stopped_early = True
        early_stopping_state.stop_epoch = current_epoch
        return True
    return False


def _should_train_backbone(config: ExperimentConfig, epoch: int) -> bool:
    if config.model.freeze_backbone:
        return False
    return epoch >= max(0, int(config.train.stage1_freeze_backbone_epochs))


def _apply_backbone_training_stage(
    model: SAM3BinarySegmentationModel,
    config: ExperimentConfig,
    epoch: int,
) -> bool:
    train_backbone = _should_train_backbone(config, epoch)
    model.set_backbone_trainable(train_backbone)
    return train_backbone


def _save_checkpoint(
    model: SAM3BinarySegmentationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    config: ExperimentConfig,
    path: Path,
    *,
    epoch: int,
    best_state: BestCheckpointState,
    early_stopping_state: EarlyStoppingState,
    train_loss: float,
    val_loss: float,
    val_metrics: dict[str, float],
    backbone_checkpoint: Optional[Path],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_epoch": best_state.best_epoch,
        "best_val_iou": best_state.best_val_iou,
        "best_val_f1": best_state.best_val_f1,
        "best_val_loss": best_state.best_val_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "backbone_checkpoint": str(backbone_checkpoint) if backbone_checkpoint is not None else None,
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint["early_stopping_state"] = asdict(early_stopping_state)
    torch.save(checkpoint, path)


def _load_resume_checkpoint(
    model: SAM3BinarySegmentationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    resume_path: Path,
    *,
    logger,
) -> tuple[int, BestCheckpointState, EarlyStoppingState]:
    checkpoint = inspect_checkpoint(resume_path).payload
    load_experiment_state(model, checkpoint)
    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
        except ValueError as exc:
            logger.warning(
                "Resume checkpoint optimizer state is incompatible with the current param-group layout. "
                "Continuing with a freshly initialized optimizer. Details: %s",
                exc,
            )
    if scheduler is not None:
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict is not None:
            try:
                scheduler.load_state_dict(scheduler_state_dict)
            except Exception as exc:  # pragma: no cover - defensive compatibility path
                logger.warning(
                    "Resume checkpoint scheduler state could not be restored cleanly. "
                    "Continuing with a fresh scheduler state. Details: %s",
                    exc,
                )
        else:
            logger.info(
                "Resume checkpoint has no scheduler_state_dict. Continuing with a fresh scheduler state."
            )

    start_epoch = int(checkpoint["epoch"]) + 1
    best_state = BestCheckpointState(
        best_epoch=int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1))),
        best_val_iou=float(checkpoint.get("best_val_iou", float("-inf"))),
        best_val_f1=float(checkpoint.get("best_val_f1", float("-inf"))),
        best_val_loss=float(checkpoint.get("best_val_loss", float("inf"))),
    )
    early_stopping_payload = checkpoint.get("early_stopping_state")
    if isinstance(early_stopping_payload, dict):
        early_stopping_state = EarlyStoppingState(
            best_metric=float(
                early_stopping_payload.get("best_metric", best_state.best_val_iou)
            ),
            no_improve_epochs=int(early_stopping_payload.get("no_improve_epochs", 0)),
            stopped_early=bool(early_stopping_payload.get("stopped_early", False)),
            stop_epoch=(
                int(early_stopping_payload["stop_epoch"])
                if early_stopping_payload.get("stop_epoch") is not None
                else None
            ),
        )
    else:
        early_stopping_state = EarlyStoppingState(best_metric=best_state.best_val_iou)
        logger.info(
            "Resume checkpoint has no early_stopping_state. Continuing with reset early stopping counters."
        )
    return start_epoch, best_state, early_stopping_state


def _run_validation(
    model: SAM3BinarySegmentationModel,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    epoch_stats = _init_epoch_stats()
    per_image_records: list[dict[str, float | int | str]] = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            model_output = model(images)
            normalized_output = normalize_model_output(model_output)
            loss_terms = compute_training_loss(model_output, masks, config)
            _accumulate_epoch_stats(epoch_stats, loss_terms)

            probs = logits_to_probabilities(normalized_output.mask_logits)
            for index in range(images.shape[0]):
                image_id = batch["image_id"][index]
                gt_mask = batch["mask_original"][index].to(dtype=torch.uint8).cpu()
                original_height = int(batch["original_height"][index])
                original_width = int(batch["original_width"][index])

                resized_probs = resize_probabilities(
                    probs[index : index + 1],
                    output_shape=(original_height, original_width),
                )
                pred_mask = probabilities_to_binary_mask(
                    resized_probs,
                    threshold=config.eval.threshold,
                )[0]
                pred_mask = postprocess_binary_mask(
                    pred_mask,
                    enable_postprocess=config.eval.enable_postprocess,
                    min_component_area=config.eval.min_component_area,
                    max_hole_area=config.eval.max_hole_area,
                )
                pred_mask_cpu = pred_mask.cpu()
                counts_metrics = confusion_counts_from_masks(pred_mask_cpu, gt_mask)
                record = {
                    **counts_metrics.to_dict(),
                    "image_id": image_id,
                    "gt_foreground_ratio": float(batch["foreground_ratio"][index]),
                    "pred_foreground_ratio": float(pred_mask_cpu.float().mean().item()),
                }
                record.update(metrics_from_counts(counts_metrics))
                per_image_records.append(enrich_record_with_foreground_stats(record))

    summary = summarize_metric_records(per_image_records)
    return {**_finalize_epoch_stats(epoch_stats), "summary": summary}


def _is_better_checkpoint(
    candidate_iou: float,
    candidate_f1: float,
    candidate_loss: float,
    best_state: BestCheckpointState,
) -> bool:
    if candidate_iou > best_state.best_val_iou:
        return True
    if candidate_iou < best_state.best_val_iou:
        return False
    if candidate_f1 > best_state.best_val_f1:
        return True
    if candidate_f1 < best_state.best_val_f1:
        return False
    return candidate_loss < best_state.best_val_loss


def train_model(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
) -> TrainingArtifacts:
    """Run supervised training on xBD train split with a reproducible validation split."""
    output_dir = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logs_dir = ensure_dir(output_dir / "logs")
    logger = setup_logger("sam_compare.train", logs_dir / "train.log")

    set_seed(config.system.seed)
    device = resolve_device(config.system.device)
    backbone_checkpoint = resolve_checkpoint_path(config.paths.checkpoint, must_exist=False)

    resume_path: Optional[Path] = None
    if config.train.resume is not None:
        resume_path = Path(config.train.resume).expanduser().resolve()
        inspection = inspect_checkpoint(resume_path)
        if inspection.checkpoint_type != "experiment":
            raise ValueError(
                f"Resume checkpoint must be a full experiment checkpoint, got {inspection.checkpoint_type}: {resume_path}"
            )
        restore_hints = apply_experiment_checkpoint_config(config, inspection.payload)
        if restore_hints.backbone_checkpoint is not None:
            backbone_checkpoint = restore_hints.backbone_checkpoint

    train_loader, val_loader, split_info = build_train_val_dataloaders(config)

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
    initial_backbone_trainable = _apply_backbone_training_stage(model, config, epoch=0)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    use_grad_scaler = (
        device.type == "cuda"
        and config.model.use_amp
        and config.model.amp_dtype.lower() in {"float16", "fp16"}
    )
    grad_scaler = GradScaler(enabled=use_grad_scaler)
    current_lrs = _get_current_lrs(optimizer)
    train_sampler_info = dict(split_info.get("train_sampler", {}))

    train_config_path = output_dir / "train_config.json"
    training_summary_path = output_dir / "training_summary.json"
    write_json(
        {
            "config": config.to_dict(),
            "resolved_backbone_checkpoint": str(backbone_checkpoint) if backbone_checkpoint else None,
            "trainable_parameters_initial": count_trainable_parameters(model),
            "initial_backbone_trainable": initial_backbone_trainable,
            "optimizer_lrs": _format_optimizer_lrs(optimizer),
            "current_backbone_lr": current_lrs.get("backbone"),
            "current_decoder_lr": current_lrs.get("decoder"),
            "scheduler_enabled": scheduler is not None,
            "early_stopping_enabled": bool(config.train.enable_early_stopping),
            "dataset_split": split_info,
        },
        train_config_path,
    )

    start_epoch = 0
    best_state = BestCheckpointState()
    early_stopping_state = EarlyStoppingState()
    if resume_path is not None:
        logger.info("Resuming training from %s", resume_path)
        start_epoch, best_state, early_stopping_state = _load_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            resume_path,
            logger=logger,
        )
    current_backbone_trainable = _apply_backbone_training_stage(model, config, start_epoch)

    logger.info("Training on device: %s", device)
    logger.info("Resolved backbone checkpoint: %s", backbone_checkpoint)
    logger.info(
        "Train/val split sizes: train=%d val=%d (val_ratio=%.3f seed=%d)",
        split_info["train_size"],
        split_info["val_size"],
        config.train.val_ratio,
        config.train.split_seed,
    )
    logger.info("Initial trainable parameters: %d", count_trainable_parameters(model))
    logger.info(
        "Optimizer LR groups: %s | stage1_freeze_backbone_epochs=%d | backbone_lr_scale=%.3f | backbone_trainable_at_start_epoch=%s",
        _format_optimizer_lrs(optimizer),
        config.train.stage1_freeze_backbone_epochs,
        config.train.backbone_lr_scale,
        current_backbone_trainable,
    )
    logger.info(
        "Scheduler config: enabled=%s | type=%s | monitor=%s | mode=%s | factor=%.3f | patience=%d | threshold=%.4f | min_lr_backbone=%.6g | min_lr_decoder=%.6g",
        scheduler is not None,
        config.train.scheduler_type,
        config.train.scheduler_monitor,
        config.train.scheduler_mode,
        config.train.scheduler_factor,
        config.train.scheduler_patience,
        config.train.scheduler_threshold,
        config.train.scheduler_min_lr_backbone,
        config.train.scheduler_min_lr_decoder,
    )
    logger.info(
        "Early stopping config: enabled=%s | patience=%d | min_delta=%.4f | start_epoch=%d",
        config.train.enable_early_stopping,
        config.train.early_stopping_patience,
        config.train.early_stopping_min_delta,
        config.train.early_stopping_start_epoch,
    )
    logger.info(
        "Train sampler: enabled=%s | sampler_bucket_counts=%s | sampler_bucket_weights=%s",
        train_sampler_info.get("enabled", False),
        train_sampler_info.get("bucket_counts", {}),
        train_sampler_info.get("bucket_weights", {}),
    )
    logger.info(
        "Validation threshold=%.3f | postprocess=%s | min_component_area=%d | max_hole_area=%d",
        config.eval.threshold,
        config.eval.enable_postprocess,
        config.eval.min_component_area,
        config.eval.max_hole_area,
    )

    best_model_path = output_dir / "best_model.pth"
    last_model_path = output_dir / "last_model.pth"
    history: list[dict[str, float | int | bool]] = []
    epochs_completed = start_epoch

    for epoch in range(start_epoch, config.train.epochs):
        backbone_trainable = _apply_backbone_training_stage(model, config, epoch)
        epoch_start_lrs = _get_current_lrs(optimizer)
        logger.info(
            "Epoch %d/%d starting | backbone_trainable=%s | current_backbone_lr=%.6g | current_decoder_lr=%.6g | lr_groups=%s",
            epoch + 1,
            config.train.epochs,
            backbone_trainable,
            epoch_start_lrs.get("backbone", 0.0),
            epoch_start_lrs.get("decoder", 0.0),
            _format_optimizer_lrs(optimizer),
        )

        model.train()
        train_epoch_stats = _init_epoch_stats()

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            model_output = model(images)
            loss_terms = compute_training_loss(model_output, masks, config)
            loss = loss_terms["loss"]

            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                if config.train.grad_clip_norm > 0:
                    grad_scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if config.train.grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
                optimizer.step()

            _accumulate_epoch_stats(train_epoch_stats, loss_terms)

            if step % config.train.log_interval == 0 or step == len(train_loader):
                logger.info(
                    "Epoch %d/%d | Step %d/%d | loss=%.6f | weighted_bce=%.6f | tversky=%.6f | boundary_loss=%.6f | presence_loss=%.6f | presence_acc=%.4f | fg_ratio=%.5f | small_fg_fraction=%.3f | empty_fraction=%.3f | sample_boost_mean=%.3f",
                    epoch + 1,
                    config.train.epochs,
                    step,
                    len(train_loader),
                    float(loss.item()),
                    float(loss_terms["weighted_bce"].item()),
                    float(loss_terms["tversky_loss"].item()),
                    float(loss_terms["boundary_loss"].item()),
                    float(loss_terms["presence_loss"].item()),
                    float(loss_terms["presence_acc"].item()),
                    float(loss_terms["foreground_ratio"].item()),
                    float(loss_terms["small_fg_fraction"].item()),
                    float(loss_terms["empty_fraction"].item()),
                    float(loss_terms["sample_boost_mean"].item()),
                )

        train_results = _finalize_epoch_stats(train_epoch_stats)
        val_results = _run_validation(model, val_loader, config, device)
        val_summary = val_results["summary"]
        val_overall = val_summary["overall"]

        candidate_iou = float(val_overall["iou"])
        candidate_f1 = float(val_overall["f1"])
        candidate_loss = float(val_results["loss"])
        scheduler_metric = _scheduler_step_metric(
            config,
            val_loss=candidate_loss,
            val_iou=candidate_iou,
            val_f1=candidate_f1,
        )
        scheduler_step_happened = False
        lrs_before_scheduler = _get_current_lrs(optimizer)
        # Plateau scheduling must observe the completed validation metrics from
        # this epoch, so we step it only after validation rather than per batch.
        if scheduler is not None:
            scheduler.step(scheduler_metric)
            lrs_after_scheduler = _get_current_lrs(optimizer)
            scheduler_step_happened = any(
                lrs_after_scheduler.get(group_name, 0.0)
                < lrs_before_scheduler.get(group_name, 0.0) - 1e-12
                for group_name in set(lrs_before_scheduler) | set(lrs_after_scheduler)
            )
        else:
            lrs_after_scheduler = lrs_before_scheduler

        current_backbone_lr = lrs_after_scheduler.get("backbone")
        current_decoder_lr = lrs_after_scheduler.get("decoder")
        should_stop = _update_early_stopping_state(
            early_stopping_state,
            candidate_metric=candidate_iou,
            epoch=epoch,
            config=config,
        )
        logged_no_improve_epochs = (
            int(early_stopping_state.no_improve_epochs)
            if config.train.enable_early_stopping
            else 0
        )
        logged_stopped_early = (
            bool(early_stopping_state.stopped_early)
            if config.train.enable_early_stopping
            else False
        )
        logged_stop_epoch = (
            early_stopping_state.stop_epoch
            if config.train.enable_early_stopping
            else None
        )

        val_metrics = {
            "iou": candidate_iou,
            "f1": candidate_f1,
            "precision": float(val_overall["precision"]),
            "recall": float(val_overall["recall"]),
            "pixel_accuracy": float(val_overall["pixel_accuracy"]),
        }

        if _is_better_checkpoint(candidate_iou, candidate_f1, candidate_loss, best_state):
            best_state = BestCheckpointState(
                best_epoch=epoch,
                best_val_iou=candidate_iou,
                best_val_f1=candidate_f1,
                best_val_loss=candidate_loss,
            )
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                config,
                best_model_path,
                epoch=epoch,
                best_state=best_state,
                early_stopping_state=early_stopping_state,
                train_loss=float(train_results["loss"]),
                val_loss=candidate_loss,
                val_metrics=val_metrics,
                backbone_checkpoint=backbone_checkpoint,
            )
            logger.info(
                "Updated best model at epoch %d | best_val_iou=%.6f | best_val_f1=%.6f | best_val_loss=%.6f",
                epoch + 1,
                best_state.best_val_iou,
                best_state.best_val_f1,
                best_state.best_val_loss,
            )

        epoch_record = {
            "epoch": epoch + 1,
            "backbone_trainable": backbone_trainable,
            "train_loss": float(train_results["loss"]),
            "train_weighted_bce": float(train_results["weighted_bce"]),
            "train_tversky_loss": float(train_results["tversky_loss"]),
            "train_boundary_loss": float(train_results["boundary_loss"]),
            "train_presence_loss": float(train_results["presence_loss"]),
            "train_presence_acc": float(train_results["presence_acc"]),
            "train_foreground_ratio": float(train_results["foreground_ratio"]),
            "train_small_fg_fraction": float(train_results["small_fg_fraction"]),
            "train_empty_fraction_mean": float(train_results["empty_fraction"]),
            "train_presence_acc_empty": float(train_results["presence_acc_empty"]),
            "train_presence_acc_non_empty": float(train_results["presence_acc_non_empty"]),
            "sample_boost_mean": float(train_results["sample_boost_mean"]),
            "train_sample_boost_mean": float(train_results["sample_boost_mean"]),
            "val_loss": candidate_loss,
            "val_weighted_bce": float(val_results["weighted_bce"]),
            "val_tversky_loss": float(val_results["tversky_loss"]),
            "val_boundary_loss": float(val_results["boundary_loss"]),
            "val_presence_loss": float(val_results["presence_loss"]),
            "val_presence_acc": float(val_results["presence_acc"]),
            "val_foreground_ratio": float(val_results["foreground_ratio"]),
            "val_small_fg_fraction": float(val_results["small_fg_fraction"]),
            "val_empty_fraction_mean": float(val_results["empty_fraction"]),
            "val_presence_acc_empty": float(val_results["presence_acc_empty"]),
            "val_presence_acc_non_empty": float(val_results["presence_acc_non_empty"]),
            "val_iou": candidate_iou,
            "val_f1": candidate_f1,
            "val_precision": float(val_overall["precision"]),
            "val_recall": float(val_overall["recall"]),
            "val_pixel_accuracy": float(val_overall["pixel_accuracy"]),
            "current_backbone_lr": current_backbone_lr,
            "current_decoder_lr": current_decoder_lr,
            "scheduler_enabled": scheduler is not None,
            "scheduler_step_metric": scheduler_metric,
            "scheduler_step_happened": scheduler_step_happened,
            "early_stopping_enabled": bool(config.train.enable_early_stopping),
            "no_improve_epochs": logged_no_improve_epochs,
            "stopped_early": logged_stopped_early,
            "stop_epoch": logged_stop_epoch,
            "best_val_iou": best_state.best_val_iou,
            "best_val_f1": best_state.best_val_f1,
            "best_val_loss": best_state.best_val_loss,
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d finished | train_loss=%.6f | val_loss=%.6f | val_iou=%.6f | val_f1=%.6f | val_precision=%.6f | val_recall=%.6f | val_boundary_loss=%.6f | val_presence_loss=%.6f | val_presence_acc=%.4f | train_empty_fraction_mean=%.3f | val_empty_fraction_mean=%.3f | train_presence_acc_empty=%.4f | train_presence_acc_non_empty=%.4f | val_presence_acc_empty=%.4f | val_presence_acc_non_empty=%.4f | sample_boost_mean=%.3f | current_backbone_lr=%.6g | current_decoder_lr=%.6g | scheduler_enabled=%s | scheduler_step_metric=%.6f | scheduler_step_happened=%s | early_stopping_enabled=%s | no_improve_epochs=%d | best_epoch=%d | best_val_iou=%.6f | best_val_f1=%.6f | best_val_loss=%.6f",
            epoch + 1,
            float(train_results["loss"]),
            candidate_loss,
            candidate_iou,
            candidate_f1,
            float(val_overall["precision"]),
            float(val_overall["recall"]),
            float(val_results["boundary_loss"]),
            float(val_results["presence_loss"]),
            float(val_results["presence_acc"]),
            float(train_results["empty_fraction"]),
            float(val_results["empty_fraction"]),
            float(train_results["presence_acc_empty"]),
            float(train_results["presence_acc_non_empty"]),
            float(val_results["presence_acc_empty"]),
            float(val_results["presence_acc_non_empty"]),
            float(train_results["sample_boost_mean"]),
            current_backbone_lr if current_backbone_lr is not None else 0.0,
            current_decoder_lr if current_decoder_lr is not None else 0.0,
            scheduler is not None,
            scheduler_metric,
            scheduler_step_happened,
            config.train.enable_early_stopping,
            logged_no_improve_epochs,
            best_state.best_epoch + 1 if best_state.best_epoch >= 0 else -1,
            best_state.best_val_iou if best_state.best_epoch >= 0 else float("-inf"),
            best_state.best_val_f1 if best_state.best_epoch >= 0 else float("-inf"),
            best_state.best_val_loss if best_state.best_epoch >= 0 else float("inf"),
        )
        if scheduler_step_happened:
            logger.info(
                "Scheduler reduced learning rates after epoch %d | current_backbone_lr=%.6g | current_decoder_lr=%.6g",
                epoch + 1,
                current_backbone_lr if current_backbone_lr is not None else 0.0,
                current_decoder_lr if current_decoder_lr is not None else 0.0,
            )

        epoch_checkpoint = checkpoints_dir / f"epoch_{epoch + 1:03d}.pth"
        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            epoch_checkpoint,
            epoch=epoch,
            best_state=best_state,
            early_stopping_state=early_stopping_state,
            train_loss=float(train_results["loss"]),
            val_loss=candidate_loss,
            val_metrics=val_metrics,
            backbone_checkpoint=backbone_checkpoint,
        )
        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            last_model_path,
            epoch=epoch,
            best_state=best_state,
            early_stopping_state=early_stopping_state,
            train_loss=float(train_results["loss"]),
            val_loss=candidate_loss,
            val_metrics=val_metrics,
            backbone_checkpoint=backbone_checkpoint,
        )

        write_json(
            {
                "config": config.to_dict(),
                "resolved_backbone_checkpoint": str(backbone_checkpoint) if backbone_checkpoint else None,
                "epochs_completed": epoch + 1,
                "best_epoch": best_state.best_epoch + 1 if best_state.best_epoch >= 0 else None,
                "best_val_iou": best_state.best_val_iou,
                "best_val_f1": best_state.best_val_f1,
                "best_val_loss": best_state.best_val_loss,
                "current_backbone_lr": current_backbone_lr,
                "current_decoder_lr": current_decoder_lr,
                "scheduler_enabled": scheduler is not None,
                "scheduler_step_metric": scheduler_metric,
                "scheduler_step_happened": scheduler_step_happened,
                "early_stopping_enabled": bool(config.train.enable_early_stopping),
                "no_improve_epochs": logged_no_improve_epochs,
                "stopped_early": logged_stopped_early,
                "stop_epoch": logged_stop_epoch,
                "train_empty_fraction_mean": float(train_results["empty_fraction"]),
                "val_empty_fraction_mean": float(val_results["empty_fraction"]),
                "train_presence_acc_empty": float(train_results["presence_acc_empty"]),
                "train_presence_acc_non_empty": float(train_results["presence_acc_non_empty"]),
                "val_presence_acc_empty": float(val_results["presence_acc_empty"]),
                "val_presence_acc_non_empty": float(val_results["presence_acc_non_empty"]),
                "sample_boost_mean": float(train_results["sample_boost_mean"]),
                "validation_threshold": config.eval.threshold,
                "foreground_ratio_stats": {
                    "train_mean": float(train_results["foreground_ratio"]),
                    "train_small_fg_fraction": float(train_results["small_fg_fraction"]),
                    "train_empty_fraction_mean": float(train_results["empty_fraction"]),
                    "train_sample_boost_mean": float(train_results["sample_boost_mean"]),
                    "val_mean": float(val_results["foreground_ratio"]),
                    "val_small_fg_fraction": float(val_results["small_fg_fraction"]),
                    "val_empty_fraction_mean": float(val_results["empty_fraction"]),
                },
                "auxiliary_losses": {
                    "train_boundary_loss": float(train_results["boundary_loss"]),
                    "train_presence_loss": float(train_results["presence_loss"]),
                    "train_presence_acc": float(train_results["presence_acc"]),
                    "train_presence_acc_empty": float(train_results["presence_acc_empty"]),
                    "train_presence_acc_non_empty": float(train_results["presence_acc_non_empty"]),
                    "val_boundary_loss": float(val_results["boundary_loss"]),
                    "val_presence_loss": float(val_results["presence_loss"]),
                    "val_presence_acc": float(val_results["presence_acc"]),
                    "val_presence_acc_empty": float(val_results["presence_acc_empty"]),
                    "val_presence_acc_non_empty": float(val_results["presence_acc_non_empty"]),
                },
                "sampler_bucket_counts": train_sampler_info.get("bucket_counts", {}),
                "sampler_bucket_weights": train_sampler_info.get("bucket_weights", {}),
                "dataset_split": split_info,
                "history": history,
            },
            training_summary_path,
        )
        epochs_completed = epoch + 1

        # Early stopping only truncates future epochs; the usual best/last/epoch
        # checkpoints are still written first so resume/export behavior stays intact.
        if should_stop:
            logger.info(
                "Early stopping triggered at epoch %d | no_improve_epochs=%d | stop_epoch=%d | early_stop_best_metric=%.6f | best_val_iou=%.6f",
                epoch + 1,
                early_stopping_state.no_improve_epochs,
                early_stopping_state.stop_epoch if early_stopping_state.stop_epoch is not None else -1,
                early_stopping_state.best_metric,
                best_state.best_val_iou,
            )
            break

    final_no_improve_epochs = (
        int(early_stopping_state.no_improve_epochs)
        if config.train.enable_early_stopping
        else 0
    )
    final_stopped_early = (
        bool(early_stopping_state.stopped_early)
        if config.train.enable_early_stopping
        else False
    )
    final_stop_epoch = (
        early_stopping_state.stop_epoch
        if config.train.enable_early_stopping
        else None
    )
    if not history:
        final_lrs = _get_current_lrs(optimizer)
        write_json(
            {
                "config": config.to_dict(),
                "resolved_backbone_checkpoint": str(backbone_checkpoint) if backbone_checkpoint else None,
                "epochs_completed": epochs_completed,
                "best_epoch": best_state.best_epoch + 1 if best_state.best_epoch >= 0 else None,
                "best_val_iou": best_state.best_val_iou,
                "best_val_f1": best_state.best_val_f1,
                "best_val_loss": best_state.best_val_loss,
                "current_backbone_lr": final_lrs.get("backbone"),
                "current_decoder_lr": final_lrs.get("decoder"),
                "scheduler_enabled": scheduler is not None,
                "scheduler_step_metric": None,
                "scheduler_step_happened": False,
                "early_stopping_enabled": bool(config.train.enable_early_stopping),
                "no_improve_epochs": final_no_improve_epochs,
                "stopped_early": final_stopped_early,
                "stop_epoch": final_stop_epoch,
                "sampler_bucket_counts": train_sampler_info.get("bucket_counts", {}),
                "sampler_bucket_weights": train_sampler_info.get("bucket_weights", {}),
                "dataset_split": split_info,
                "history": history,
            },
            training_summary_path,
        )

    logger.info(
        "Training complete. epochs_completed=%d | stopped_early=%s | stop_epoch=%s | best_epoch=%d | best_val_iou=%.6f | best_val_f1=%.6f | best_val_loss=%.6f",
        epochs_completed,
        final_stopped_early,
        final_stop_epoch,
        best_state.best_epoch + 1 if best_state.best_epoch >= 0 else -1,
        best_state.best_val_iou,
        best_state.best_val_f1,
        best_state.best_val_loss,
    )
    return TrainingArtifacts(
        output_dir=Path(output_dir),
        best_model_path=best_model_path,
        last_model_path=last_model_path,
        checkpoints_dir=checkpoints_dir,
        train_config_path=train_config_path,
        training_summary_path=training_summary_path,
        backbone_checkpoint=backbone_checkpoint,
        epochs_completed=epochs_completed,
        best_epoch=best_state.best_epoch + 1 if best_state.best_epoch >= 0 else -1,
        best_val_iou=best_state.best_val_iou,
    )
