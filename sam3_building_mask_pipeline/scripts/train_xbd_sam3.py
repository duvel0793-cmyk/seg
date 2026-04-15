#!/usr/bin/env python3
"""Train SAM3-based binary building segmentation on xBD train split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_compare.config import load_experiment_config
from sam_compare.paths import CONFIGS_DIR
from sam_compare.trainer import train_model
from sam_compare.utils import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Optional extra YAML config to merge.")
    parser.add_argument("--xbd-root", type=str, default=None, help="Path to xBD root.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM3 base checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Training output directory.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a saved experiment checkpoint.")
    parser.add_argument("--image-size", type=int, default=None, help="Resized square image size fed to SAM3.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation ratio split from xBD train.")
    parser.add_argument("--split-seed", type=int, default=None, help="Seed for reproducible train/val split.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Foreground weight for weighted BCE.")
    parser.add_argument("--tversky-alpha", type=float, default=None, help="Tversky alpha term.")
    parser.add_argument("--tversky-beta", type=float, default=None, help="Tversky beta term.")
    parser.add_argument("--small-fg-ratio-thr", type=float, default=None, help="Foreground-ratio threshold for small-sample boosting.")
    parser.add_argument("--empty-fg-boost", type=float, default=None, help="Loss multiplier for empty-foreground samples.")
    parser.add_argument("--small-fg-boost", type=float, default=None, help="Loss multiplier for small-foreground samples.")
    parser.add_argument("--boundary-aux-weight", type=float, default=None, help="Auxiliary boundary-loss weight.")
    parser.add_argument("--presence-aux-weight", type=float, default=None, help="Auxiliary presence-loss weight.")
    parser.add_argument("--presence-empty-weight", type=float, default=None, help="Presence-loss weight for empty images.")
    parser.add_argument("--presence-small-weight", type=float, default=None, help="Presence-loss weight for small-foreground images.")
    parser.add_argument("--stage1-freeze-backbone-epochs", type=int, default=None, help="Initial epochs with the backbone frozen.")
    parser.add_argument("--backbone-lr-scale", type=float, default=None, help="Learning-rate scale applied to backbone parameters.")
    parser.add_argument("--scheduler-type", type=str, default=None, help="Scheduler type. Currently supports plateau.")
    parser.add_argument("--scheduler-monitor", type=str, default=None, help="Validation metric used to step the scheduler.")
    parser.add_argument("--scheduler-mode", type=str, default=None, help="Scheduler mode, for example max or min.")
    parser.add_argument("--scheduler-factor", type=float, default=None, help="Factor applied when the scheduler reduces learning rates.")
    parser.add_argument("--scheduler-patience", type=int, default=None, help="Number of plateau epochs before reducing learning rates.")
    parser.add_argument("--scheduler-threshold", type=float, default=None, help="Minimum metric change treated as scheduler improvement.")
    parser.add_argument("--scheduler-min-lr-backbone", type=float, default=None, help="Lower bound for the backbone learning rate.")
    parser.add_argument("--scheduler-min-lr-decoder", type=float, default=None, help="Lower bound for the decoder learning rate.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Number of monitored epochs without improvement before stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None, help="Minimum val IoU gain required to reset early stopping.")
    parser.add_argument("--early-stopping-start-epoch", type=int, default=None, help="Epoch index (1-based) from which early stopping becomes active.")
    parser.add_argument("--sample-weight-empty", type=float, default=None, help="Train sampler weight for empty-foreground samples.")
    parser.add_argument("--sample-weight-small", type=float, default=None, help="Train sampler weight for small-foreground samples.")
    parser.add_argument("--sample-weight-medium", type=float, default=None, help="Train sampler weight for medium-foreground samples.")
    parser.add_argument("--sample-weight-large", type=float, default=None, help="Train sampler weight for large-foreground samples.")
    parser.add_argument("--decoder-dropout", type=float, default=None, help="Dropout before the segmentation logits head.")
    parser.add_argument(
        "--enable-scheduler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable validation-driven learning-rate scheduling.",
    )
    parser.add_argument(
        "--enable-early-stopping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable validation-driven early stopping.",
    )
    parser.add_argument(
        "--enable-weighted-sampler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable train-only bucket-aware weighted sampling.",
    )
    parser.add_argument(
        "--use-resaspp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the multi-dilation context block in the decoder.",
    )
    parser.add_argument(
        "--use-attention-fusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gated top-down feature fusion in the decoder.",
    )
    parser.add_argument(
        "--use-presence-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the image-level building-presence head.",
    )
    parser.add_argument(
        "--use-boundary-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the boundary auxiliary head.",
    )
    parser.add_argument(
        "--enable-train-aug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable train-only data augmentation.",
    )
    parser.add_argument(
        "--aug-hflip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable random horizontal flip in train augmentation.",
    )
    parser.add_argument(
        "--aug-vflip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable random vertical flip in train augmentation.",
    )
    parser.add_argument(
        "--aug-rot90",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable random 90-degree rotation in train augmentation.",
    )
    parser.add_argument(
        "--aug-brightness-contrast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable mild brightness/contrast jitter in train augmentation.",
    )
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze SAM3 backbone and train decoder only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_paths = [CONFIGS_DIR / "default.yaml", CONFIGS_DIR / "train.yaml"]
    if args.config:
        config_paths.append(Path(args.config))
    config = load_experiment_config(config_paths)

    if args.xbd_root is not None:
        config.paths.xbd_root = Path(args.xbd_root).expanduser().resolve()
    if args.checkpoint is not None:
        config.paths.checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.output_dir is None:
        output_dir = config.paths.output_root / "xbd_sam3_train"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    if args.device is not None:
        config.system.device = args.device
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.num_workers is not None:
        config.system.num_workers = args.num_workers
    if args.resume is not None:
        config.train.resume = Path(args.resume).expanduser().resolve()
    if args.image_size is not None:
        config.data.image_size = args.image_size
    if args.val_ratio is not None:
        config.train.val_ratio = args.val_ratio
    if args.split_seed is not None:
        config.train.split_seed = args.split_seed
    if args.pos_weight is not None:
        config.train.pos_weight = args.pos_weight
    if args.tversky_alpha is not None:
        config.train.tversky_alpha = args.tversky_alpha
    if args.tversky_beta is not None:
        config.train.tversky_beta = args.tversky_beta
    if args.small_fg_ratio_thr is not None:
        config.train.small_fg_ratio_thr = args.small_fg_ratio_thr
    if args.empty_fg_boost is not None:
        config.train.empty_fg_boost = args.empty_fg_boost
    if args.small_fg_boost is not None:
        config.train.small_fg_boost = args.small_fg_boost
    if args.boundary_aux_weight is not None:
        config.train.boundary_aux_weight = args.boundary_aux_weight
    if args.presence_aux_weight is not None:
        config.train.presence_aux_weight = args.presence_aux_weight
    if args.presence_empty_weight is not None:
        config.train.presence_empty_weight = args.presence_empty_weight
    if args.presence_small_weight is not None:
        config.train.presence_small_weight = args.presence_small_weight
    if args.stage1_freeze_backbone_epochs is not None:
        config.train.stage1_freeze_backbone_epochs = args.stage1_freeze_backbone_epochs
    if args.backbone_lr_scale is not None:
        config.train.backbone_lr_scale = args.backbone_lr_scale
    if args.enable_scheduler is not None:
        config.train.enable_scheduler = args.enable_scheduler
    if args.scheduler_type is not None:
        config.train.scheduler_type = args.scheduler_type
    if args.scheduler_monitor is not None:
        config.train.scheduler_monitor = args.scheduler_monitor
    if args.scheduler_mode is not None:
        config.train.scheduler_mode = args.scheduler_mode
    if args.scheduler_factor is not None:
        config.train.scheduler_factor = args.scheduler_factor
    if args.scheduler_patience is not None:
        config.train.scheduler_patience = args.scheduler_patience
    if args.scheduler_threshold is not None:
        config.train.scheduler_threshold = args.scheduler_threshold
    if args.scheduler_min_lr_backbone is not None:
        config.train.scheduler_min_lr_backbone = args.scheduler_min_lr_backbone
    if args.scheduler_min_lr_decoder is not None:
        config.train.scheduler_min_lr_decoder = args.scheduler_min_lr_decoder
    if args.enable_early_stopping is not None:
        config.train.enable_early_stopping = args.enable_early_stopping
    if args.early_stopping_patience is not None:
        config.train.early_stopping_patience = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        config.train.early_stopping_min_delta = args.early_stopping_min_delta
    if args.early_stopping_start_epoch is not None:
        config.train.early_stopping_start_epoch = args.early_stopping_start_epoch
    if args.enable_weighted_sampler is not None:
        config.train.enable_weighted_sampler = args.enable_weighted_sampler
    if args.sample_weight_empty is not None:
        config.train.sample_weight_empty = args.sample_weight_empty
    if args.sample_weight_small is not None:
        config.train.sample_weight_small = args.sample_weight_small
    if args.sample_weight_medium is not None:
        config.train.sample_weight_medium = args.sample_weight_medium
    if args.sample_weight_large is not None:
        config.train.sample_weight_large = args.sample_weight_large
    if args.decoder_dropout is not None:
        config.model.decoder_dropout = args.decoder_dropout
    if args.use_resaspp is not None:
        config.model.use_resaspp = args.use_resaspp
    if args.use_attention_fusion is not None:
        config.model.use_attention_fusion = args.use_attention_fusion
    if args.use_presence_head is not None:
        config.model.use_presence_head = args.use_presence_head
    if args.use_boundary_head is not None:
        config.model.use_boundary_head = args.use_boundary_head
    if args.enable_train_aug is not None:
        config.data.enable_train_aug = args.enable_train_aug
    if args.aug_hflip is not None:
        config.data.aug_hflip = args.aug_hflip
    if args.aug_vflip is not None:
        config.data.aug_vflip = args.aug_vflip
    if args.aug_rot90 is not None:
        config.data.aug_rot90 = args.aug_rot90
    if args.aug_brightness_contrast is not None:
        config.data.aug_brightness_contrast = args.aug_brightness_contrast
    if args.freeze_backbone:
        config.model.freeze_backbone = True

    artifacts = train_model(config, output_dir=output_dir)
    summary = {
        "output_dir": str(artifacts.output_dir),
        "best_model": str(artifacts.best_model_path),
        "last_model": str(artifacts.last_model_path),
        "checkpoints_dir": str(artifacts.checkpoints_dir),
        "train_config": str(artifacts.train_config_path),
        "training_summary": str(artifacts.training_summary_path),
        "resolved_backbone_checkpoint": str(artifacts.backbone_checkpoint) if artifacts.backbone_checkpoint else None,
        "epochs_completed": artifacts.epochs_completed,
        "best_epoch": artifacts.best_epoch,
        "best_val_iou": artifacts.best_val_iou,
    }
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
