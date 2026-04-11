from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.xbd_oracle_instance_damage import (
    CLASS_NAMES,
    XBDOracleInstanceDamageDataset,
    oracle_instance_collate_fn,
)
from models import build_model
from utils.io import (
    append_jsonl,
    ensure_dir,
    load_checkpoint,
    read_yaml,
    save_checkpoint,
    write_json,
    write_text,
    write_yaml,
)
from utils.losses import (
    CORNLoss,
    DamageLossModule,
    OrdinalSupConLoss,
    build_loss_function,
    compute_class_weights,
    compute_expected_severity_from_probabilities,
    compute_js_divergence_from_probabilities,
)
from utils.metrics import compute_classification_metrics
from utils.rebalance import build_stage2_sampler
from utils.seed import seed_worker, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_TYPE_CHOICES = ["post_only", "siamese_simple", "oracle_mcd", "oracle_mcd_corn"]
HEAD_TYPE_CHOICES = ["standard", "corn"]
LOSS_MODE_CHOICES = [
    "weighted_ce",
    "fixed_cda",
    "learnable_cda",
    "adaptive_ucl_cda",
    "adaptive_ucl_cda_v2",
    "adaptive_ucl_cda_v3",
    "corn",
    "corn_adaptive_tau_safe",
]


def resolve_default_config_path() -> Path:
    preferred = PROJECT_ROOT / "configs" / "vscode_run.yaml"
    if preferred.exists():
        return preferred
    return PROJECT_ROOT / "configs" / "default.yaml"


DEFAULT_CONFIG_PATH = resolve_default_config_path()


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xBD Oracle Instance Damage Classification")
    bool_action = getattr(argparse, "BooleanOptionalAction", None)

    def add_optional_bool_argument(name: str, *, default: bool | None = None) -> None:
        if bool_action is None:
            raise RuntimeError("BooleanOptionalAction is required for this training CLI.")
        parser.add_argument(name, action=bool_action, default=default)

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))

    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--instance_source", type=str, default=None)
    parser.add_argument("--allow_tier3", action="store_true")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--context_ratio", type=float, default=None)
    parser.add_argument("--min_polygon_area", type=float, default=None)
    parser.add_argument("--min_mask_pixels", type=int, default=None)
    parser.add_argument("--max_out_of_bound_ratio", type=float, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--model_type", type=str, choices=MODEL_TYPE_CHOICES, default=None)
    parser.add_argument("--head_type", type=str, choices=HEAD_TYPE_CHOICES, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--loss_mode", type=str, choices=LOSS_MODE_CHOICES, default=None)
    parser.add_argument("--lambda_ord", type=float, default=None)
    parser.add_argument("--lambda_uni", type=float, default=None)
    parser.add_argument("--lambda_conc", type=float, default=None)
    parser.add_argument("--lambda_gap_reg", type=float, default=None)
    parser.add_argument("--lambda_tau", type=float, default=None)
    parser.add_argument("--lambda_tau_mean", type=float, default=None)
    parser.add_argument("--lambda_tau_diff", type=float, default=None)
    parser.add_argument("--lambda_tau_rank", type=float, default=None)
    parser.add_argument("--lambda_raw_tau_diff", type=float, default=None)
    parser.add_argument("--lambda_raw_tau_center", type=float, default=None)
    parser.add_argument("--lambda_raw_tau_bound", type=float, default=None)
    parser.add_argument("--lambda_corn_soft", type=float, default=None)
    parser.add_argument("--fixed_cda_alpha", type=float, default=None)
    parser.add_argument("--tau_init", type=float, default=None)
    parser.add_argument("--tau_min", type=float, default=None)
    parser.add_argument("--tau_max", type=float, default=None)
    parser.add_argument("--tau_base", type=float, default=None)
    parser.add_argument("--delta_scale", type=float, default=None)
    parser.add_argument("--tau_parameterization", type=str, choices=["sigmoid", "residual", "bounded_sigmoid"], default=None)
    parser.add_argument("--tau_logit_scale", type=float, default=None)
    parser.add_argument("--tau_target", type=float, default=None)
    parser.add_argument("--tau_easy", type=float, default=None)
    parser.add_argument("--tau_hard", type=float, default=None)
    parser.add_argument("--tau_variance_weight", type=float, default=None)
    parser.add_argument("--tau_std_floor", type=float, default=None)
    parser.add_argument("--tau_rank_margin_difficulty", type=float, default=None)
    parser.add_argument("--tau_rank_margin_value", type=float, default=None)
    parser.add_argument("--raw_tau_soft_margin", type=float, default=None)
    parser.add_argument("--tau_freeze_epochs", type=int, default=None)
    parser.add_argument("--tau_warmup_value", type=float, default=None)
    parser.add_argument("--corn_soft_start_epoch", type=int, default=None)
    parser.add_argument("--ambiguity_lr_scale", type=float, default=None)
    parser.add_argument("--concentration_margin", type=float, default=None)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--save_topk", type=int, default=None)
    add_optional_bool_argument("--dual_view_enabled", default=None)
    parser.add_argument("--lambda_view_consistency", type=float, default=None)
    parser.add_argument("--lambda_view_severity_consistency", type=float, default=None)
    parser.add_argument("--dual_view_stronger_context_prob", type=float, default=None)
    add_optional_bool_argument("--enable_ordinal_contrastive", default=None)
    parser.add_argument("--lambda_contrastive", type=float, default=None)
    parser.add_argument("--contrastive_temperature", type=float, default=None)
    parser.add_argument("--contrastive_proj_dim", type=int, default=None)
    parser.add_argument("--contrastive_hidden_features", type=int, default=None)
    parser.add_argument("--contrastive_dropout", type=float, default=None)
    parser.add_argument("--contrastive_minor_major_weight", type=float, default=None)
    parser.add_argument("--contrastive_adjacent_weight", type=float, default=None)
    parser.add_argument("--contrastive_far_weight", type=float, default=None)
    parser.add_argument("--contrastive_min_positives", type=int, default=None)
    parser.add_argument("--contrastive_loss_warmup_epochs", type=int, default=None)
    add_optional_bool_argument("--enable_corn_soft_emd", default=None)
    parser.add_argument("--lambda_corn_soft_emd", type=float, default=None)
    add_optional_bool_argument("--enable_decoded_unimodal_reg", default=None)
    parser.add_argument("--lambda_decoded_unimodal", type=float, default=None)
    parser.add_argument("--decoded_unimodal_margin", type=float, default=None)
    add_optional_bool_argument("--stage2_head_rebalance_enabled", default=None)
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--stage2_lr", type=float, default=None)
    parser.add_argument("--stage2_weight_decay", type=float, default=None)
    parser.add_argument("--stage2_early_stop_patience", type=int, default=None)
    parser.add_argument("--stage2_sampler_mode", type=str, default=None)
    parser.add_argument("--stage2_focus_minor_major_weight", type=float, default=None)
    add_optional_bool_argument("--stage2_freeze_trunk", default=None)
    add_optional_bool_argument("--stage2_freeze_ambiguity", default=None)
    add_optional_bool_argument("--stage2_freeze_ordinal", default=None)

    parser.add_argument("--context_dropout_prob", type=float, default=None)
    parser.add_argument("--context_blur_prob", type=float, default=None)
    parser.add_argument("--context_grayscale_prob", type=float, default=None)
    parser.add_argument("--context_noise_prob", type=float, default=None)
    parser.add_argument("--context_mix_prob", type=float, default=None)
    parser.add_argument("--context_edge_soften_pixels", type=int, default=None)
    parser.add_argument("--context_dilate_pixels", type=int, default=None)
    add_optional_bool_argument("--context_apply_to_pre_and_post_independently", default=None)
    add_optional_bool_argument("--context_preserve_instance_strictly", default=None)

    add_optional_bool_argument("--enable_threshold_calibration", default=None)
    parser.add_argument("--calibration_type", type=str, default=None)
    parser.add_argument("--calibration_max_iter", type=int, default=None)
    parser.add_argument("--calibration_lr", type=float, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()


def ensure_training_defaults(config: dict[str, Any], default_loss_mode: str = "corn_adaptive_tau_safe") -> dict[str, Any]:
    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("head_type", "standard")
    model_cfg.setdefault("ambiguity_hidden_features", 256)

    aug_cfg = config.setdefault("augmentation", {})
    aug_cfg.setdefault("context_dropout_prob", 0.0)
    aug_cfg.setdefault("context_blur_prob", 0.0)
    aug_cfg.setdefault("context_grayscale_prob", 0.0)
    aug_cfg.setdefault("context_noise_prob", 0.0)
    aug_cfg.setdefault("context_mix_prob", 0.0)
    aug_cfg.setdefault("context_edge_soften_pixels", 4)
    aug_cfg.setdefault("context_dilate_pixels", 3)
    aug_cfg.setdefault("context_apply_to_pre_and_post_independently", False)
    aug_cfg.setdefault("context_preserve_instance_strictly", True)

    train_cfg = config.setdefault("training", {})
    train_cfg.setdefault("loss_mode", default_loss_mode)
    default_loss_mode_name = str(train_cfg.get("loss_mode", default_loss_mode))
    default_is_v3 = default_loss_mode_name in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
    train_cfg.setdefault("lambda_ord", 0.20)
    train_cfg.setdefault("lambda_uni", 0.10)
    train_cfg.setdefault("lambda_conc", 0.01)
    train_cfg.setdefault("lambda_gap_reg", 1e-3)
    train_cfg.setdefault("lambda_tau", 0.01)
    train_cfg.setdefault("lambda_tau_mean", 0.05)
    train_cfg.setdefault("lambda_tau_diff", 0.20)
    train_cfg.setdefault("lambda_tau_rank", 0.05)
    train_cfg.setdefault("lambda_raw_tau_diff", 0.10)
    train_cfg.setdefault("lambda_raw_tau_center", 0.02)
    train_cfg.setdefault("lambda_raw_tau_bound", 0.02)
    train_cfg.setdefault("lambda_corn_soft", 0.03)
    train_cfg.setdefault("fixed_cda_alpha", 0.3)
    train_cfg.setdefault("tau_init", 0.22)
    train_cfg.setdefault("tau_min", 0.12)
    train_cfg.setdefault("tau_max", 0.45)
    train_cfg.setdefault("tau_base", 0.22)
    train_cfg.setdefault("delta_scale", 0.10)
    train_cfg.setdefault("tau_parameterization", "bounded_sigmoid" if default_is_v3 else "sigmoid")
    train_cfg.setdefault("tau_logit_scale", 2.0)
    train_cfg.setdefault("tau_target", 0.22)
    train_cfg.setdefault("tau_easy", 0.16)
    train_cfg.setdefault("tau_hard", 0.32)
    train_cfg.setdefault("tau_variance_weight", 0.01)
    train_cfg.setdefault("tau_std_floor", 0.03)
    train_cfg.setdefault("tau_rank_margin_difficulty", 0.10)
    train_cfg.setdefault("tau_rank_margin_value", 0.01)
    train_cfg.setdefault("raw_tau_soft_margin", 1.5)
    train_cfg.setdefault("tau_freeze_epochs", 1)
    train_cfg.setdefault("tau_warmup_value", 0.22)
    train_cfg.setdefault("corn_soft_start_epoch", 3)
    train_cfg.setdefault("ambiguity_lr_scale", 0.30 if default_is_v3 else 0.5)
    train_cfg.setdefault("concentration_margin", 0.05)
    train_cfg.setdefault("lambda_aux", 0.2)
    train_cfg.setdefault("label_smoothing", 0.05)
    train_cfg.setdefault("use_focal", False)
    train_cfg.setdefault("focal_gamma", 2.0)
    train_cfg.setdefault("early_stop_patience", 6)
    train_cfg.setdefault("save_topk", 3)
    train_cfg.setdefault("dual_view_enabled", False)
    train_cfg.setdefault("lambda_view_consistency", 0.0)
    train_cfg.setdefault("lambda_view_severity_consistency", 0.0)
    train_cfg.setdefault("dual_view_stronger_context_prob", 0.20)
    train_cfg.setdefault("enable_ordinal_contrastive", False)
    train_cfg.setdefault("lambda_contrastive", 0.0)
    train_cfg.setdefault("contrastive_temperature", 0.10)
    train_cfg.setdefault("contrastive_proj_dim", 128)
    train_cfg.setdefault("contrastive_hidden_features", 512)
    train_cfg.setdefault("contrastive_dropout", 0.10)
    train_cfg.setdefault("contrastive_minor_major_weight", 1.75)
    train_cfg.setdefault("contrastive_adjacent_weight", 1.00)
    train_cfg.setdefault("contrastive_far_weight", 0.50)
    train_cfg.setdefault("contrastive_min_positives", 1)
    train_cfg.setdefault("contrastive_loss_warmup_epochs", 3)
    train_cfg.setdefault("enable_corn_soft_emd", False)
    train_cfg.setdefault("lambda_corn_soft_emd", 0.02)
    train_cfg.setdefault("enable_decoded_unimodal_reg", False)
    train_cfg.setdefault("lambda_decoded_unimodal", 0.01)
    train_cfg.setdefault("decoded_unimodal_margin", 0.0)
    train_cfg.setdefault("stage2_head_rebalance_enabled", False)
    train_cfg.setdefault("stage2_epochs", 8)
    train_cfg.setdefault("stage2_lr", 1e-4)
    train_cfg.setdefault("stage2_weight_decay", 1e-4)
    train_cfg.setdefault("stage2_early_stop_patience", 3)
    train_cfg.setdefault("stage2_sampler_mode", "class_balanced")
    train_cfg.setdefault("stage2_focus_minor_major_weight", 1.25)
    train_cfg.setdefault("stage2_freeze_trunk", True)
    train_cfg.setdefault("stage2_freeze_ambiguity", True)
    train_cfg.setdefault("stage2_freeze_ordinal", True)

    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg.setdefault("enable_threshold_calibration", False)
    eval_cfg.setdefault("calibration_type", "temperature_plus_bias")
    eval_cfg.setdefault("calibration_max_iter", 200)
    eval_cfg.setdefault("calibration_lr", 0.01)
    return config


def harmonize_model_and_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    train_cfg = config["training"]

    model_type = str(model_cfg["model_type"])
    head_type = str(model_cfg.get("head_type", "standard"))
    loss_mode = str(train_cfg["loss_mode"])

    if model_type in {"post_only", "siamese_simple"}:
        if loss_mode in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", "corn", "corn_adaptive_tau_safe"} or head_type == "corn":
            raise ValueError(f"{model_type} does not support loss_mode={loss_mode} / head_type={head_type}.")
        model_cfg["head_type"] = "standard"
        return config

    if model_type not in {"oracle_mcd", "oracle_mcd_corn"}:
        raise ValueError(f"Unsupported model_type='{model_type}'.")

    if model_type == "oracle_mcd_corn" or head_type == "corn" or loss_mode in {"corn", "corn_adaptive_tau_safe"}:
        model_cfg["model_type"] = "oracle_mcd_corn"
        model_cfg["head_type"] = "corn"
        if loss_mode not in {"corn", "corn_adaptive_tau_safe"}:
            train_cfg["loss_mode"] = "corn"
        return config

    model_cfg["model_type"] = "oracle_mcd"
    model_cfg["head_type"] = "standard"
    if loss_mode not in {
        "weighted_ce",
        "fixed_cda",
        "learnable_cda",
        "adaptive_ucl_cda",
        "adaptive_ucl_cda_v2",
        "adaptive_ucl_cda_v3",
        "corn_adaptive_tau_safe",
    }:
        raise ValueError(f"Unsupported loss_mode='{loss_mode}' for oracle_mcd.")
    return config


def load_and_override_config(args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(read_yaml(args.config))
    config = ensure_training_defaults(config, default_loss_mode="corn_adaptive_tau_safe")

    if args.seed is not None:
        config["seed"] = args.seed

    data_cfg = config["data"]
    if args.root_dir is not None:
        data_cfg["root_dir"] = args.root_dir
    if args.train_list is not None:
        data_cfg["train_list"] = args.train_list
    if args.val_list is not None:
        data_cfg["val_list"] = args.val_list
    if args.instance_source is not None:
        data_cfg["instance_source"] = args.instance_source
    if args.allow_tier3:
        data_cfg["allow_tier3"] = True
    if args.image_size is not None:
        data_cfg["image_size"] = args.image_size
    if args.context_ratio is not None:
        data_cfg["context_ratio"] = args.context_ratio
    if args.min_polygon_area is not None:
        data_cfg["min_polygon_area"] = args.min_polygon_area
    if args.min_mask_pixels is not None:
        data_cfg["min_mask_pixels"] = args.min_mask_pixels
    if args.max_out_of_bound_ratio is not None:
        data_cfg["max_out_of_bound_ratio"] = args.max_out_of_bound_ratio
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = args.cache_dir

    aug_cfg = config.setdefault("augmentation", {})
    for arg_name in [
        "context_dropout_prob",
        "context_blur_prob",
        "context_grayscale_prob",
        "context_noise_prob",
        "context_mix_prob",
        "context_edge_soften_pixels",
        "context_dilate_pixels",
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            aug_cfg[arg_name] = value
    for arg_name in [
        "context_apply_to_pre_and_post_independently",
        "context_preserve_instance_strictly",
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            aug_cfg[arg_name] = value

    model_cfg = config["model"]
    if args.model_type is not None:
        model_cfg["model_type"] = args.model_type
    if args.head_type is not None:
        model_cfg["head_type"] = args.head_type
    if args.dropout is not None:
        model_cfg["dropout"] = args.dropout
    if args.no_pretrained:
        model_cfg["pretrained"] = False

    train_cfg = config["training"]
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        train_cfg["warmup_epochs"] = args.warmup_epochs
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.no_amp:
        train_cfg["amp"] = False
    if args.label_smoothing is not None:
        train_cfg["label_smoothing"] = args.label_smoothing
    if args.loss_mode is not None:
        train_cfg["loss_mode"] = args.loss_mode
    if args.lambda_ord is not None:
        train_cfg["lambda_ord"] = args.lambda_ord
    if args.lambda_uni is not None:
        train_cfg["lambda_uni"] = args.lambda_uni
    if args.lambda_conc is not None:
        train_cfg["lambda_conc"] = args.lambda_conc
    if args.lambda_gap_reg is not None:
        train_cfg["lambda_gap_reg"] = args.lambda_gap_reg
    if args.lambda_tau is not None:
        train_cfg["lambda_tau"] = args.lambda_tau
    if args.lambda_tau_mean is not None:
        train_cfg["lambda_tau_mean"] = args.lambda_tau_mean
    if args.lambda_tau_diff is not None:
        train_cfg["lambda_tau_diff"] = args.lambda_tau_diff
    if args.lambda_tau_rank is not None:
        train_cfg["lambda_tau_rank"] = args.lambda_tau_rank
    if args.lambda_raw_tau_diff is not None:
        train_cfg["lambda_raw_tau_diff"] = args.lambda_raw_tau_diff
    if args.lambda_raw_tau_center is not None:
        train_cfg["lambda_raw_tau_center"] = args.lambda_raw_tau_center
    if args.lambda_raw_tau_bound is not None:
        train_cfg["lambda_raw_tau_bound"] = args.lambda_raw_tau_bound
    if args.lambda_corn_soft is not None:
        train_cfg["lambda_corn_soft"] = args.lambda_corn_soft
    if args.fixed_cda_alpha is not None:
        train_cfg["fixed_cda_alpha"] = args.fixed_cda_alpha
    if args.tau_init is not None:
        train_cfg["tau_init"] = args.tau_init
    if args.tau_min is not None:
        train_cfg["tau_min"] = args.tau_min
    if args.tau_max is not None:
        train_cfg["tau_max"] = args.tau_max
    if args.tau_base is not None:
        train_cfg["tau_base"] = args.tau_base
    if args.delta_scale is not None:
        train_cfg["delta_scale"] = args.delta_scale
    if args.tau_parameterization is not None:
        train_cfg["tau_parameterization"] = args.tau_parameterization
    if args.tau_logit_scale is not None:
        train_cfg["tau_logit_scale"] = args.tau_logit_scale
    if args.tau_target is not None:
        train_cfg["tau_target"] = args.tau_target
    if args.tau_easy is not None:
        train_cfg["tau_easy"] = args.tau_easy
    if args.tau_hard is not None:
        train_cfg["tau_hard"] = args.tau_hard
    if args.tau_variance_weight is not None:
        train_cfg["tau_variance_weight"] = args.tau_variance_weight
    if args.tau_std_floor is not None:
        train_cfg["tau_std_floor"] = args.tau_std_floor
    if args.tau_rank_margin_difficulty is not None:
        train_cfg["tau_rank_margin_difficulty"] = args.tau_rank_margin_difficulty
    if args.tau_rank_margin_value is not None:
        train_cfg["tau_rank_margin_value"] = args.tau_rank_margin_value
    if args.raw_tau_soft_margin is not None:
        train_cfg["raw_tau_soft_margin"] = args.raw_tau_soft_margin
    if args.tau_freeze_epochs is not None:
        train_cfg["tau_freeze_epochs"] = args.tau_freeze_epochs
    if args.tau_warmup_value is not None:
        train_cfg["tau_warmup_value"] = args.tau_warmup_value
    if args.corn_soft_start_epoch is not None:
        train_cfg["corn_soft_start_epoch"] = args.corn_soft_start_epoch
    if args.ambiguity_lr_scale is not None:
        train_cfg["ambiguity_lr_scale"] = args.ambiguity_lr_scale
    if args.concentration_margin is not None:
        train_cfg["concentration_margin"] = args.concentration_margin
    if args.use_focal:
        train_cfg["use_focal"] = True
    if args.focal_gamma is not None:
        train_cfg["focal_gamma"] = args.focal_gamma
    if args.early_stop_patience is not None:
        train_cfg["early_stop_patience"] = args.early_stop_patience
    if args.save_topk is not None:
        train_cfg["save_topk"] = args.save_topk
    for arg_name in [
        "dual_view_enabled",
        "enable_ordinal_contrastive",
        "enable_corn_soft_emd",
        "enable_decoded_unimodal_reg",
        "stage2_head_rebalance_enabled",
        "stage2_freeze_trunk",
        "stage2_freeze_ambiguity",
        "stage2_freeze_ordinal",
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            train_cfg[arg_name] = value
    for arg_name in [
        "lambda_view_consistency",
        "lambda_view_severity_consistency",
        "dual_view_stronger_context_prob",
        "lambda_contrastive",
        "contrastive_temperature",
        "contrastive_proj_dim",
        "contrastive_hidden_features",
        "contrastive_dropout",
        "contrastive_minor_major_weight",
        "contrastive_adjacent_weight",
        "contrastive_far_weight",
        "contrastive_min_positives",
        "contrastive_loss_warmup_epochs",
        "lambda_corn_soft_emd",
        "lambda_decoded_unimodal",
        "decoded_unimodal_margin",
        "stage2_epochs",
        "stage2_lr",
        "stage2_weight_decay",
        "stage2_early_stop_patience",
        "stage2_sampler_mode",
        "stage2_focus_minor_major_weight",
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            train_cfg[arg_name] = value

    output_cfg = config["output"]
    if args.output_root is not None:
        output_cfg["output_root"] = args.output_root
    if args.exp_name is not None:
        output_cfg["exp_name"] = args.exp_name

    eval_cfg = config.setdefault("evaluation", {})
    for arg_name in [
        "enable_threshold_calibration",
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            eval_cfg[arg_name] = value
    for arg_name in ["calibration_type", "calibration_max_iter", "calibration_lr"]:
        value = getattr(args, arg_name)
        if value is not None:
            eval_cfg[arg_name] = value

    return harmonize_model_and_loss_config(config)


def make_run_dir(config: dict[str, Any]) -> Path:
    return (
        Path(config["output"]["output_root"])
        / config["output"]["exp_name"]
        / config["model"]["model_type"]
        / config["training"]["loss_mode"]
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_epoch_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        multiplier = float(epoch + 1) / float(warmup_epochs)
    else:
        denom = max(total_epochs - warmup_epochs, 1)
        progress = float(epoch - warmup_epochs + 1) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

    current_lr = base_lr * multiplier
    for param_group in optimizer.param_groups:
        lr_scale = float(param_group.get("lr_scale", 1.0))
        param_group["lr"] = current_lr * lr_scale
    return current_lr


def make_dataloaders(
    config: dict[str, Any],
    *,
    train_sampler: torch.utils.data.Sampler[int] | None = None,
) -> tuple[DataLoader, DataLoader, XBDOracleInstanceDamageDataset, XBDOracleInstanceDamageDataset]:
    train_dataset = XBDOracleInstanceDamageDataset(
        config=config,
        split_name="train",
        list_path=config["data"]["train_list"],
        is_train=True,
    )
    val_dataset = XBDOracleInstanceDamageDataset(
        config=config,
        split_name=config["evaluation"]["split_name"],
        list_path=config["data"]["val_list"],
        is_train=False,
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    common_loader_kwargs = {
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": oracle_instance_collate_fn,
        "worker_init_fn": seed_worker,
        "persistent_workers": int(config["training"]["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def build_optimizer(model: torch.nn.Module, criterion: DamageLossModule, config: dict[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    loss_mode = str(config["training"]["loss_mode"])

    ambiguity_params: list[torch.nn.Parameter] = []
    if hasattr(model, "get_ambiguity_head_parameters"):
        ambiguity_params = [param for param in model.get_ambiguity_head_parameters() if param.requires_grad]
    classifier_params: list[torch.nn.Parameter] = []
    if hasattr(model, "get_classifier_head_parameters"):
        classifier_params = [param for param in model.get_classifier_head_parameters() if param.requires_grad]
    trunk_params: list[torch.nn.Parameter] = []
    if hasattr(model, "get_trunk_parameters"):
        trunk_params = [param for param in model.get_trunk_parameters() if param.requires_grad]

    ambiguity_param_ids = {id(param) for param in ambiguity_params}
    classifier_param_ids = {id(param) for param in classifier_params}
    if not trunk_params:
        trunk_params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) not in ambiguity_param_ids and id(param) not in classifier_param_ids
        ]

    if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}:
        ordinal_params = [param for param in criterion.get_gap_parameters() if param.requires_grad]
        ambiguity_lr_scale = float(config["training"].get("ambiguity_lr_scale", 0.30))
    elif loss_mode == "adaptive_ucl_cda_v2":
        ordinal_params = [param for param in criterion.get_gap_parameters() if param.requires_grad]
        ordinal_param_ids = {id(param) for param in ordinal_params}
        ordinal_params.extend(
            [
                param
                for param in criterion.get_non_gap_trainable_parameters()
                if param.requires_grad and id(param) not in ordinal_param_ids
            ]
        )
        ambiguity_lr_scale = float(config["training"].get("ambiguity_lr_scale", 0.25))
    else:
        ordinal_params = [param for param in criterion.parameters() if param.requires_grad]
        ambiguity_lr_scale = float(config["training"].get("ambiguity_lr_scale", 0.5))

    param_groups: list[dict[str, Any]] = [
        {
            "params": trunk_params,
            "weight_decay": weight_decay,
            "lr_scale": 1.0,
            "group_name": "trunk",
        }
    ]
    if classifier_params:
        param_groups.append(
            {
                "params": classifier_params,
                "weight_decay": weight_decay,
                "lr_scale": 1.0,
                "group_name": "corn_head" if str(config["model"].get("head_type", "standard")) == "corn" else "classifier_head",
            }
        )
    if ordinal_params:
        param_groups.append(
            {
                "params": ordinal_params,
                "weight_decay": 0.0,
                "lr_scale": 0.5,
                "group_name": "ordinal",
            }
        )
    if ambiguity_params:
        param_groups.append(
            {
                "params": ambiguity_params,
                "weight_decay": 0.0,
                "lr_scale": ambiguity_lr_scale,
                "group_name": "ambiguity",
            }
        )

    optimizer_name = str(config["training"]["optimizer"]).lower()
    if optimizer_name != "adamw":
        raise ValueError("Current implementation supports optimizer=adamw only.")
    return torch.optim.AdamW(param_groups, lr=base_lr)


def build_loss_module_from_config(
    config: dict[str, Any],
    class_weights: torch.Tensor,
    *,
    device: torch.device,
) -> DamageLossModule:
    return build_loss_function(
        class_weights=class_weights,
        loss_mode=str(config["training"]["loss_mode"]),
        label_smoothing=float(config["training"]["label_smoothing"]),
        lambda_ord=float(config["training"]["lambda_ord"]),
        lambda_uni=float(config["training"].get("lambda_uni", 0.10)),
        lambda_conc=float(config["training"].get("lambda_conc", 0.01)),
        lambda_gap_reg=float(config["training"]["lambda_gap_reg"]),
        lambda_tau=float(config["training"].get("lambda_tau", 0.01)),
        lambda_tau_mean=float(config["training"].get("lambda_tau_mean", 0.05)),
        lambda_tau_diff=float(config["training"].get("lambda_tau_diff", 0.20)),
        lambda_tau_rank=float(config["training"].get("lambda_tau_rank", 0.05)),
        lambda_raw_tau_diff=float(config["training"].get("lambda_raw_tau_diff", 0.10)),
        lambda_raw_tau_center=float(config["training"].get("lambda_raw_tau_center", 0.02)),
        lambda_raw_tau_bound=float(config["training"].get("lambda_raw_tau_bound", 0.02)),
        lambda_corn_soft=float(config["training"].get("lambda_corn_soft", 0.03)),
        enable_corn_soft_emd=bool(config["training"].get("enable_corn_soft_emd", False)),
        lambda_corn_soft_emd=float(config["training"].get("lambda_corn_soft_emd", 0.02)),
        enable_decoded_unimodal_reg=bool(config["training"].get("enable_decoded_unimodal_reg", False)),
        lambda_decoded_unimodal=float(config["training"].get("lambda_decoded_unimodal", 0.01)),
        decoded_unimodal_margin=float(config["training"].get("decoded_unimodal_margin", 0.0)),
        fixed_cda_alpha=float(config["training"]["fixed_cda_alpha"]),
        tau_init=float(config["training"]["tau_init"]),
        tau_min=float(config["training"].get("tau_min", 0.10)),
        tau_max=float(config["training"].get("tau_max", 0.60)),
        tau_base=float(config["training"].get("tau_base", config["training"].get("tau_target", 0.22))),
        delta_scale=float(config["training"].get("delta_scale", 0.12)),
        tau_parameterization=str(
            config["training"].get(
                "tau_parameterization",
                "bounded_sigmoid"
                if str(config["training"]["loss_mode"]) in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
                else "sigmoid",
            )
        ),
        tau_logit_scale=float(config["training"].get("tau_logit_scale", 2.0)),
        tau_target=float(config["training"].get("tau_target", 0.22)),
        tau_easy=float(config["training"].get("tau_easy", 0.16)),
        tau_hard=float(config["training"].get("tau_hard", 0.32)),
        tau_variance_weight=float(config["training"].get("tau_variance_weight", 0.01)),
        tau_std_floor=float(config["training"].get("tau_std_floor", 0.03)),
        tau_rank_margin_difficulty=float(config["training"].get("tau_rank_margin_difficulty", 0.10)),
        tau_rank_margin_value=float(config["training"].get("tau_rank_margin_value", 0.01)),
        raw_tau_soft_margin=float(config["training"].get("raw_tau_soft_margin", 1.5)),
        concentration_margin=float(config["training"].get("concentration_margin", 0.05)),
        lambda_aux=float(config["training"].get("lambda_aux", 0.2)),
        use_focal=bool(config["training"].get("use_focal", False)),
        focal_gamma=float(config["training"].get("focal_gamma", 2.0)),
        num_classes=len(CLASS_NAMES),
    ).to(device)


def unpack_model_outputs(model_output: Any) -> dict[str, Any]:
    if isinstance(model_output, torch.Tensor):
        return {
            "logits": model_output,
            "corn_logits": None,
            "aux_logits": None,
            "pooled_feature": None,
            "contrastive_embedding": None,
            "tau": None,
            "raw_tau": None,
            "raw_delta_tau": None,
            "head_type": "standard",
            "ambiguity_input_detached": False,
        }
    if not isinstance(model_output, dict) or "logits" not in model_output:
        raise TypeError("Model forward must return a logits tensor or a dict containing 'logits'.")
    return {
        "logits": model_output["logits"],
        "corn_logits": model_output.get("corn_logits"),
        "aux_logits": model_output.get("aux_logits"),
        "pooled_feature": model_output.get("pooled_feature"),
        "contrastive_embedding": model_output.get("contrastive_embedding"),
        "tau": model_output.get("tau"),
        "raw_tau": model_output.get("raw_tau"),
        "raw_delta_tau": model_output.get("raw_delta_tau"),
        "head_type": model_output.get("head_type", "standard"),
        "ambiguity_input_detached": bool(model_output.get("ambiguity_input_detached", False)),
    }


def resolve_class_probabilities(
    logits: torch.Tensor,
    *,
    head_type: str,
    class_probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    if class_probabilities is not None:
        return class_probabilities
    if head_type == "corn" or logits.size(1) == len(CLASS_NAMES) - 1:
        return CORNLoss.logits_to_class_probabilities(logits)
    return logits.softmax(dim=1)


def aggregate_augmentation_stats(stats_list: list[dict[str, Any]] | None) -> dict[str, float] | None:
    if not stats_list:
        return None
    aggregated: dict[str, list[float]] = {}
    for stats in stats_list:
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                aggregated.setdefault(key, []).append(float(value))
    if not aggregated:
        return None
    return {
        key: float(sum(values) / max(len(values), 1))
        for key, values in aggregated.items()
    }


def set_requires_grad(parameters: list[torch.nn.Parameter], trainable: bool) -> None:
    for param in parameters:
        param.requires_grad = trainable


def build_contrastive_loss_module(config: dict[str, Any], device: torch.device) -> OrdinalSupConLoss | None:
    if not bool(config["training"].get("enable_ordinal_contrastive", False)):
        return None
    return OrdinalSupConLoss(
        temperature=float(config["training"].get("contrastive_temperature", 0.10)),
        adjacent_weight=float(config["training"].get("contrastive_adjacent_weight", 1.0)),
        far_weight=float(config["training"].get("contrastive_far_weight", 0.5)),
        minor_major_weight=float(config["training"].get("contrastive_minor_major_weight", 1.75)),
        min_positives=int(config["training"].get("contrastive_min_positives", 1)),
    ).to(device)


def summarize_distribution_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    value_tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(value_tensor, torch.tensor([0.10, 0.50, 0.90], dtype=torch.float32))
    std = value_tensor.std(unbiased=False) if value_tensor.numel() > 1 else torch.tensor(0.0, dtype=torch.float32)
    return {
        "mean": float(value_tensor.mean().item()),
        "std": float(std.item()),
        "min": float(value_tensor.min().item()),
        "max": float(value_tensor.max().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def summarize_tau_values(values: list[float]) -> dict[str, float] | None:
    return summarize_distribution_values(values)


def summarize_mean_std_min_max(values: list[float]) -> dict[str, float] | None:
    summary = summarize_distribution_values(values)
    if summary is None:
        return None
    return {
        "mean": summary["mean"],
        "std": summary["std"],
        "min": summary["min"],
        "max": summary["max"],
    }


def compute_scalar_correlation(first: list[float], second: list[float]) -> float | None:
    if not first or not second or len(first) != len(second):
        return None
    first_tensor = torch.tensor(first, dtype=torch.float32)
    second_tensor = torch.tensor(second, dtype=torch.float32)
    if first_tensor.numel() < 2:
        return 0.0
    first_centered = first_tensor - first_tensor.mean()
    second_centered = second_tensor - second_tensor.mean()
    denominator = torch.sqrt(first_centered.pow(2).sum() * second_centered.pow(2).sum()).clamp_min(1e-12)
    return float(((first_centered * second_centered).sum() / denominator).item())


def compute_boundary_ratio(values: list[float], boundary: float, *, is_min: bool, eps: float = 1e-6) -> float | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    if is_min:
        hits = tensor <= float(boundary) + float(eps)
    else:
        hits = tensor >= float(boundary) - float(eps)
    return float(hits.float().mean().item())


def compute_centered_edge_ratio(
    values: list[float],
    center: float | None,
    max_offset: float | None,
    *,
    threshold_ratio: float = 0.95,
) -> float | None:
    if not values or center is None or max_offset is None or float(max_offset) <= 0.0:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    hits = (tensor - float(center)).abs() >= (float(threshold_ratio) * float(max_offset))
    return float(hits.float().mean().item())


def summarize_tau_by_difficulty_bin(
    tau_values: list[float],
    difficulty_values: list[float],
    num_bins: int = 4,
) -> list[dict[str, Any]]:
    records = [
        {
            "bin_index": int(bin_index),
            "count": 0,
            "difficulty_min": None,
            "difficulty_max": None,
            "difficulty_mean": None,
            "tau_mean": None,
            "tau_std": None,
        }
        for bin_index in range(num_bins)
    ]
    if not tau_values or not difficulty_values or len(tau_values) != len(difficulty_values):
        return records

    tau_tensor = torch.tensor(tau_values, dtype=torch.float32)
    difficulty_tensor = torch.tensor(difficulty_values, dtype=torch.float32)
    sorted_indices = torch.argsort(difficulty_tensor)
    sample_count = int(sorted_indices.numel())
    for bin_index in range(num_bins):
        start = (bin_index * sample_count) // num_bins
        end = ((bin_index + 1) * sample_count) // num_bins
        if end <= start:
            continue
        chunk_indices = sorted_indices[start:end]
        chunk_tau = tau_tensor[chunk_indices]
        chunk_difficulty = difficulty_tensor[chunk_indices]
        chunk_tau_std = chunk_tau.std(unbiased=False) if chunk_tau.numel() > 1 else chunk_tau.new_tensor(0.0)
        records[bin_index] = {
            "bin_index": int(bin_index),
            "count": int(chunk_indices.numel()),
            "difficulty_min": float(chunk_difficulty.min().item()),
            "difficulty_max": float(chunk_difficulty.max().item()),
            "difficulty_mean": float(chunk_difficulty.mean().item()),
            "tau_mean": float(chunk_tau.mean().item()),
            "tau_std": float(chunk_tau_std.item()),
        }
    return records


def summarize_tau_by_class(
    tau_values: list[float],
    labels: list[int],
    class_names: list[str],
) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {
        class_name: {
            "mean": None,
            "std": None,
            "count": 0,
        }
        for class_name in class_names
    }
    if not tau_values or not labels or len(tau_values) != len(labels):
        return summary

    tau_tensor = torch.tensor(tau_values, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    for class_index, class_name in enumerate(class_names):
        mask = label_tensor == int(class_index)
        if not mask.any():
            continue
        class_tau = tau_tensor[mask]
        class_std = class_tau.std(unbiased=False) if class_tau.numel() > 1 else class_tau.new_tensor(0.0)
        summary[class_name] = {
            "mean": float(class_tau.mean().item()),
            "std": float(class_std.item()),
            "count": int(class_tau.numel()),
        }
    return summary


def set_ambiguity_head_trainable(model: torch.nn.Module, trainable: bool) -> None:
    ambiguity_head = getattr(model, "ambiguity_head", None)
    if ambiguity_head is None:
        return
    for param in ambiguity_head.parameters():
        param.requires_grad = trainable


def resolve_tau_phase(config: dict[str, Any], epoch_index: int) -> dict[str, Any]:
    loss_mode = str(config["training"]["loss_mode"])
    if loss_mode not in {"adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}:
        return {
            "phase_name": "disabled",
            "use_fixed_tau": False,
            "tau_override_value": None,
            "ambiguity_trainable": True,
        }

    freeze_epochs = int(config["training"].get("tau_freeze_epochs", 1 if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"} else 3))
    warmup_value = float(config["training"].get("tau_warmup_value", 0.22 if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"} else 0.25))
    if epoch_index < freeze_epochs:
        return {
            "phase_name": "warmup_fixed_tau",
            "use_fixed_tau": True,
            "tau_override_value": warmup_value,
            "ambiguity_trainable": False,
        }
    return {
        "phase_name": "adaptive_tau",
        "use_fixed_tau": False,
        "tau_override_value": None,
        "ambiguity_trainable": True,
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DamageLossModule,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    epoch_index: int,
    stage_name: str = "stage1",
    tau_override_value: float | None = None,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler | None = None,
    corn_soft_enabled: bool = False,
    corn_head_parameters: list[torch.nn.Parameter] | None = None,
    contrastive_loss_module: OrdinalSupConLoss | None = None,
    lambda_contrastive: float = 0.0,
    contrastive_loss_warmup_epochs: int = 0,
    lambda_view_consistency: float = 0.0,
    lambda_view_severity_consistency: float = 0.0,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    tau_values: list[float] = []
    difficulty_values: list[float] = []
    raw_tau_values: list[float] = []
    raw_tau_ref_values: list[float] = []
    paired_tau_values: list[float] = []
    paired_difficulty_values: list[float] = []
    tau_minus_tau_ref_values: list[float] = []
    raw_tau_minus_ref_values: list[float] = []
    tau_class_labels: list[int] = []
    corn_max_confidence_values: list[float] = []
    corn_entropy_values: list[float] = []
    corn_gt_probability_values: list[float] = []
    loss_terms = {
        "loss_ce": 0.0,
        "loss_ord": 0.0,
        "loss_corn_main": 0.0,
        "loss_corn_soft": 0.0,
        "loss_gap_reg": 0.0,
        "loss_aux": 0.0,
        "loss_uni": 0.0,
        "loss_conc": 0.0,
        "loss_tau_reg": 0.0,
        "loss_tau_mean": 0.0,
        "loss_tau_var": 0.0,
        "loss_tau_diff": 0.0,
        "loss_tau_rank": 0.0,
        "loss_raw_tau_diff": 0.0,
        "loss_raw_tau_center": 0.0,
        "loss_raw_tau_bound": 0.0,
        "loss_corn_soft_emd": 0.0,
        "loss_corn_unimodal": 0.0,
        "loss_contrastive": 0.0,
        "loss_view_consistency": 0.0,
        "loss_view_severity_consistency": 0.0,
    }
    soft_target_sums = {
        "entropy_mean": 0.0,
        "gt_mass_mean": 0.0,
    }
    adaptive_soft_target_detached = False
    ambiguity_input_detached = False
    context_view1_sums: dict[str, float] = {}
    context_view2_sums: dict[str, float] = {}
    corn_task_loss_sums = [0.0 for _ in range(len(CLASS_NAMES) - 1)]
    corn_task_count_sums = [0.0 for _ in range(len(CLASS_NAMES) - 1)]
    contrastive_active = bool(
        is_train
        and contrastive_loss_module is not None
        and float(lambda_contrastive) > 0.0
        and (epoch_index + 1) >= int(contrastive_loss_warmup_epochs)
    )
    dual_view_active = bool(
        is_train and (float(lambda_view_consistency) > 0.0 or float(lambda_view_severity_consistency) > 0.0)
    )

    progress = tqdm(loader, leave=False, desc="train" if is_train else "val")
    for batch in progress:
        pre_image = batch["pre_image"].to(device, non_blocking=True)
        post_image = batch["post_image"].to(device, non_blocking=True)
        instance_mask = batch["instance_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
        with context:
            model_outputs = unpack_model_outputs(model(pre_image, post_image, instance_mask))
            logits = model_outputs["corn_logits"] if model_outputs.get("corn_logits") is not None else model_outputs["logits"]
            tau_used = model_outputs["tau"]
            if tau_override_value is not None:
                tau_used = torch.full(
                    (labels.size(0),),
                    float(tau_override_value),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            loss_outputs = criterion(
                logits,
                labels,
                aux_logits=model_outputs["aux_logits"],
                sample_tau=tau_used,
                raw_tau=model_outputs["raw_tau"],
                corn_soft_enabled=corn_soft_enabled,
            )
            loss = loss_outputs["loss"]
            loss_backward = loss_outputs.get("loss_backward", loss)
            positions = criterion.get_current_positions().to(device=logits.device, dtype=logits.dtype)
            class_probabilities = resolve_class_probabilities(
                logits,
                head_type=str(model_outputs.get("head_type", "standard")),
                class_probabilities=loss_outputs.get("class_probabilities"),
            )

            view2_model_outputs = None
            view2_probabilities = None
            loss_view_consistency = logits.new_tensor(0.0)
            loss_view_severity_consistency = logits.new_tensor(0.0)
            if dual_view_active and "view2_pre_image" in batch:
                # Dual-view consistency reuses the same oracle instance crop and mask, then changes context exposure only.
                view2_pre_image = batch["view2_pre_image"].to(device, non_blocking=True)
                view2_post_image = batch["view2_post_image"].to(device, non_blocking=True)
                view2_instance_mask = batch["view2_instance_mask"].to(device, non_blocking=True)
                view2_model_outputs = unpack_model_outputs(model(view2_pre_image, view2_post_image, view2_instance_mask))
                view2_logits = (
                    view2_model_outputs["corn_logits"]
                    if view2_model_outputs.get("corn_logits") is not None
                    else view2_model_outputs["logits"]
                )
                view2_probabilities = resolve_class_probabilities(
                    view2_logits,
                    head_type=str(view2_model_outputs.get("head_type", "standard")),
                )
                if float(lambda_view_consistency) > 0.0:
                    loss_view_consistency = compute_js_divergence_from_probabilities(class_probabilities, view2_probabilities)
                if float(lambda_view_severity_consistency) > 0.0:
                    severity_view1 = compute_expected_severity_from_probabilities(class_probabilities, positions)
                    severity_view2 = compute_expected_severity_from_probabilities(view2_probabilities, positions)
                    loss_view_severity_consistency = F.smooth_l1_loss(severity_view1, severity_view2)
                loss = (
                    loss
                    + (float(lambda_view_consistency) * loss_view_consistency)
                    + (float(lambda_view_severity_consistency) * loss_view_severity_consistency)
                )
                loss_backward = (
                    loss_backward
                    + (float(lambda_view_consistency) * loss_view_consistency)
                    + (float(lambda_view_severity_consistency) * loss_view_severity_consistency)
                )

            loss_contrastive = logits.new_tensor(0.0)
            if contrastive_active and contrastive_loss_module is not None and model_outputs.get("contrastive_embedding") is not None:
                contrastive_embeddings = [model_outputs["contrastive_embedding"]]
                contrastive_labels = [labels]
                if view2_model_outputs is not None and view2_model_outputs.get("contrastive_embedding") is not None:
                    contrastive_embeddings.append(view2_model_outputs["contrastive_embedding"])
                    contrastive_labels.append(labels)
                merged_embeddings = torch.cat(contrastive_embeddings, dim=0)
                merged_labels = torch.cat(contrastive_labels, dim=0)
                loss_contrastive = contrastive_loss_module(merged_embeddings, merged_labels)
                loss = loss + (float(lambda_contrastive) * loss_contrastive)
                loss_backward = loss_backward + (float(lambda_contrastive) * loss_contrastive)

            loss_outputs["loss_contrastive"] = loss_contrastive
            loss_outputs["loss_view_consistency"] = loss_view_consistency
            loss_outputs["loss_view_severity_consistency"] = loss_view_severity_consistency

        non_finite_terms: list[str] = []
        for name, value in (
            ("logits", logits),
            ("tau_used", tau_used),
            ("raw_tau", model_outputs.get("raw_tau")),
            ("loss", loss),
            ("loss_backward", loss_backward),
            ("loss_corn_main", loss_outputs.get("loss_corn_main")),
            ("loss_corn_soft", loss_outputs.get("loss_corn_soft")),
            ("loss_corn_soft_emd", loss_outputs.get("loss_corn_soft_emd")),
            ("loss_corn_unimodal", loss_outputs.get("loss_corn_unimodal")),
            ("loss_contrastive", loss_outputs.get("loss_contrastive")),
            ("loss_view_consistency", loss_outputs.get("loss_view_consistency")),
        ):
            if value is None:
                continue
            if not torch.isfinite(value.detach().float()).all():
                non_finite_terms.append(name)
        if non_finite_terms:
            raise FloatingPointError(
                "Non-finite values detected during "
                f"{'training' if is_train else 'validation'}: {', '.join(non_finite_terms)} "
                f"(corn_soft_enabled={corn_soft_enabled}, batch_size={labels.size(0)})."
            )

        if is_train:
            assert optimizer is not None
            assert scaler is not None
            head_only_soft_grads: tuple[torch.Tensor | None, ...] | None = None
            if (
                criterion.loss_mode == "corn_adaptive_tau_safe"
                and corn_soft_enabled
                and corn_head_parameters
                and float(loss_outputs.get("loss_safe_head_only", 0.0).detach().item()) > 0.0
            ):
                # Safe decoded-space teacher losses stay head-only: they regularize CORN thresholds
                # without letting the soft branch dominate the shared trunk or ambiguity pathway.
                head_only_soft_grads = torch.autograd.grad(
                    scaler.scale(loss_outputs["loss_safe_head_only"]),
                    corn_head_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            scaler.scale(loss_backward).backward()
            if head_only_soft_grads is not None:
                for param, grad in zip(corn_head_parameters, head_only_soft_grads):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad.detach()
                    else:
                        param.grad.add_(grad.detach())
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

        aggregated_view1_context = aggregate_augmentation_stats(batch.get("augmentation_stats"))
        if aggregated_view1_context is not None:
            for key, value in aggregated_view1_context.items():
                context_view1_sums[key] = context_view1_sums.get(key, 0.0) + (float(value) * batch_size)
        aggregated_view2_context = aggregate_augmentation_stats(batch.get("view2_augmentation_stats"))
        if aggregated_view2_context is not None:
            for key, value in aggregated_view2_context.items():
                context_view2_sums[key] = context_view2_sums.get(key, 0.0) + (float(value) * batch_size)

        for key in loss_terms:
            loss_terms[key] += float(loss_outputs[key].detach().item()) * batch_size
        if loss_outputs.get("soft_target_stats") is not None:
            for key in soft_target_sums:
                soft_target_sums[key] += float(loss_outputs["soft_target_stats"][key].detach().item()) * batch_size
        adaptive_soft_target_detached = adaptive_soft_target_detached or bool(
            loss_outputs.get("adaptive_soft_target_detached", False)
        )
        ambiguity_input_detached = ambiguity_input_detached or bool(model_outputs.get("ambiguity_input_detached", False))
        batch_corn_task_losses = loss_outputs.get("corn_task_losses")
        batch_corn_task_counts = loss_outputs.get("corn_task_counts")
        if batch_corn_task_losses is not None and batch_corn_task_counts is not None:
            for index, (task_loss, task_count) in enumerate(
                zip(batch_corn_task_losses.detach().cpu().tolist(), batch_corn_task_counts.detach().cpu().tolist())
            ):
                corn_task_loss_sums[index] += float(task_loss) * float(task_count)
                corn_task_count_sums[index] += float(task_count)

        class_probabilities = class_probabilities.detach()
        predictions = class_probabilities.argmax(dim=1)
        max_confidence = class_probabilities.max(dim=1).values
        corn_max_confidence_values.extend(float(value) for value in max_confidence.cpu().tolist())
        safe_class_probabilities = class_probabilities.float().clamp_min(1e-8)
        corn_entropy = -(safe_class_probabilities * safe_class_probabilities.log()).sum(dim=1)
        corn_entropy_values.extend(float(value) for value in corn_entropy.cpu().tolist())
        gt_probability = class_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        corn_gt_probability_values.extend(float(value) for value in gt_probability.cpu().tolist())

        if tau_used is not None:
            tau_batch_values = [float(value) for value in tau_used.detach().float().cpu().tolist()]
            tau_values.extend(tau_batch_values)
            tau_class_labels.extend(labels.detach().cpu().tolist())
        raw_tau = model_outputs.get("raw_tau")
        if raw_tau is not None:
            raw_tau_values.extend([float(value) for value in raw_tau.detach().float().cpu().tolist()])
        batch_difficulty = loss_outputs.get("difficulty")
        if batch_difficulty is None:
            batch_difficulty = loss_outputs.get("difficulty_values")
        if batch_difficulty is not None:
            batch_difficulty_values = [float(value) for value in batch_difficulty.detach().float().cpu().tolist()]
            difficulty_values.extend(batch_difficulty_values)
            if tau_used is not None and len(batch_difficulty_values) == labels.size(0):
                paired_tau_values.extend([float(value) for value in tau_used.detach().float().cpu().tolist()])
                paired_difficulty_values.extend(batch_difficulty_values)
        batch_tau_ref = loss_outputs.get("tau_ref")
        if batch_tau_ref is not None:
            batch_tau_ref_values = [float(value) for value in batch_tau_ref.detach().float().cpu().tolist()]
            if tau_used is not None and len(batch_tau_ref_values) == labels.size(0):
                tau_minus_tau_ref_values.extend(
                    [
                        float(value)
                        for value in (tau_used.detach().float().cpu() - batch_tau_ref.detach().float().cpu()).tolist()
                    ]
                )
        batch_raw_tau_ref = loss_outputs.get("raw_tau_ref")
        if batch_raw_tau_ref is not None:
            batch_raw_tau_ref_values = [float(value) for value in batch_raw_tau_ref.detach().float().cpu().tolist()]
            raw_tau_ref_values.extend(batch_raw_tau_ref_values)
            if raw_tau is not None and len(batch_raw_tau_ref_values) == labels.size(0):
                raw_tau_minus_ref_values.extend(
                    [
                        float(value)
                        for value in (raw_tau.detach().float().cpu() - batch_raw_tau_ref.detach().float().cpu()).tolist()
                    ]
                )

        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

        postfix = {
            "loss": f"{loss.detach().item():.4f}",
            "ce": f"{loss_outputs['loss_ce'].detach().item():.4f}",
            "ord": f"{loss_outputs['loss_ord'].detach().item():.4f}",
        }
        if loss_outputs["loss_corn_main"].detach().item() > 0:
            postfix["corn"] = f"{loss_outputs['loss_corn_main'].detach().item():.4f}"
        if criterion.loss_mode == "corn_adaptive_tau_safe":
            postfix["soft"] = f"{loss_outputs['loss_corn_soft'].detach().item():.4f}"
            postfix["soft_on"] = "1" if bool(loss_outputs.get("corn_soft_enabled")) else "0"
            if loss_outputs["loss_corn_soft_emd"].detach().item() > 0:
                postfix["soft_emd"] = f"{loss_outputs['loss_corn_soft_emd'].detach().item():.4f}"
            if loss_outputs["loss_corn_unimodal"].detach().item() > 0:
                postfix["soft_uni"] = f"{loss_outputs['loss_corn_unimodal'].detach().item():.4f}"
        if tau_used is not None:
            postfix["tau"] = f"{tau_used.detach().float().mean().item():.3f}"
        if loss_outputs["loss_view_consistency"].detach().item() > 0:
            postfix["view"] = f"{loss_outputs['loss_view_consistency'].detach().item():.4f}"
        if loss_outputs["loss_contrastive"].detach().item() > 0:
            postfix["ctr"] = f"{loss_outputs['loss_contrastive'].detach().item():.4f}"
        if loss_outputs["loss_uni"].detach().item() > 0 or loss_outputs["loss_conc"].detach().item() > 0:
            postfix["uni"] = f"{loss_outputs['loss_uni'].detach().item():.4f}"
            postfix["conc"] = f"{loss_outputs['loss_conc'].detach().item():.4f}"
        if loss_outputs["loss_tau_reg"].detach().item() > 0:
            postfix["tau_reg"] = f"{loss_outputs['loss_tau_reg'].detach().item():.4f}"
        if loss_outputs["loss_tau_var"].detach().item() > 0:
            postfix["tau_var"] = f"{loss_outputs['loss_tau_var'].detach().item():.4f}"
        if loss_outputs["loss_tau_diff"].detach().item() > 0:
            postfix["tau_diff"] = f"{loss_outputs['loss_tau_diff'].detach().item():.4f}"
        if loss_outputs["loss_tau_rank"].detach().item() > 0:
            postfix["tau_rank"] = f"{loss_outputs['loss_tau_rank'].detach().item():.4f}"
        if loss_outputs["loss_raw_tau_diff"].detach().item() > 0:
            postfix["raw_tau_diff"] = f"{loss_outputs['loss_raw_tau_diff'].detach().item():.4f}"
        if loss_outputs["loss_raw_tau_center"].detach().item() > 0:
            postfix["raw_tau_ctr"] = f"{loss_outputs['loss_raw_tau_center'].detach().item():.4f}"
        if loss_outputs["loss_raw_tau_bound"].detach().item() > 0:
            postfix["raw_tau_bnd"] = f"{loss_outputs['loss_raw_tau_bound'].detach().item():.4f}"
        progress.set_postfix(**postfix)

    metrics, _ = compute_classification_metrics(all_targets, all_predictions, CLASS_NAMES)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["loss_terms"] = {
        key: value / max(total_samples, 1)
        for key, value in loss_terms.items()
    }
    metrics["tau_stats"] = summarize_tau_values(tau_values)
    metrics["difficulty_stats"] = summarize_distribution_values(difficulty_values)
    metrics["corr_tau_difficulty"] = compute_scalar_correlation(tau_values, difficulty_values)
    metrics["tau_at_min_ratio"] = compute_boundary_ratio(tau_values, float(getattr(criterion, "tau_min", 0.0)), is_min=True)
    metrics["tau_at_max_ratio"] = compute_boundary_ratio(tau_values, float(getattr(criterion, "tau_max", 0.0)), is_min=False)
    metrics["raw_tau_stats"] = summarize_distribution_values(raw_tau_values)
    metrics["raw_tau_ref_stats"] = summarize_distribution_values(raw_tau_ref_values)
    metrics["corr_raw_tau_difficulty"] = (
        compute_scalar_correlation(raw_tau_values, difficulty_values)
        if getattr(criterion, "loss_mode", "") in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
        else None
    )
    metrics["raw_tau_edge_ratio"] = (
        compute_centered_edge_ratio(
            raw_tau_values,
            getattr(criterion, "raw_tau_center", None),
            getattr(criterion, "tau_logit_scale", None),
        )
        if (
            getattr(criterion, "loss_mode", "") in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
            and str(getattr(criterion, "tau_parameterization", "")) == "bounded_sigmoid"
        )
        else None
    )
    metrics["tau_minus_tau_ref_stats"] = summarize_mean_std_min_max(tau_minus_tau_ref_values)
    metrics["raw_tau_minus_ref_stats"] = summarize_mean_std_min_max(raw_tau_minus_ref_values)
    metrics["tau_by_difficulty_bin"] = summarize_tau_by_difficulty_bin(paired_tau_values, paired_difficulty_values, num_bins=4)
    metrics["tau_by_class"] = summarize_tau_by_class(tau_values, tau_class_labels, CLASS_NAMES)
    metrics["soft_target_stats"] = {
        key: value / max(total_samples, 1)
        for key, value in soft_target_sums.items()
    }
    metrics["corn_soft_enabled"] = bool(corn_soft_enabled and criterion.loss_mode == "corn_adaptive_tau_safe")
    metrics["corn_mode"] = criterion.loss_mode if criterion.loss_mode in {"corn", "corn_adaptive_tau_safe"} else None
    metrics["probs_from_corn_stats"] = (
        {
            "max_confidence": summarize_mean_std_min_max(corn_max_confidence_values),
            "entropy": summarize_mean_std_min_max(corn_entropy_values),
            "gt_probability": summarize_mean_std_min_max(corn_gt_probability_values),
        }
        if corn_max_confidence_values
        else None
    )
    metrics["corn_task_losses"] = [
        (corn_task_loss_sums[index] / max(corn_task_count_sums[index], 1.0))
        for index in range(len(corn_task_loss_sums))
    ]
    metrics["corn_task_counts"] = [float(value) for value in corn_task_count_sums]
    metrics["adaptive_soft_target_detached"] = adaptive_soft_target_detached
    metrics["ambiguity_input_detached"] = ambiguity_input_detached
    metrics["context_aug_usage_stats"] = {
        "view1": {
            key: value / max(total_samples, 1)
            for key, value in context_view1_sums.items()
        } if context_view1_sums else None,
        "view2": {
            key: value / max(total_samples, 1)
            for key, value in context_view2_sums.items()
        } if context_view2_sums else None,
    }
    metrics["dual_view_enabled"] = bool(dual_view_active)
    metrics["contrastive_enabled"] = bool(contrastive_active)
    metrics["stage_name"] = stage_name
    return metrics


def save_class_statistics(run_dir: Path, class_counts: list[int], class_weights: torch.Tensor) -> None:
    payload = {
        "class_names": CLASS_NAMES,
        "class_counts": {name: int(count) for name, count in zip(CLASS_NAMES, class_counts)},
        "class_weights": {name: float(weight) for name, weight in zip(CLASS_NAMES, class_weights.tolist())},
    }
    write_json(run_dir / "class_stats_train.json", payload)
    lines = ["Train class statistics"]
    for name in CLASS_NAMES:
        lines.append(
            f"- {name}: count={payload['class_counts'][name]} weight={payload['class_weights'][name]:.6f}"
        )
    write_text(run_dir / "class_stats_train.txt", "\n".join(lines) + "\n")


def make_ordinal_epoch_record(
    epoch: int,
    criterion: DamageLossModule,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    tau_phase: dict[str, Any],
) -> dict[str, Any]:
    ordinal_state = criterion.export_state(CLASS_NAMES)
    positions = ordinal_state["positions"]
    gaps = ordinal_state["gaps"]
    return {
        "epoch": int(epoch),
        "loss_mode": ordinal_state["loss_mode"],
        "tau_phase": tau_phase["phase_name"],
        "gap_01": float(gaps["gap_01"]),
        "gap_12": float(gaps["gap_12"]),
        "gap_23": float(gaps["gap_23"]),
        "s0": float(positions[0]),
        "s1": float(positions[1]),
        "s2": float(positions[2]),
        "s3": float(positions[3]),
        "positions_by_class": ordinal_state["positions_by_class"],
        "tau_reference": ordinal_state["tau"],
        "train_loss": float(train_metrics["loss"]),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "val_loss": float(val_metrics["loss"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
    }


def make_tau_epoch_record(
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    tau_phase: dict[str, Any],
) -> dict[str, Any]:
    train_tau = train_metrics.get("tau_stats") or {}
    val_tau = val_metrics.get("tau_stats") or {}
    train_difficulty = train_metrics.get("difficulty_stats") or {}
    val_difficulty = val_metrics.get("difficulty_stats") or {}
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "tau_phase": tau_phase["phase_name"],
        "tau_override_value": tau_phase["tau_override_value"],
        "train_corn_soft_enabled": train_metrics.get("corn_soft_enabled"),
        "val_corn_soft_enabled": val_metrics.get("corn_soft_enabled"),
    }
    for prefix, payload in [("train", train_tau), ("val", val_tau)]:
        for key in ["mean", "std", "min", "max", "p10", "p50", "p90"]:
            record[f"{prefix}_tau_{key}"] = payload.get(key)
    for prefix, payload in [("train", train_difficulty), ("val", val_difficulty)]:
        for key in ["mean", "std"]:
            record[f"{prefix}_difficulty_{key}"] = payload.get(key)
    for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
        record[f"{prefix}_tau_at_min_ratio"] = metrics.get("tau_at_min_ratio")
        record[f"{prefix}_tau_at_max_ratio"] = metrics.get("tau_at_max_ratio")

        raw_tau_stats = metrics.get("raw_tau_stats") or {}
        for key in ["mean", "std", "min", "max", "p10", "p50", "p90"]:
            record[f"{prefix}_raw_tau_{key}"] = raw_tau_stats.get(key)

        raw_tau_ref_stats = metrics.get("raw_tau_ref_stats") or {}
        for key in ["mean", "std", "min", "max", "p10", "p50", "p90"]:
            record[f"{prefix}_raw_tau_ref_{key}"] = raw_tau_ref_stats.get(key)

        tau_minus_tau_ref_stats = metrics.get("tau_minus_tau_ref_stats") or {}
        for key in ["mean", "std", "min", "max"]:
            record[f"{prefix}_tau_minus_tau_ref_{key}"] = tau_minus_tau_ref_stats.get(key)

        raw_tau_minus_ref_stats = metrics.get("raw_tau_minus_ref_stats") or {}
        for key in ["mean", "std", "min", "max"]:
            record[f"{prefix}_raw_tau_minus_ref_{key}"] = raw_tau_minus_ref_stats.get(key)

        record[f"{prefix}_raw_tau_edge_ratio"] = metrics.get("raw_tau_edge_ratio")
        record[f"{prefix}_tau_by_difficulty_bin"] = metrics.get("tau_by_difficulty_bin")
        record[f"{prefix}_tau_by_class"] = metrics.get("tau_by_class")
    record["train_corr_tau_difficulty"] = train_metrics.get("corr_tau_difficulty")
    record["train_corr_raw_tau_difficulty"] = train_metrics.get("corr_raw_tau_difficulty")
    record["val_corr_tau_difficulty"] = val_metrics.get("corr_tau_difficulty")
    record["val_corr_raw_tau_difficulty"] = val_metrics.get("corr_raw_tau_difficulty")
    return record


def write_ordinal_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    write_json(
        run_dir / "learned_ordinal_params.json",
        {
            "class_names": CLASS_NAMES,
            "history": history,
        },
    )

    csv_path = run_dir / "learned_ordinal_params.csv"
    fieldnames = [
        "epoch",
        "loss_mode",
        "tau_phase",
        "gap_01",
        "gap_12",
        "gap_23",
        "s0",
        "s1",
        "s2",
        "s3",
        "tau_reference",
        "train_loss",
        "train_macro_f1",
        "val_loss",
        "val_macro_f1",
    ]
    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_tau_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    write_json(run_dir / "learned_tau_stats.json", {"history": history})
    csv_path = run_dir / "learned_tau_stats.csv"
    fieldnames = [
        "epoch",
        "tau_phase",
        "tau_override_value",
        "train_corn_soft_enabled",
        "val_corn_soft_enabled",
        "train_tau_mean",
        "train_tau_std",
        "train_tau_min",
        "train_tau_max",
        "train_tau_p10",
        "train_tau_p50",
        "train_tau_p90",
        "val_tau_mean",
        "val_tau_std",
        "val_tau_min",
        "val_tau_max",
        "val_tau_p10",
        "val_tau_p50",
        "val_tau_p90",
        "train_difficulty_mean",
        "train_difficulty_std",
        "train_tau_at_min_ratio",
        "train_tau_at_max_ratio",
        "train_raw_tau_mean",
        "train_raw_tau_std",
        "train_raw_tau_min",
        "train_raw_tau_max",
        "train_raw_tau_p10",
        "train_raw_tau_p50",
        "train_raw_tau_p90",
        "train_raw_tau_ref_mean",
        "train_raw_tau_ref_std",
        "train_raw_tau_ref_min",
        "train_raw_tau_ref_max",
        "train_raw_tau_ref_p10",
        "train_raw_tau_ref_p50",
        "train_raw_tau_ref_p90",
        "train_raw_tau_edge_ratio",
        "train_tau_minus_tau_ref_mean",
        "train_tau_minus_tau_ref_std",
        "train_tau_minus_tau_ref_min",
        "train_tau_minus_tau_ref_max",
        "train_raw_tau_minus_ref_mean",
        "train_raw_tau_minus_ref_std",
        "train_raw_tau_minus_ref_min",
        "train_raw_tau_minus_ref_max",
        "train_corr_tau_difficulty",
        "train_corr_raw_tau_difficulty",
        "val_difficulty_mean",
        "val_difficulty_std",
        "val_tau_at_min_ratio",
        "val_tau_at_max_ratio",
        "val_raw_tau_mean",
        "val_raw_tau_std",
        "val_raw_tau_min",
        "val_raw_tau_max",
        "val_raw_tau_p10",
        "val_raw_tau_p50",
        "val_raw_tau_p90",
        "val_raw_tau_ref_mean",
        "val_raw_tau_ref_std",
        "val_raw_tau_ref_min",
        "val_raw_tau_ref_max",
        "val_raw_tau_ref_p10",
        "val_raw_tau_ref_p50",
        "val_raw_tau_ref_p90",
        "val_raw_tau_edge_ratio",
        "val_tau_minus_tau_ref_mean",
        "val_tau_minus_tau_ref_std",
        "val_tau_minus_tau_ref_min",
        "val_tau_minus_tau_ref_max",
        "val_raw_tau_minus_ref_mean",
        "val_raw_tau_minus_ref_std",
        "val_raw_tau_minus_ref_min",
        "val_raw_tau_minus_ref_max",
        "val_corr_tau_difficulty",
        "val_corr_raw_tau_difficulty",
    ]
    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fieldnames})


def print_ordinal_state(criterion: DamageLossModule) -> None:
    ordinal_state = criterion.export_state(CLASS_NAMES)
    print("Ordinal severity axis:")
    for class_name, position in ordinal_state["positions_by_class"].items():
        print(f"  {class_name} -> {position:.4f}")
    print(
        "Ordinal gaps: "
        f"gap_01={ordinal_state['gaps']['gap_01']:.4f}, "
        f"gap_12={ordinal_state['gaps']['gap_12']:.4f}, "
        f"gap_23={ordinal_state['gaps']['gap_23']:.4f}"
    )
    if ordinal_state["tau"] is not None:
        print(f"Reference tau: {ordinal_state['tau']:.4f}")


def print_tau_stats(prefix: str, tau_stats: dict[str, Any] | None) -> None:
    if tau_stats is None:
        return
    print(
        f"{prefix} tau "
        f"mean={tau_stats['mean']:.4f} std={tau_stats['std']:.4f} "
        f"min={tau_stats['min']:.4f} p10={tau_stats['p10']:.4f} "
        f"p50={tau_stats['p50']:.4f} p90={tau_stats['p90']:.4f} "
        f"max={tau_stats['max']:.4f}"
    )


def print_tau_difficulty_stats(
    prefix: str,
    difficulty_stats: dict[str, Any] | None,
    corr_tau_difficulty: float | None,
    corr_raw_tau_difficulty: float | None = None,
) -> None:
    if difficulty_stats is None:
        return
    corr_text = "n/a" if corr_tau_difficulty is None else f"{corr_tau_difficulty:.4f}"
    corr_raw_text = "n/a" if corr_raw_tau_difficulty is None else f"{corr_raw_tau_difficulty:.4f}"
    print(
        f"{prefix} difficulty "
        f"mean={difficulty_stats['mean']:.4f} std={difficulty_stats['std']:.4f} "
        f"corr_tau_difficulty={corr_text} "
        f"corr_raw_tau_difficulty={corr_raw_text}"
    )


def print_tau_boundary_stats(
    prefix: str,
    tau_at_min_ratio: float | None,
    tau_at_max_ratio: float | None,
) -> None:
    if tau_at_min_ratio is None and tau_at_max_ratio is None:
        return
    min_text = "n/a" if tau_at_min_ratio is None else f"{tau_at_min_ratio:.4f}"
    max_text = "n/a" if tau_at_max_ratio is None else f"{tau_at_max_ratio:.4f}"
    print(f"{prefix} tau_boundary at_min_ratio={min_text} at_max_ratio={max_text}")


def print_distribution_stats(
    prefix: str,
    label: str,
    stats: dict[str, Any] | None,
    *,
    keys: list[str],
) -> None:
    if stats is None:
        return
    parts = []
    for key in keys:
        value = stats.get(key)
        parts.append(f"{key}=n/a" if value is None else f"{key}={float(value):.4f}")
    print(f"{prefix} {label} " + " ".join(parts))


def print_optional_ratio(prefix: str, label: str, value: float | None) -> None:
    if value is None:
        return
    print(f"{prefix} {label}={value:.4f}")


def update_topk_checkpoints(
    checkpoints_dir: Path,
    checkpoint_state: dict[str, Any],
    topk_records: list[dict[str, Any]],
    *,
    epoch: int,
    score: float,
    save_topk: int,
) -> list[dict[str, Any]]:
    if save_topk <= 0:
        return topk_records

    if len(topk_records) >= save_topk:
        worst_record = min(topk_records, key=lambda item: (item["score"], -item["epoch"]))
        if float(score) < float(worst_record["score"]):
            return topk_records

    save_path = checkpoints_dir / f"epoch_{int(epoch):03d}_macro_f1_{float(score):.4f}.pth"
    save_checkpoint(save_path, checkpoint_state)
    topk_records = [
        record
        for record in topk_records
        if int(record["epoch"]) != int(epoch)
    ]
    topk_records.append(
        {
            "epoch": int(epoch),
            "score": float(score),
            "path": save_path,
        }
    )
    topk_records.sort(key=lambda item: (-float(item["score"]), int(item["epoch"])))
    while len(topk_records) > save_topk:
        removed = topk_records.pop()
        removed_path = Path(removed["path"])
        if removed_path.exists():
            removed_path.unlink()
    return topk_records


def make_checkpoint_state(
    epoch: int,
    model: torch.nn.Module,
    criterion: DamageLossModule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler,
    config: dict[str, Any],
    best_macro_f1: float,
    class_weights: torch.Tensor,
    class_counts: list[int],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    tau_phase: dict[str, Any],
    stage_name: str = "stage1",
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "stage_name": stage_name,
        "model_state_dict": model.state_dict(),
        "loss_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "best_macro_f1": float(best_macro_f1),
        "class_names": CLASS_NAMES,
        "class_weights": class_weights.detach().cpu(),
        "class_counts": [int(count) for count in class_counts],
        "ordinal_state": criterion.export_state(CLASS_NAMES),
        "tau_statistics": {
            "train": train_metrics.get("tau_stats"),
            "val": val_metrics.get("tau_stats"),
        },
        "difficulty_statistics": {
            "train": train_metrics.get("difficulty_stats"),
            "val": val_metrics.get("difficulty_stats"),
        },
        "corr_tau_difficulty": {
            "train": train_metrics.get("corr_tau_difficulty"),
            "val": val_metrics.get("corr_tau_difficulty"),
        },
        "corr_raw_tau_difficulty": {
            "train": train_metrics.get("corr_raw_tau_difficulty"),
            "val": val_metrics.get("corr_raw_tau_difficulty"),
        },
        "tau_phase": tau_phase,
        "dual_view_enabled": bool(train_metrics.get("dual_view_enabled", False)),
        "contrastive_enabled": bool(train_metrics.get("contrastive_enabled", False)),
    }


def configure_stage2_trainability(
    model: torch.nn.Module,
    criterion: DamageLossModule,
    config: dict[str, Any],
) -> None:
    # Stage2 is meant to rebalance classifier bias, so we freeze the representation path by default.
    if bool(config["training"].get("stage2_freeze_trunk", True)) and hasattr(model, "get_trunk_parameters"):
        set_requires_grad(list(model.get_trunk_parameters()), False)
    if hasattr(model, "get_primary_classifier_head_parameters"):
        set_requires_grad(list(model.get_primary_classifier_head_parameters()), True)
    if hasattr(model, "get_contrastive_head_parameters"):
        set_requires_grad(list(model.get_contrastive_head_parameters()), False)
    if bool(config["training"].get("stage2_freeze_ambiguity", True)) and hasattr(model, "get_ambiguity_head_parameters"):
        set_requires_grad(list(model.get_ambiguity_head_parameters()), False)
    if bool(config["training"].get("stage2_freeze_ordinal", True)):
        set_requires_grad(list(criterion.get_gap_parameters()), False)
        set_requires_grad(list(criterion.get_non_gap_trainable_parameters()), False)


def prepare_stage2_config(config: dict[str, Any]) -> dict[str, Any]:
    stage2_config = copy.deepcopy(config)
    stage2_config["training"]["epochs"] = int(config["training"].get("stage2_epochs", 8))
    stage2_config["training"]["lr"] = float(config["training"].get("stage2_lr", 1e-4))
    stage2_config["training"]["weight_decay"] = float(config["training"].get("stage2_weight_decay", 1e-4))
    stage2_config["training"]["early_stop_patience"] = int(config["training"].get("stage2_early_stop_patience", 3))
    stage2_config["training"]["dual_view_enabled"] = False
    stage2_config["training"]["enable_ordinal_contrastive"] = False
    stage2_config["training"]["lambda_contrastive"] = 0.0
    stage2_config["training"]["lambda_view_consistency"] = 0.0
    stage2_config["training"]["lambda_view_severity_consistency"] = 0.0
    return stage2_config


def run_stage2_head_rebalance(
    *,
    base_config: dict[str, Any],
    stage1_best_checkpoint_path: Path,
    run_dir: Path,
    shared_history_path: Path,
    class_weights: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    stage2_config = prepare_stage2_config(base_config)
    stage2_dir = ensure_dir(run_dir / "stage2")
    stage2_logs_dir = ensure_dir(stage2_dir / "logs")
    stage2_checkpoints_dir = ensure_dir(stage2_dir / "checkpoints")
    stage2_history_path = stage2_logs_dir / "history.jsonl"
    if stage2_history_path.exists():
        stage2_history_path.unlink()
    write_text(stage2_history_path, "")
    write_yaml(stage2_dir / "run_config.yaml", stage2_config)

    train_dataset = XBDOracleInstanceDamageDataset(
        config=stage2_config,
        split_name="train",
        list_path=stage2_config["data"]["train_list"],
        is_train=True,
    )
    val_dataset = XBDOracleInstanceDamageDataset(
        config=stage2_config,
        split_name=stage2_config["evaluation"]["split_name"],
        list_path=stage2_config["data"]["val_list"],
        is_train=False,
    )
    train_sampler = build_stage2_sampler(
        train_dataset,
        mode=str(stage2_config["training"].get("stage2_sampler_mode", "class_balanced")),
        focus_minor_major_weight=float(stage2_config["training"].get("stage2_focus_minor_major_weight", 1.25)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(stage2_config["seed"]))
    common_loader_kwargs = {
        "num_workers": int(stage2_config["training"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": oracle_instance_collate_fn,
        "worker_init_fn": seed_worker,
        "persistent_workers": int(stage2_config["training"]["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(stage2_config["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(stage2_config["training"]["batch_size"]),
        shuffle=False,
        **common_loader_kwargs,
    )

    checkpoint = load_checkpoint(stage1_best_checkpoint_path, map_location="cpu")
    model = build_model(stage2_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    criterion = build_loss_module_from_config(stage2_config, class_weights, device=device)
    if "loss_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["loss_state_dict"], strict=False)
    configure_stage2_trainability(model, criterion, stage2_config)
    optimizer = build_optimizer(model, criterion, stage2_config)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    total_epochs = int(stage2_config["training"]["epochs"])
    stage2_summary: dict[str, Any] = {
        "enabled": True,
        "run_dir": str(stage2_dir),
        "best_epoch": -1,
        "best_macro_f1": -1.0,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
    }

    for epoch in range(total_epochs):
        tau_phase = {
            "phase_name": "stage2_rebalance",
            "use_fixed_tau": False,
            "tau_override_value": None,
            "ambiguity_trainable": not bool(stage2_config["training"].get("stage2_freeze_ambiguity", True)),
        }
        corn_soft_enabled = bool(
            str(stage2_config["training"]["loss_mode"]) == "corn_adaptive_tau_safe"
            and float(stage2_config["training"].get("lambda_corn_soft", 0.0)) > 0.0
        )
        current_lr = set_epoch_lr(
            optimizer=optimizer,
            base_lr=float(stage2_config["training"]["lr"]),
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=min(int(stage2_config["training"].get("warmup_epochs", 0)), total_epochs),
        )
        print(
            f"\n[Stage2] Epoch [{epoch + 1}/{total_epochs}] lr={current_lr:.8f} "
            f"sampler={stage2_config['training'].get('stage2_sampler_mode', 'class_balanced')}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            epoch_index=epoch,
            stage_name="stage2",
            tau_override_value=None,
            scaler=scaler,
            corn_soft_enabled=corn_soft_enabled,
            corn_head_parameters=(
                [param for param in model.get_primary_classifier_head_parameters() if param.requires_grad]
                if hasattr(model, "get_primary_classifier_head_parameters")
                else None
            ),
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            amp_enabled=amp_enabled,
            epoch_index=epoch,
            stage_name="stage2",
            tau_override_value=None,
            scaler=None,
            corn_soft_enabled=corn_soft_enabled,
            corn_head_parameters=None,
        )

        epoch_record = {
            "epoch": epoch + 1,
            "stage_name": "stage2",
            "lr": current_lr,
            "tau_phase": tau_phase,
            "corn_soft_enabled": corn_soft_enabled,
            "train": train_metrics,
            "val": val_metrics,
            "ordinal": criterion.export_state(CLASS_NAMES),
        }
        append_jsonl(stage2_history_path, epoch_record)
        append_jsonl(shared_history_path, epoch_record)

        checkpoint_state = make_checkpoint_state(
            epoch=epoch + 1,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            config=stage2_config,
            best_macro_f1=max(best_macro_f1, float(val_metrics["macro_f1"])),
            class_weights=class_weights,
            class_counts=train_dataset.class_counts,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            tau_phase=tau_phase,
            stage_name="stage2",
        )
        last_path = stage2_checkpoints_dir / "last.pth"
        save_checkpoint(last_path, checkpoint_state)
        stage2_summary["last_checkpoint"] = str(last_path)

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_path = stage2_checkpoints_dir / "best_macro_f1.pth"
            save_checkpoint(best_path, checkpoint_state)
            stage2_summary["best_checkpoint"] = str(best_path)
            print(f"[Stage2] Best macro_f1 updated to {best_macro_f1:.4f} at epoch {best_epoch}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(stage2_config["training"].get("early_stop_patience", 3)):
            print(f"[Stage2] Early stopping after {epochs_without_improvement} epochs without improvement.")
            break

    stage2_summary["best_epoch"] = best_epoch
    stage2_summary["best_macro_f1"] = best_macro_f1
    stage2_summary["sampler_mode"] = stage2_config["training"].get("stage2_sampler_mode", "class_balanced")
    stage2_summary["focus_minor_major_weight"] = float(
        stage2_config["training"].get("stage2_focus_minor_major_weight", 1.25)
    )
    write_json(stage2_dir / "stage2_summary.json", stage2_summary)
    return stage2_summary


def main() -> None:
    args = parse_args()
    direct_run = len(sys.argv) == 1
    config = load_and_override_config(args)
    run_dir = make_run_dir(config)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    train_log_path = logs_dir / "train.log"
    history_path = logs_dir / "history.jsonl"

    if history_path.exists():
        history_path.unlink()
    write_text(history_path, "")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with train_log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            _run_training(
                config=config,
                direct_run=direct_run,
                run_dir=run_dir,
                checkpoints_dir=checkpoints_dir,
                logs_dir=logs_dir,
                history_path=history_path,
            )
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _run_training(
    config: dict[str, Any],
    direct_run: bool,
    run_dir: Path,
    checkpoints_dir: Path,
    logs_dir: Path,
    history_path: Path,
) -> None:

    set_seed(int(config["seed"]))
    device = get_device()
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    write_yaml(run_dir / "run_config.yaml", config)

    train_loader, val_loader, train_dataset, val_dataset = make_dataloaders(config)
    class_weights = compute_class_weights(train_dataset.class_counts)
    save_class_statistics(run_dir, train_dataset.class_counts, class_weights)

    model = build_model(config).to(device)
    criterion = build_loss_module_from_config(config, class_weights, device=device)
    contrastive_loss_module = build_contrastive_loss_module(config, device=device)

    optimizer = build_optimizer(model, criterion, config)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    ordinal_history: list[dict[str, Any]] = []
    tau_history: list[dict[str, Any]] = []
    topk_records: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    total_epochs = int(config["training"]["epochs"])
    early_stop_patience = int(config["training"].get("early_stop_patience", 6))
    save_topk = int(config["training"].get("save_topk", 3))
    corn_soft_start_epoch = int(config["training"].get("corn_soft_start_epoch", 3))
    corn_head_parameters = []
    if hasattr(model, "get_classifier_head_parameters"):
        corn_head_parameters = [param for param in model.get_classifier_head_parameters() if param.requires_grad]
    loss_mode = str(config["training"]["loss_mode"])
    default_tau_parameterization = (
        "bounded_sigmoid"
        if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
        else "sigmoid"
    )
    default_ambiguity_lr_scale = 0.30 if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"} else 0.5

    print(f"DIRECT_RUN={direct_run}")
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Logs dir: {logs_dir}")
    print(f"Model type: {config['model']['model_type']}")
    print(f"Head type: {config['model'].get('head_type', 'standard')}")
    print(f"Loss mode: {loss_mode}")
    print(f"Tau parameterization: {config['training'].get('tau_parameterization', default_tau_parameterization)}")
    print(f"Tau logit scale: {float(config['training'].get('tau_logit_scale', 2.0)):.4f}")
    print(f"Raw tau soft margin: {float(config['training'].get('raw_tau_soft_margin', 1.5)):.4f}")
    print(f"Ambiguity lr scale: {float(config['training'].get('ambiguity_lr_scale', default_ambiguity_lr_scale)):.4f}")
    print(f"Lambda corn soft: {float(config['training'].get('lambda_corn_soft', 0.03)):.4f}")
    print(f"Lambda corn soft EMD: {float(config['training'].get('lambda_corn_soft_emd', 0.02)):.4f}")
    print(f"Lambda decoded unimodal: {float(config['training'].get('lambda_decoded_unimodal', 0.01)):.4f}")
    print(f"Corn soft start epoch: {corn_soft_start_epoch}")
    print(f"Dual view enabled: {bool(config['training'].get('dual_view_enabled', False))}")
    print(f"Contrastive enabled: {bool(config['training'].get('enable_ordinal_contrastive', False))}")
    print(f"Early stop patience: {early_stop_patience}")
    print(f"Save top-k checkpoints: {save_topk}")
    print(f"Ambiguity input detached in safe mode: {loss_mode == 'corn_adaptive_tau_safe'}")
    print(f"Train instances: {len(train_dataset)} | Val instances: {len(val_dataset)}")
    print(f"Train cache: {train_dataset.cache_path}")
    print(f"Val cache: {val_dataset.cache_path}")
    print(f"Class counts: {dict(zip(CLASS_NAMES, train_dataset.class_counts))}")
    print(f"Class weights: {dict(zip(CLASS_NAMES, [round(x, 6) for x in class_weights.tolist()]))}")
    print_ordinal_state(criterion)

    for epoch in range(total_epochs):
        tau_phase = resolve_tau_phase(config, epoch)
        set_ambiguity_head_trainable(model, bool(tau_phase["ambiguity_trainable"]))
        corn_soft_enabled = bool(loss_mode == "corn_adaptive_tau_safe" and (epoch + 1) >= corn_soft_start_epoch)

        current_lr = set_epoch_lr(
            optimizer=optimizer,
            base_lr=float(config["training"]["lr"]),
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=int(config["training"]["warmup_epochs"]),
        )
        print(
            f"\nEpoch [{epoch + 1}/{total_epochs}] lr={current_lr:.8f} "
            f"tau_phase={tau_phase['phase_name']} "
            f"corn_soft_enabled={corn_soft_enabled}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            epoch_index=epoch,
            stage_name="stage1",
            tau_override_value=tau_phase["tau_override_value"],
            scaler=scaler,
            corn_soft_enabled=corn_soft_enabled,
            corn_head_parameters=corn_head_parameters,
            contrastive_loss_module=contrastive_loss_module,
            lambda_contrastive=float(config["training"].get("lambda_contrastive", 0.0)),
            contrastive_loss_warmup_epochs=int(config["training"].get("contrastive_loss_warmup_epochs", 3)),
            lambda_view_consistency=float(config["training"].get("lambda_view_consistency", 0.0)),
            lambda_view_severity_consistency=float(config["training"].get("lambda_view_severity_consistency", 0.0)),
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            amp_enabled=amp_enabled,
            epoch_index=epoch,
            stage_name="stage1",
            tau_override_value=tau_phase["tau_override_value"],
            scaler=None,
            corn_soft_enabled=corn_soft_enabled,
            corn_head_parameters=corn_head_parameters,
            contrastive_loss_module=contrastive_loss_module,
        )

        ordinal_record = make_ordinal_epoch_record(epoch + 1, criterion, train_metrics, val_metrics, tau_phase)
        tau_record = make_tau_epoch_record(epoch + 1, train_metrics, val_metrics, tau_phase)
        ordinal_history.append(ordinal_record)
        tau_history.append(tau_record)
        write_ordinal_history(run_dir, ordinal_history)
        write_tau_history(run_dir, tau_history)

        epoch_record = {
            "epoch": epoch + 1,
            "stage_name": "stage1",
            "lr": current_lr,
            "tau_phase": tau_phase,
            "corn_soft_enabled": corn_soft_enabled,
            "train": train_metrics,
            "val": val_metrics,
            "ordinal": criterion.export_state(CLASS_NAMES),
        }
        append_jsonl(history_path, epoch_record)

        print(
            "Train "
            f"loss={train_metrics['loss']:.4f} "
            f"ce={train_metrics['loss_terms']['loss_ce']:.4f} "
            f"ord={train_metrics['loss_terms']['loss_ord']:.4f} "
            f"corn_main={train_metrics['loss_terms']['loss_corn_main']:.4f} "
            f"corn_soft={train_metrics['loss_terms']['loss_corn_soft']:.4f} "
            f"corn_soft_emd={train_metrics['loss_terms']['loss_corn_soft_emd']:.4f} "
            f"corn_unimodal={train_metrics['loss_terms']['loss_corn_unimodal']:.4f} "
            f"view_cons={train_metrics['loss_terms']['loss_view_consistency']:.4f} "
            f"view_sev={train_metrics['loss_terms']['loss_view_severity_consistency']:.4f} "
            f"contrastive={train_metrics['loss_terms']['loss_contrastive']:.4f} "
            f"uni={train_metrics['loss_terms']['loss_uni']:.4f} "
            f"conc={train_metrics['loss_terms']['loss_conc']:.4f} "
            f"tau_reg={train_metrics['loss_terms']['loss_tau_reg']:.4f} "
            f"tau_mean={train_metrics['loss_terms']['loss_tau_mean']:.4f} "
            f"tau_var={train_metrics['loss_terms']['loss_tau_var']:.4f} "
            f"tau_diff={train_metrics['loss_terms']['loss_tau_diff']:.4f} "
            f"tau_rank={train_metrics['loss_terms']['loss_tau_rank']:.4f} "
            f"raw_tau_diff={train_metrics['loss_terms']['loss_raw_tau_diff']:.4f} "
            f"raw_tau_center={train_metrics['loss_terms']['loss_raw_tau_center']:.4f} "
            f"raw_tau_bound={train_metrics['loss_terms']['loss_raw_tau_bound']:.4f} "
            f"gap={train_metrics['loss_terms']['loss_gap_reg']:.4f} "
            f"q_entropy={train_metrics['soft_target_stats']['entropy_mean']:.4f} "
            f"q_gt={train_metrics['soft_target_stats']['gt_mass_mean']:.4f} "
            f"acc={train_metrics['overall_accuracy']:.4f} "
            f"macro_f1={train_metrics['macro_f1']:.4f} "
            f"far_error={train_metrics['far_error_rate']:.4f} "
            f"corn_soft_enabled={train_metrics['corn_soft_enabled']}"
        )
        print(
            "Val   "
            f"loss={val_metrics['loss']:.4f} "
            f"ce={val_metrics['loss_terms']['loss_ce']:.4f} "
            f"ord={val_metrics['loss_terms']['loss_ord']:.4f} "
            f"corn_main={val_metrics['loss_terms']['loss_corn_main']:.4f} "
            f"corn_soft={val_metrics['loss_terms']['loss_corn_soft']:.4f} "
            f"corn_soft_emd={val_metrics['loss_terms']['loss_corn_soft_emd']:.4f} "
            f"corn_unimodal={val_metrics['loss_terms']['loss_corn_unimodal']:.4f} "
            f"view_cons={val_metrics['loss_terms']['loss_view_consistency']:.4f} "
            f"view_sev={val_metrics['loss_terms']['loss_view_severity_consistency']:.4f} "
            f"contrastive={val_metrics['loss_terms']['loss_contrastive']:.4f} "
            f"uni={val_metrics['loss_terms']['loss_uni']:.4f} "
            f"conc={val_metrics['loss_terms']['loss_conc']:.4f} "
            f"tau_reg={val_metrics['loss_terms']['loss_tau_reg']:.4f} "
            f"tau_mean={val_metrics['loss_terms']['loss_tau_mean']:.4f} "
            f"tau_var={val_metrics['loss_terms']['loss_tau_var']:.4f} "
            f"tau_diff={val_metrics['loss_terms']['loss_tau_diff']:.4f} "
            f"tau_rank={val_metrics['loss_terms']['loss_tau_rank']:.4f} "
            f"raw_tau_diff={val_metrics['loss_terms']['loss_raw_tau_diff']:.4f} "
            f"raw_tau_center={val_metrics['loss_terms']['loss_raw_tau_center']:.4f} "
            f"raw_tau_bound={val_metrics['loss_terms']['loss_raw_tau_bound']:.4f} "
            f"gap={val_metrics['loss_terms']['loss_gap_reg']:.4f} "
            f"q_entropy={val_metrics['soft_target_stats']['entropy_mean']:.4f} "
            f"q_gt={val_metrics['soft_target_stats']['gt_mass_mean']:.4f} "
            f"acc={val_metrics['overall_accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"far_error={val_metrics['far_error_rate']:.4f} "
            f"corn_soft_enabled={val_metrics['corn_soft_enabled']}"
        )
        print(f"Train corn_mode={train_metrics.get('corn_mode')} corn_task_losses={train_metrics.get('corn_task_losses')}")
        print(f"Val   corn_mode={val_metrics.get('corn_mode')} corn_task_losses={val_metrics.get('corn_task_losses')}")
        print(f"Train probs_from_corn_stats={train_metrics.get('probs_from_corn_stats')}")
        print(f"Val   probs_from_corn_stats={val_metrics.get('probs_from_corn_stats')}")
        print(
            "Train detach_flags "
            f"soft_target={train_metrics.get('adaptive_soft_target_detached')} "
            f"ambiguity_input={train_metrics.get('ambiguity_input_detached')}"
        )
        print(
            "Val   detach_flags "
            f"soft_target={val_metrics.get('adaptive_soft_target_detached')} "
            f"ambiguity_input={val_metrics.get('ambiguity_input_detached')}"
        )
        print(f"Train context_aug_usage_stats={train_metrics.get('context_aug_usage_stats')}")
        print(f"Val   context_aug_usage_stats={val_metrics.get('context_aug_usage_stats')}")
        print_tau_stats("Train", train_metrics.get("tau_stats"))
        print_tau_stats("Val  ", val_metrics.get("tau_stats"))
        print_tau_boundary_stats(
            "Train",
            train_metrics.get("tau_at_min_ratio"),
            train_metrics.get("tau_at_max_ratio"),
        )
        print_tau_boundary_stats(
            "Val  ",
            val_metrics.get("tau_at_min_ratio"),
            val_metrics.get("tau_at_max_ratio"),
        )
        print_tau_difficulty_stats(
            "Train",
            train_metrics.get("difficulty_stats"),
            train_metrics.get("corr_tau_difficulty"),
            train_metrics.get("corr_raw_tau_difficulty"),
        )
        print_tau_difficulty_stats(
            "Val  ",
            val_metrics.get("difficulty_stats"),
            val_metrics.get("corr_tau_difficulty"),
            val_metrics.get("corr_raw_tau_difficulty"),
        )
        print_distribution_stats(
            "Train",
            "raw_tau",
            train_metrics.get("raw_tau_stats"),
            keys=["mean", "std", "min", "max", "p10", "p50", "p90"],
        )
        print_distribution_stats(
            "Val  ",
            "raw_tau",
            val_metrics.get("raw_tau_stats"),
            keys=["mean", "std", "min", "max", "p10", "p50", "p90"],
        )
        print_distribution_stats(
            "Train",
            "raw_tau_ref",
            train_metrics.get("raw_tau_ref_stats"),
            keys=["mean", "std", "min", "max", "p10", "p50", "p90"],
        )
        print_distribution_stats(
            "Val  ",
            "raw_tau_ref",
            val_metrics.get("raw_tau_ref_stats"),
            keys=["mean", "std", "min", "max", "p10", "p50", "p90"],
        )
        print_optional_ratio("Train", "raw_tau_edge_ratio", train_metrics.get("raw_tau_edge_ratio"))
        print_optional_ratio("Val  ", "raw_tau_edge_ratio", val_metrics.get("raw_tau_edge_ratio"))
        print_distribution_stats(
            "Train",
            "tau_minus_tau_ref",
            train_metrics.get("tau_minus_tau_ref_stats"),
            keys=["mean", "std", "min", "max"],
        )
        print_distribution_stats(
            "Val  ",
            "tau_minus_tau_ref",
            val_metrics.get("tau_minus_tau_ref_stats"),
            keys=["mean", "std", "min", "max"],
        )
        print_distribution_stats(
            "Train",
            "raw_tau_minus_ref",
            train_metrics.get("raw_tau_minus_ref_stats"),
            keys=["mean", "std", "min", "max"],
        )
        print_distribution_stats(
            "Val  ",
            "raw_tau_minus_ref",
            val_metrics.get("raw_tau_minus_ref_stats"),
            keys=["mean", "std", "min", "max"],
        )
        print(
            "Train class_f1 "
            f"minor={train_metrics['per_class']['minor-damage']['f1']:.4f} "
            f"major={train_metrics['per_class']['major-damage']['f1']:.4f}"
        )
        print(
            "Val   class_f1 "
            f"minor={val_metrics['per_class']['minor-damage']['f1']:.4f} "
            f"major={val_metrics['per_class']['major-damage']['f1']:.4f}"
        )
        print_ordinal_state(criterion)

        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_state = make_checkpoint_state(
            epoch=epoch + 1,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            best_macro_f1=best_macro_f1,
            class_weights=class_weights,
            class_counts=train_dataset.class_counts,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            tau_phase=tau_phase,
            stage_name="stage1",
        )
        save_checkpoint(checkpoints_dir / "last.pth", checkpoint_state)

        topk_records = update_topk_checkpoints(
            checkpoints_dir,
            checkpoint_state,
            topk_records,
            epoch=epoch + 1,
            score=float(val_metrics["macro_f1"]),
            save_topk=save_topk,
        )

        if improved:
            save_checkpoint(checkpoints_dir / "best_macro_f1.pth", checkpoint_state)
            print(f"Best macro_f1 updated to {best_macro_f1:.4f} at epoch {best_epoch}")

        if epochs_without_improvement >= early_stop_patience:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs without macro_f1 improvement."
            )
            break

    stage1_best_checkpoint = checkpoints_dir / "best_macro_f1.pth"
    stage1_summary = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_checkpoint": str(stage1_best_checkpoint),
    }
    stage2_summary = {"enabled": False}
    final_best_source = "stage1_best"
    final_best_checkpoint = stage1_best_checkpoint
    final_best_macro_f1 = best_macro_f1
    if bool(config["training"].get("stage2_head_rebalance_enabled", False)) and stage1_best_checkpoint.exists():
        stage2_summary = run_stage2_head_rebalance(
            base_config=config,
            stage1_best_checkpoint_path=stage1_best_checkpoint,
            run_dir=run_dir,
            shared_history_path=history_path,
            class_weights=class_weights,
            device=device,
            amp_enabled=amp_enabled,
        )
        stage2_best_macro_f1 = float(stage2_summary.get("best_macro_f1", -1.0))
        stage2_best_checkpoint = stage2_summary.get("best_checkpoint")
        if stage2_best_checkpoint is not None and stage2_best_macro_f1 > final_best_macro_f1:
            final_best_source = "stage2_best"
            final_best_checkpoint = Path(stage2_best_checkpoint)
            final_best_macro_f1 = stage2_best_macro_f1

    final_best_path = checkpoints_dir / "final_best.pth"
    if final_best_checkpoint.exists():
        save_checkpoint(final_best_path, load_checkpoint(final_best_checkpoint, map_location="cpu"))

    summary = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "run_dir": str(run_dir),
        "model_type": config["model"]["model_type"],
        "head_type": config["model"].get("head_type", "standard"),
        "loss_mode": config["training"]["loss_mode"],
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
        "ordinal_state": criterion.export_state(CLASS_NAMES),
        "tau_history_path": str(run_dir / "learned_tau_stats.json"),
        "saved_topk_checkpoints": [str(record["path"]) for record in topk_records],
        "stage1_best": stage1_summary,
        "stage2_best": stage2_summary,
        "final_best_source": final_best_source,
        "final_best_checkpoint": str(final_best_checkpoint),
        "final_best_macro_f1": float(final_best_macro_f1),
    }
    write_json(run_dir / "train_summary.json", summary)
    print("\nTraining complete.")
    print(f"Best macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")
    print(f"Final best source: {final_best_source} ({final_best_macro_f1:.4f})")
    print(f"Checkpoints saved to: {checkpoints_dir}")


if __name__ == "__main__":
    main()
