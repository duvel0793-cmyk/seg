from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.multi_source_instance_damage import build_instance_dataset
from datasets.xbd_oracle_instance_damage import (
    CLASS_NAMES,
    oracle_instance_collate_fn,
)
from models import build_model
from utils.config_utils import ensure_auxiliary_config_defaults
from utils.io import ensure_dir, load_checkpoint, load_state_dict_with_fallback, read_yaml, write_json, write_text, write_yaml
from utils.losses import CORNLoss, DamageLossModule, build_loss_function, compute_expected_severity_from_probabilities
from utils.metrics import (
    compute_classification_metrics,
    compute_emd_from_probabilities,
    compute_ordinal_error_profile,
    save_confusion_matrix_plot,
    save_matrix_heatmap_plot,
)
from utils.seed import seed_worker, set_seed
from utils.visualize import save_prediction_visual

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.calibration import apply_corn_calibration, fit_corn_threshold_calibration

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
    "corn_ordinal_multitask_v1",
]
STANDARD_ORACLE_LOSS_MODES = [
    "weighted_ce",
    "fixed_cda",
    "learnable_cda",
    "adaptive_ucl_cda",
    "adaptive_ucl_cda_v2",
    "adaptive_ucl_cda_v3",
]


def resolve_default_config_path() -> Path:
    preferred = PROJECT_ROOT / "configs" / "vscode_run.yaml"
    if preferred.exists():
        return preferred
    return PROJECT_ROOT / "configs" / "default.yaml"


DEFAULT_CONFIG_PATH = resolve_default_config_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate xBD Oracle Instance Damage Classification")
    bool_action = getattr(argparse, "BooleanOptionalAction", None)

    def add_optional_bool_argument(name: str, *, default: bool | None = None) -> None:
        if bool_action is None:
            raise RuntimeError("BooleanOptionalAction is required for this evaluation CLI.")
        parser.add_argument(name, action=bool_action, default=default)

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--xbd_root", type=str, default=None)
    parser.add_argument("--bright_root", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--train_source", type=str, choices=["xbd_only", "bright_only", "xbd_bright_mixed"], default=None)
    parser.add_argument("--eval_source", type=str, choices=["xbd_only", "bright_only", "xbd_bright_mixed"], default=None)
    parser.add_argument("--instance_source", type=str, default=None)
    parser.add_argument("--allow_tier3", action="store_true")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--context_ratio", type=float, default=None)
    parser.add_argument("--min_polygon_area", type=float, default=None)
    parser.add_argument("--min_mask_pixels", type=int, default=None)
    parser.add_argument("--max_out_of_bound_ratio", type=float, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--bright_positive_class_name", type=str, default=None)
    parser.add_argument("--bright_train_list", type=str, default=None)
    parser.add_argument("--bright_val_list", type=str, default=None)
    parser.add_argument("--bright_test_list", type=str, default=None)
    parser.add_argument("--bright_train_ratio", type=float, default=None)
    parser.add_argument("--bright_val_ratio", type=float, default=None)
    parser.add_argument("--bright_split_seed", type=int, default=None)
    add_optional_bool_argument("--use_dual_scale", default=None)
    parser.add_argument("--local_size", type=int, default=None)
    parser.add_argument("--context_size", type=int, default=None)
    parser.add_argument("--local_margin_ratio", type=float, default=None)
    parser.add_argument("--context_scale", type=float, default=None)
    parser.add_argument("--min_crop_size", type=int, default=None)
    add_optional_bool_argument("--use_boundary_prior", default=None)
    parser.add_argument("--encoder_in_channels", type=int, default=None)
    parser.add_argument("--boundary_width_px", type=int, default=None)
    parser.add_argument("--dilation_radius", type=int, default=None)
    parser.add_argument("--erosion_radius", type=int, default=None)
    add_optional_bool_argument("--return_abs_diff", default=None)
    add_optional_bool_argument("--feed_abs_diff_to_model", default=None)

    parser.add_argument("--model_type", type=str, choices=MODEL_TYPE_CHOICES, default=None)
    parser.add_argument("--head_type", type=str, choices=HEAD_TYPE_CHOICES, default=None)
    parser.add_argument("--loss_mode", type=str, choices=LOSS_MODE_CHOICES, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    add_optional_bool_argument("--use_context_branch", default=None)
    parser.add_argument("--fuse_local_context_mode", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_visuals_per_class", type=int, default=None)
    parser.add_argument("--calibration_file", type=str, default=None)
    add_optional_bool_argument("--fit_calibration_on_val", default=None)
    add_optional_bool_argument("--save_calibration_file", default=None)
    add_optional_bool_argument("--enable_threshold_calibration", default=None)
    parser.add_argument("--calibration_type", type=str, default=None)
    parser.add_argument("--calibration_max_iter", type=int, default=None)
    parser.add_argument("--calibration_lr", type=float, default=None)
    return parser.parse_args()


def ensure_training_defaults(config: dict[str, Any], default_loss_mode: str = "corn_adaptive_tau_safe") -> dict[str, Any]:
    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("backbone", "convnextv2_tiny")
    model_cfg.setdefault("pretrained", True)
    model_cfg.setdefault("head_type", "standard")
    model_cfg.setdefault("ambiguity_hidden_features", 256)
    model_cfg.setdefault("use_context_branch", False)
    model_cfg.setdefault("fuse_local_context_mode", "concat_absdiff")
    model_cfg.setdefault("aux_soft_label_head", {})
    model_cfg.setdefault("ordinal_contrastive", {})

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
    train_cfg.setdefault("enable_corn_soft_emd", False)
    train_cfg.setdefault("lambda_corn_soft_emd", 0.02)
    train_cfg.setdefault("enable_decoded_unimodal_reg", False)
    train_cfg.setdefault("lambda_decoded_unimodal", 0.01)
    train_cfg.setdefault("decoded_unimodal_margin", 0.0)
    train_cfg.setdefault("fixed_cda_alpha", 0.3)
    train_cfg.setdefault("tau_init", 0.22)
    train_cfg.setdefault("tau_min", 0.12)
    train_cfg.setdefault("tau_max", 0.45)
    train_cfg.setdefault("tau_base", 0.22)
    train_cfg.setdefault("delta_scale", 0.12)
    train_cfg.setdefault("tau_parameterization", "bounded_sigmoid")
    train_cfg.setdefault("tau_logit_scale", 2.0)
    train_cfg.setdefault("tau_variance_weight", 0.01)
    train_cfg.setdefault("tau_std_floor", 0.03)
    train_cfg.setdefault("tau_rank_margin_difficulty", 0.10)
    train_cfg.setdefault("tau_rank_margin_value", 0.01)
    train_cfg.setdefault("raw_tau_soft_margin", 1.5)
    train_cfg.setdefault("tau_target", 0.22)
    train_cfg.setdefault("tau_easy", 0.16)
    train_cfg.setdefault("tau_hard", 0.32)
    train_cfg.setdefault("tau_freeze_epochs", 1)
    train_cfg.setdefault("tau_warmup_value", 0.22)
    train_cfg.setdefault("corn_soft_start_epoch", 3)
    train_cfg.setdefault("ambiguity_lr_scale", 0.30)
    train_cfg.setdefault("concentration_margin", 0.05)
    train_cfg.setdefault("lambda_aux", 0.2)
    train_cfg.setdefault("label_smoothing", 0.05)
    train_cfg.setdefault("use_focal", False)
    train_cfg.setdefault("focal_gamma", 2.0)
    train_cfg.setdefault("dual_view_enabled", False)
    train_cfg.setdefault("enable_ordinal_contrastive", False)
    train_cfg.setdefault("lambda_distribution", 0.18)
    train_cfg.setdefault("lambda_severity_reg", 0.08)
    train_cfg.setdefault("lambda_consistency_dist", 0.03)
    train_cfg.setdefault("lambda_consistency_severity", 0.03)
    train_cfg.setdefault("contrastive_proj_dim", 128)
    train_cfg.setdefault("contrastive_hidden_features", 512)
    train_cfg.setdefault("contrastive_dropout", 0.10)
    train_cfg.setdefault("save_epoch_checkpoints", False)
    train_cfg.setdefault("save_every_n_epochs", 1)

    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg.setdefault("enable_threshold_calibration", False)
    eval_cfg.setdefault("calibration_type", "temperature_plus_bias")
    eval_cfg.setdefault("calibration_max_iter", 200)
    eval_cfg.setdefault("calibration_lr", 0.01)
    eval_cfg.setdefault("fit_calibration_on_val", False)
    eval_cfg.setdefault("save_calibration_file", False)
    return ensure_auxiliary_config_defaults(config)


def harmonize_model_and_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    train_cfg = config["training"]

    model_type = str(model_cfg["model_type"])
    head_type = str(model_cfg.get("head_type", "standard"))
    loss_mode = str(train_cfg["loss_mode"])

    if model_type in {"post_only", "siamese_simple"}:
        if loss_mode in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", "corn", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} or head_type == "corn":
            raise ValueError(f"{model_type} does not support loss_mode={loss_mode} / head_type={head_type}.")
        model_cfg["head_type"] = "standard"
        return config

    if model_type not in {"oracle_mcd", "oracle_mcd_corn"}:
        raise ValueError(f"Unsupported model_type='{model_type}'.")

    if model_type == "oracle_mcd_corn" or head_type == "corn" or loss_mode in {"corn", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"}:
        model_cfg["model_type"] = "oracle_mcd_corn"
        model_cfg["head_type"] = "corn"
        if loss_mode not in {"corn", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"}:
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


def load_config(path: str | Path) -> dict[str, Any]:
    return copy.deepcopy(read_yaml(path))


def apply_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    default_loss_mode: str = "corn_adaptive_tau_safe",
) -> dict[str, Any]:
    config = ensure_training_defaults(config, default_loss_mode=default_loss_mode)

    if args.seed is not None:
        config["seed"] = args.seed

    data_cfg = config["data"]
    if args.root_dir is not None:
        data_cfg["root_dir"] = args.root_dir
        data_cfg["xbd_root"] = args.root_dir
    if args.xbd_root is not None:
        data_cfg["xbd_root"] = args.xbd_root
        data_cfg["root_dir"] = args.xbd_root
    if args.bright_root is not None:
        data_cfg["bright_root"] = args.bright_root
    if args.val_list is not None:
        data_cfg["val_list"] = args.val_list
    if args.train_source is not None:
        data_cfg["train_source"] = args.train_source
    if args.eval_source is not None:
        data_cfg["eval_source"] = args.eval_source
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
    bright_cfg = data_cfg.setdefault("bright", {})
    if args.bright_positive_class_name is not None:
        bright_cfg["positive_class_name"] = args.bright_positive_class_name
    if args.bright_train_list is not None:
        bright_cfg["train_list"] = args.bright_train_list
    if args.bright_val_list is not None:
        bright_cfg["val_list"] = args.bright_val_list
    if args.bright_test_list is not None:
        bright_cfg["test_list"] = args.bright_test_list
    if args.bright_train_ratio is not None:
        bright_cfg["train_ratio"] = args.bright_train_ratio
    if args.bright_val_ratio is not None:
        bright_cfg["val_ratio"] = args.bright_val_ratio
    if args.bright_split_seed is not None:
        bright_cfg["split_seed"] = args.bright_split_seed

    input_mode_cfg = config.setdefault("input_mode", {})
    for arg_name in ["use_dual_scale"]:
        value = getattr(args, arg_name)
        if value is not None:
            input_mode_cfg[arg_name] = value
    for arg_name in ["local_size", "context_size", "local_margin_ratio", "context_scale", "min_crop_size"]:
        value = getattr(args, arg_name)
        if value is not None:
            input_mode_cfg[arg_name] = value

    geometry_cfg = config.setdefault("geometry_prior", {})
    for arg_name in ["use_boundary_prior"]:
        value = getattr(args, arg_name)
        if value is not None:
            geometry_cfg[arg_name] = value
    for arg_name in ["encoder_in_channels", "boundary_width_px", "dilation_radius", "erosion_radius"]:
        value = getattr(args, arg_name)
        if value is not None:
            geometry_cfg[arg_name] = value

    diff_cfg = config.setdefault("diff_input", {})
    for arg_name in ["return_abs_diff", "feed_abs_diff_to_model"]:
        value = getattr(args, arg_name)
        if value is not None:
            diff_cfg[arg_name] = value

    model_cfg = config["model"]
    if args.model_type is not None:
        model_cfg["model_type"] = args.model_type
    if args.head_type is not None:
        model_cfg["head_type"] = args.head_type
    if args.no_pretrained:
        model_cfg["pretrained"] = False
    if args.use_context_branch is not None:
        model_cfg["use_context_branch"] = args.use_context_branch
    if args.fuse_local_context_mode is not None:
        model_cfg["fuse_local_context_mode"] = args.fuse_local_context_mode

    train_cfg = config["training"]
    if args.loss_mode is not None:
        train_cfg["loss_mode"] = args.loss_mode
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.no_amp:
        train_cfg["amp"] = False

    output_cfg = config["output"]
    if args.output_root is not None:
        output_cfg["output_root"] = args.output_root
    if args.exp_name is not None:
        output_cfg["exp_name"] = args.exp_name

    if args.num_visuals_per_class is not None:
        config["evaluation"]["num_visuals_per_class"] = args.num_visuals_per_class
    for arg_name in [
        "enable_threshold_calibration",
        "fit_calibration_on_val",
        "save_calibration_file",
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            config["evaluation"][arg_name] = value
    for arg_name in ["calibration_type", "calibration_max_iter", "calibration_lr"]:
        value = getattr(args, arg_name)
        if value is not None:
            config["evaluation"][arg_name] = value
    return harmonize_model_and_loss_config(ensure_auxiliary_config_defaults(config))


def make_run_dir(config: dict[str, Any]) -> Path:
    return (
        Path(config["output"]["output_root"])
        / config["output"]["exp_name"]
        / config["model"]["model_type"]
        / config["training"]["loss_mode"]
    )


def resolve_checkpoint_path(config: dict[str, Any], explicit_checkpoint: str | None) -> Path:
    if explicit_checkpoint is not None:
        return Path(explicit_checkpoint)
    return make_run_dir(config) / "checkpoints" / "best_macro_f1.pth"


def make_dataloader(config: dict[str, Any]) -> tuple[DataLoader, Any]:
    dataset = build_instance_dataset(
        config,
        split_name=str(config["evaluation"]["split_name"]),
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=oracle_instance_collate_fn,
        worker_init_fn=seed_worker,
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    return loader, dataset


def build_loss_module(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> DamageLossModule:
    checkpoint_weights = checkpoint.get("class_weights")
    class_weights = None
    if checkpoint_weights is not None:
        if isinstance(checkpoint_weights, torch.Tensor):
            class_weights = checkpoint_weights.float()
        else:
            class_weights = torch.tensor(checkpoint_weights, dtype=torch.float32)

    aux_head_cfg = config["model"].get("aux_soft_label_head", {})
    multitask_cfg = config.get("ordinal_multitask", {})
    dist_cfg = multitask_cfg.get("distribution_head", {})
    reg_cfg = multitask_cfg.get("severity_regression", {})
    consistency_cfg = multitask_cfg.get("consistency", {})
    criterion = build_loss_function(
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
                if str(config["training"]["loss_mode"]) in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"}
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
        aux_soft_label_enabled=bool(aux_head_cfg.get("enabled", False)),
        aux_soft_label_weight=float(aux_head_cfg.get("weight", 0.0)),
        aux_soft_label_target_distribution=aux_head_cfg.get("target_distribution"),
        lambda_distribution=float(config["training"].get("lambda_distribution", dist_cfg.get("weight", 0.18))),
        lambda_severity_reg=float(config["training"].get("lambda_severity_reg", reg_cfg.get("weight", 0.08))),
        lambda_consistency_dist=float(
            config["training"].get("lambda_consistency_dist", consistency_cfg.get("weight_distribution", 0.03))
        ),
        lambda_consistency_severity=float(
            config["training"].get("lambda_consistency_severity", consistency_cfg.get("weight_severity", 0.03))
        ),
        distribution_head_enabled=bool(dist_cfg.get("enabled", False)),
        distribution_target_distribution=dist_cfg.get("target_distribution"),
        distribution_class_weights=dist_cfg.get("class_weights"),
        severity_regression_enabled=bool(reg_cfg.get("enabled", False)),
        severity_regression_class_weights=reg_cfg.get("class_weights"),
        severity_regression_loss=str(reg_cfg.get("loss", "smooth_l1")),
        consistency_enabled=bool(consistency_cfg.get("enabled", False)),
        use_focal=bool(config["training"].get("use_focal", False)),
        focal_gamma=float(config["training"].get("focal_gamma", 2.0)),
        num_classes=len(CLASS_NAMES),
    )
    if "loss_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["loss_state_dict"], strict=False)
    criterion = criterion.to(device)
    criterion.eval()
    return criterion


def unpack_model_outputs(model_output: Any) -> dict[str, Any]:
    if isinstance(model_output, torch.Tensor):
        return {
            "logits": model_output,
            "corn_logits": None,
            "aux_logits": None,
            "distribution_logits": None,
            "severity_score": None,
            "pooled_feature": None,
            "contrastive_embedding": None,
            "tau": None,
            "raw_tau": None,
            "raw_delta_tau": None,
            "head_type": "standard",
        }
    if not isinstance(model_output, dict) or "logits" not in model_output:
        raise TypeError("Model forward must return a logits tensor or a dict containing 'logits'.")
    return {
        "logits": model_output["logits"],
        "corn_logits": model_output.get("corn_logits"),
        "aux_logits": model_output.get("aux_logits"),
        "distribution_logits": model_output.get("distribution_logits"),
        "severity_score": model_output.get("severity_score"),
        "pooled_feature": model_output.get("pooled_feature"),
        "contrastive_embedding": model_output.get("contrastive_embedding"),
        "tau": model_output.get("tau"),
        "raw_tau": model_output.get("raw_tau"),
        "raw_delta_tau": model_output.get("raw_delta_tau"),
        "head_type": model_output.get("head_type", "standard"),
    }


def build_model_forward_kwargs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    def _tensor(key: str) -> torch.Tensor | None:
        value = batch.get(key)
        if value is None or not isinstance(value, torch.Tensor):
            return None
        return value.to(device, non_blocking=True)

    pre_image = _tensor("pre_image")
    if pre_image is None:
        pre_image = _tensor("local_pre")
    post_image = _tensor("post_image")
    if post_image is None:
        post_image = _tensor("local_post")
    instance_mask = _tensor("instance_mask")
    if instance_mask is None:
        instance_mask = _tensor("local_mask")
    if pre_image is None or post_image is None or instance_mask is None:
        raise KeyError("Missing required evaluation inputs in batch.")

    kwargs: dict[str, torch.Tensor] = {
        "pre_image": pre_image,
        "post_image": post_image,
        "instance_mask": instance_mask,
    }
    instance_boundary = _tensor("instance_boundary")
    if instance_boundary is None:
        instance_boundary = _tensor("local_boundary")
    if instance_boundary is not None:
        kwargs["instance_boundary"] = instance_boundary

    context_pre = _tensor("context_pre")
    context_post = _tensor("context_post")
    context_mask = _tensor("context_mask")
    context_boundary = _tensor("context_boundary")
    if context_pre is not None and context_post is not None and context_mask is not None:
        kwargs["context_pre_image"] = context_pre
        kwargs["context_post_image"] = context_post
        kwargs["context_mask"] = context_mask
        if context_boundary is not None:
            kwargs["context_boundary"] = context_boundary
    return kwargs


def summarize_distribution_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    value_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(value_array)),
        "std": float(np.std(value_array)),
        "min": float(np.min(value_array)),
        "max": float(np.max(value_array)),
        "p10": float(np.percentile(value_array, 10)),
        "p50": float(np.percentile(value_array, 50)),
        "p90": float(np.percentile(value_array, 90)),
        "support": int(value_array.size),
    }


def summarize_tau_values(values: list[float]) -> dict[str, float] | None:
    return summarize_distribution_values(values)


def compute_scalar_correlation(first: list[float], second: list[float]) -> float | None:
    if not first or not second or len(first) != len(second):
        return None
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.size < 2:
        return 0.0
    first_centered = first_array - np.mean(first_array)
    second_centered = second_array - np.mean(second_array)
    denominator = float(np.sqrt(np.sum(first_centered ** 2) * np.sum(second_centered ** 2)))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def resolve_class_probabilities(
    logits: torch.Tensor,
    *,
    head_type: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if head_type == "corn" or logits.size(1) == len(CLASS_NAMES) - 1:
        threshold_probabilities = CORNLoss.logits_to_threshold_probabilities(logits)
        class_probabilities = CORNLoss.logits_to_class_probabilities(logits)
        return class_probabilities, threshold_probabilities
    return logits.softmax(dim=1), None


def build_prediction_records(
    static_records: list[dict[str, Any]],
    probabilities: torch.Tensor,
    positions: torch.Tensor,
    *,
    threshold_probabilities: torch.Tensor | None = None,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    class_probabilities = probabilities.float().clamp_min(1e-8)
    predictions = class_probabilities.argmax(dim=1)
    confidences = class_probabilities.max(dim=1).values
    labels = torch.tensor([int(record["gt"]) for record in static_records], device=class_probabilities.device, dtype=torch.long)
    true_probabilities = class_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    difficulty = 1.0 - true_probabilities
    expected_severity = compute_expected_severity_from_probabilities(class_probabilities, positions)
    gt_severity = positions[labels]
    severity_abs_error = (expected_severity - gt_severity).abs()

    prediction_records: list[dict[str, Any]] = []
    for index, record in enumerate(static_records):
        threshold_values = None
        if threshold_probabilities is not None:
            threshold_values = [float(value) for value in threshold_probabilities[index].detach().cpu().tolist()]
        prediction_records.append(
            {
                "sample_index": int(record["sample_index"]),
                "gt": int(record["gt"]),
                "pred": int(predictions[index].detach().cpu().item()),
                "confidence": float(confidences[index].detach().cpu().item()),
                "true_probability": float(true_probabilities[index].detach().cpu().item()),
                "difficulty": float(difficulty[index].detach().cpu().item()),
                "probabilities": [float(value) for value in class_probabilities[index].detach().cpu().tolist()],
                "threshold_probabilities": threshold_values,
                "tau": record["tau"],
                "raw_delta_tau": record["raw_delta_tau"],
                "expected_severity": float(expected_severity[index].detach().cpu().item()),
                "gt_severity": float(gt_severity[index].detach().cpu().item()),
                "severity_abs_error": float(severity_abs_error[index].detach().cpu().item()),
                "meta": record["meta"],
            }
        )

    return labels.detach().cpu().tolist(), predictions.detach().cpu().tolist(), prediction_records


def compute_contrastive_embedding_stats(
    embeddings: torch.Tensor | None,
    labels: list[int],
) -> dict[str, Any] | None:
    if embeddings is None or embeddings.numel() == 0:
        return None
    embedding_tensor = embeddings.detach().float()
    norm_values = embedding_tensor.norm(dim=1)
    payload: dict[str, Any] = {
        "num_embeddings": int(embedding_tensor.size(0)),
        "embedding_dim": int(embedding_tensor.size(1)),
        "norm_mean": float(norm_values.mean().item()),
        "norm_std": float(norm_values.std(unbiased=False).item()) if norm_values.numel() > 1 else 0.0,
        "per_class_centroid_norm": {},
    }
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if label_tensor.numel() == embedding_tensor.size(0):
        centroids: list[torch.Tensor] = []
        centroid_labels: list[str] = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_mask = label_tensor == int(class_idx)
            if not class_mask.any():
                continue
            centroid = embedding_tensor[class_mask].mean(dim=0)
            payload["per_class_centroid_norm"][class_name] = float(centroid.norm().item())
            centroids.append(torch.nn.functional.normalize(centroid.unsqueeze(0), dim=1).squeeze(0))
            centroid_labels.append(class_name)
        if len(centroids) >= 2:
            centroid_matrix = torch.stack(centroids, dim=0)
            cosine = torch.matmul(centroid_matrix, centroid_matrix.t())
            payload["centroid_cosine_matrix"] = {
                row_name: {
                    col_name: float(cosine[row_idx, col_idx].item())
                    for col_idx, col_name in enumerate(centroid_labels)
                }
                for row_idx, row_name in enumerate(centroid_labels)
            }
    return payload


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DamageLossModule,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[int], list[int], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model.eval()
    criterion.eval()
    static_records: list[dict[str, Any]] = []
    tau_values: list[float] = []
    difficulty_values: list[float] = []
    logits_list: list[torch.Tensor] = []
    threshold_probability_list: list[torch.Tensor] = []
    probability_list: list[torch.Tensor] = []
    distribution_probability_list: list[torch.Tensor] = []
    severity_score_list: list[torch.Tensor] = []
    embedding_list: list[torch.Tensor] = []
    distribution_correct = 0.0
    distribution_total = 0.0
    severity_regression_abs_errors: list[float] = []
    corn_expected_severity_abs_errors: list[float] = []

    positions = criterion.get_current_positions().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            model_forward_kwargs = build_model_forward_kwargs(batch, device)
            labels = batch["label"].to(device, non_blocking=True)

            context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                model_outputs = unpack_model_outputs(model(**model_forward_kwargs))
                logits = model_outputs["corn_logits"] if model_outputs.get("corn_logits") is not None else model_outputs["logits"]
                probabilities, threshold_probabilities = resolve_class_probabilities(
                    logits,
                    head_type=str(model_outputs["head_type"]),
                )
                distribution_probabilities = None
                if model_outputs.get("distribution_logits") is not None:
                    distribution_probabilities = torch.softmax(model_outputs["distribution_logits"], dim=1)

            logits_list.append(logits.detach().cpu())
            probability_list.append(probabilities.detach().cpu())
            if distribution_probabilities is not None:
                distribution_probability_list.append(distribution_probabilities.detach().cpu())
            if threshold_probabilities is not None:
                threshold_probability_list.append(threshold_probabilities.detach().cpu())
            if model_outputs.get("contrastive_embedding") is not None:
                embedding_list.append(model_outputs["contrastive_embedding"].detach().cpu())
            if model_outputs.get("severity_score") is not None:
                severity_score_list.append(model_outputs["severity_score"].detach().cpu())

            true_probs = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
            difficulty = 1.0 - true_probs
            difficulty_values.extend(float(value) for value in difficulty.detach().cpu().tolist())
            class_positions = torch.arange(len(CLASS_NAMES), device=probabilities.device, dtype=probabilities.dtype)
            expected_severity = compute_expected_severity_from_probabilities(probabilities, class_positions)
            corn_expected_severity_abs_errors.extend(
                float(value)
                for value in (expected_severity - labels.float()).abs().detach().cpu().tolist()
            )
            if distribution_probabilities is not None:
                distribution_pred = distribution_probabilities.argmax(dim=1)
                distribution_correct += float((distribution_pred == labels).sum().item())
                distribution_total += float(labels.numel())
            if model_outputs.get("severity_score") is not None:
                severity_regression_abs_errors.extend(
                    float(value)
                    for value in (model_outputs["severity_score"].float() - labels.float()).abs().detach().cpu().tolist()
                )
            tau_cpu = None if model_outputs["tau"] is None else model_outputs["tau"].detach().cpu().tolist()
            raw_delta_tau_cpu = None if model_outputs["raw_delta_tau"] is None else model_outputs["raw_delta_tau"].detach().cpu().tolist()
            sample_indices = batch["sample_index"].cpu().tolist()
            meta_list = batch["meta"]

            if tau_cpu is not None:
                tau_values.extend(float(value) for value in tau_cpu)

            for row_idx, (sample_index, label_value, meta) in enumerate(zip(sample_indices, labels.cpu().tolist(), meta_list)):
                tau_value = None if tau_cpu is None else float(tau_cpu[row_idx])
                raw_delta_tau_value = None if raw_delta_tau_cpu is None else float(raw_delta_tau_cpu[row_idx])
                static_records.append(
                    {
                        "sample_index": int(sample_index),
                        "gt": int(label_value),
                        "tau": tau_value,
                        "raw_delta_tau": raw_delta_tau_value,
                        "meta": meta,
                    }
                )

    concatenated_probabilities = torch.cat(probability_list, dim=0) if probability_list else torch.zeros(0, len(CLASS_NAMES))
    concatenated_threshold_probabilities = (
        torch.cat(threshold_probability_list, dim=0)
        if threshold_probability_list
        else None
    )
    all_targets, all_predictions, prediction_records = build_prediction_records(
        static_records,
        concatenated_probabilities,
        positions.cpu(),
        threshold_probabilities=concatenated_threshold_probabilities,
    )
    return all_targets, all_predictions, prediction_records, {
        "tau_stats": summarize_tau_values(tau_values),
        "difficulty_stats": summarize_distribution_values(difficulty_values),
        "corr_tau_difficulty": compute_scalar_correlation(tau_values, difficulty_values),
        "distribution_head_acc": (distribution_correct / max(distribution_total, 1.0)) if distribution_total > 0 else None,
        "severity_regression_mae": (
            float(np.mean(np.asarray(severity_regression_abs_errors, dtype=np.float64)))
            if severity_regression_abs_errors
            else None
        ),
        "corn_expected_severity_mae": (
            float(np.mean(np.asarray(corn_expected_severity_abs_errors, dtype=np.float64)))
            if corn_expected_severity_abs_errors
            else None
        ),
    }, {
        "logits": torch.cat(logits_list, dim=0) if logits_list else torch.zeros(0, len(CLASS_NAMES) - 1),
        "probabilities": concatenated_probabilities,
        "distribution_probabilities": (
            torch.cat(distribution_probability_list, dim=0)
            if distribution_probability_list
            else None
        ),
        "severity_scores": (torch.cat(severity_score_list, dim=0) if severity_score_list else None),
        "threshold_probabilities": concatenated_threshold_probabilities,
        "positions": positions.cpu(),
        "static_records": static_records,
        "contrastive_embeddings": torch.cat(embedding_list, dim=0) if embedding_list else None,
    }


def save_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_prediction_records_with_calibration(
    inference_cache: dict[str, Any],
    calibration_payload: dict[str, Any] | None,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    logits = inference_cache["logits"]
    if logits.numel() == 0:
        return [], [], []
    calibrated_logits = apply_corn_calibration(logits, calibration_payload)
    calibrated_probabilities = CORNLoss.logits_to_class_probabilities(calibrated_logits)
    calibrated_threshold_probabilities = CORNLoss.logits_to_threshold_probabilities(calibrated_logits)
    return build_prediction_records(
        inference_cache["static_records"],
        calibrated_probabilities,
        inference_cache["positions"],
        threshold_probabilities=calibrated_threshold_probabilities,
    )


def select_top_records(records: list[dict[str, Any]], class_idx: int, correct: bool, top_k: int) -> list[dict[str, Any]]:
    selected = [record for record in records if record["gt"] == class_idx and ((record["pred"] == class_idx) == correct)]
    if correct:
        selected.sort(key=lambda item: item["confidence"], reverse=True)
    else:
        selected.sort(key=lambda item: (item["confidence"], 1.0 - item["true_probability"]), reverse=True)
    return selected[:top_k]


def save_visual_examples(
    dataset: Any,
    records: list[dict[str, Any]],
    eval_dir: Path,
    num_visuals_per_class: int,
) -> None:
    correct_dir = ensure_dir(eval_dir / "visuals" / "correct")
    wrong_dir = ensure_dir(eval_dir / "visuals" / "wrong")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        correct_records = select_top_records(records, class_idx, correct=True, top_k=num_visuals_per_class)
        wrong_records = select_top_records(records, class_idx, correct=False, top_k=num_visuals_per_class)

        for rank, record in enumerate(correct_records, start=1):
            sample = dataset[record["sample_index"]]
            save_prediction_visual(
                save_path=correct_dir / class_name / f"{rank:02d}_{record['meta']['tile_id']}_{record['meta']['building_idx']}.png",
                pre_image=sample["pre_image"],
                post_image=sample["post_image"],
                mask=sample["instance_mask"],
                gt_label=CLASS_NAMES[record["gt"]],
                pred_label=CLASS_NAMES[record["pred"]],
                confidence=record["confidence"],
                meta=record["meta"],
                extra_lines=[f"expected={record['expected_severity']:.4f} gt_severity={record['gt_severity']:.4f}"],
            )

        for rank, record in enumerate(wrong_records, start=1):
            sample = dataset[record["sample_index"]]
            extra_lines = [
                f"expected={record['expected_severity']:.4f} gt_severity={record['gt_severity']:.4f}",
                f"severity_abs_error={record['severity_abs_error']:.4f}",
                f"difficulty={record['difficulty']:.4f}",
            ]
            if record["tau"] is not None:
                extra_lines.append(f"tau={record['tau']:.4f}")
            save_prediction_visual(
                save_path=wrong_dir / class_name / f"{rank:02d}_{record['meta']['tile_id']}_{record['meta']['building_idx']}.png",
                pre_image=sample["pre_image"],
                post_image=sample["post_image"],
                mask=sample["instance_mask"],
                gt_label=CLASS_NAMES[record["gt"]],
                pred_label=CLASS_NAMES[record["pred"]],
                confidence=record["confidence"],
                meta=record["meta"],
                extra_lines=extra_lines,
            )


def save_hardest_minor_major_visuals(
    dataset: Any,
    records: list[dict[str, Any]],
    eval_dir: Path,
    top_k: int,
) -> None:
    candidates = [
        record
        for record in records
        if record["gt"] != record["pred"] and {record["gt"], record["pred"]} == {1, 2}
    ]
    candidates.sort(
        key=lambda item: (
            item["confidence"],
            1.0 - item["true_probability"],
            item["severity_abs_error"],
        ),
        reverse=True,
    )
    selected = candidates[:top_k]
    output_dir = ensure_dir(eval_dir / "visuals" / "hardest_minor_major")
    summary_rows = []
    for rank, record in enumerate(selected, start=1):
        sample = dataset[record["sample_index"]]
        threshold_text = (
            "threshold_probs=" + ",".join(f"{value:.3f}" for value in record["threshold_probabilities"])
            if record["threshold_probabilities"] is not None
            else None
        )
        extra_lines = [
            f"expected={record['expected_severity']:.4f} gt_severity={record['gt_severity']:.4f}",
            f"severity_abs_error={record['severity_abs_error']:.4f}",
            f"difficulty={record['difficulty']:.4f}",
        ]
        if record["tau"] is not None:
            extra_lines.append(f"tau={record['tau']:.4f}")
        if threshold_text is not None:
            extra_lines.append(threshold_text)
        save_prediction_visual(
            save_path=output_dir / f"{rank:02d}_{record['meta']['tile_id']}_{record['meta']['building_idx']}.png",
            pre_image=sample["pre_image"],
            post_image=sample["post_image"],
            mask=sample["instance_mask"],
            gt_label=CLASS_NAMES[record["gt"]],
            pred_label=CLASS_NAMES[record["pred"]],
            confidence=record["confidence"],
            meta=record["meta"],
            extra_lines=extra_lines,
        )
        summary_rows.append(
            {
                "rank": rank,
                "tile_id": record["meta"]["tile_id"],
                "building_idx": record["meta"]["building_idx"],
                "gt": CLASS_NAMES[record["gt"]],
                "pred": CLASS_NAMES[record["pred"]],
                "confidence": record["confidence"],
                "true_probability": record["true_probability"],
                "severity_abs_error": record["severity_abs_error"],
                "tau": record["tau"],
            }
        )

    write_json(eval_dir / "hardest_minor_major_confusions.json", summary_rows)
    lines = ["Hardest minor-major confusions"]
    for row in summary_rows:
        lines.append(
            f"- rank={row['rank']} tile={row['tile_id']} building={row['building_idx']} "
            f"gt={row['gt']} pred={row['pred']} conf={row['confidence']:.4f} "
            f"true_prob={row['true_probability']:.4f} severity_abs_error={row['severity_abs_error']:.4f} "
            f"tau={row['tau'] if row['tau'] is not None else 'n/a'}"
        )
    write_text(eval_dir / "hardest_minor_major_confusions.txt", "\n".join(lines) + "\n")


def find_top_confusions(confusion: list[list[int]]) -> tuple[str, int]:
    cm = np.asarray(confusion)
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    if off_diag.sum() == 0:
        return "No cross-class confusion observed.", 0
    index = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    return f"{CLASS_NAMES[index[0]]} -> {CLASS_NAMES[index[1]]}", int(off_diag[index])


def save_ordinal_exports(eval_dir: Path, criterion: DamageLossModule) -> dict[str, Any]:
    ordinal_state = criterion.export_state(CLASS_NAMES)
    write_json(
        eval_dir / "ordinal_positions.json",
        {
            "class_names": CLASS_NAMES,
            "loss_mode": ordinal_state["loss_mode"],
            "positions": ordinal_state["positions_by_class"],
            "positions_list": ordinal_state["positions"],
            "gaps": ordinal_state["gaps"],
            "tau": ordinal_state["tau"],
            "tau_statistics": ordinal_state["tau_statistics"],
            "difficulty_statistics": ordinal_state.get("difficulty_statistics"),
            "corr_tau_difficulty": ordinal_state.get("corr_tau_difficulty"),
            "tau_bounds": ordinal_state["tau_bounds"],
            "fixed_cda_alpha": ordinal_state["fixed_cda_alpha"],
            "tau_regularizer": ordinal_state.get("tau_regularizer"),
        },
    )

    csv_path = eval_dir / "ordinal_soft_targets.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt_class", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, ordinal_state["soft_target_matrix"]):
            writer.writerow([class_name, *[f"{float(value):.8f}" for value in row]])

    save_matrix_heatmap_plot(
        matrix=np.asarray(ordinal_state["soft_target_matrix"], dtype=np.float32),
        row_labels=CLASS_NAMES,
        col_labels=CLASS_NAMES,
        save_path=eval_dir / "ordinal_soft_targets.png",
        title="Ordinal Soft Targets (Reference Matrix)",
        cmap="magma",
        value_format=".3f",
        xlabel="Prediction Class",
        ylabel="Ground Truth Class",
    )
    return ordinal_state


def build_adjacency_confusion_report(metrics: dict[str, Any]) -> dict[str, Any]:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    supports = cm.sum(axis=1)
    pair_specs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    pairs: dict[str, dict[str, float | int]] = {}
    for src, dst in pair_specs:
        key = f"{CLASS_NAMES[src]}->{CLASS_NAMES[dst]}"
        support = int(supports[src])
        count = int(cm[src, dst])
        pairs[key] = {
            "count": count,
            "support": support,
            "rate_within_gt_class": float(count / support) if support > 0 else 0.0,
        }

    return {
        "pairs": pairs,
        "summary": {
            "minor_major_bidirectional": int(cm[1, 2] + cm[2, 1]),
            "no_minor_bidirectional": int(cm[0, 1] + cm[1, 0]),
            "major_destroyed_bidirectional": int(cm[2, 3] + cm[3, 2]),
            "adjacent_bidirectional_total": int(sum(pairs[key]["count"] for key in pairs)),
        },
    }


def write_adjacency_confusion_report(eval_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    report = build_adjacency_confusion_report(metrics)
    write_json(eval_dir / "adjacency_confusion.json", report)

    lines = ["Adjacency confusion report"]
    for key in [
        "no-damage->minor-damage",
        "minor-damage->no-damage",
        "minor-damage->major-damage",
        "major-damage->minor-damage",
        "major-damage->destroyed",
        "destroyed->major-damage",
    ]:
        payload = report["pairs"][key]
        lines.append(
            f"- {key}: count={payload['count']} support={payload['support']} rate={payload['rate_within_gt_class']:.4f}"
        )
    lines.append("")
    lines.append(f"- minor_major_bidirectional={report['summary']['minor_major_bidirectional']}")
    lines.append(f"- major_destroyed_bidirectional={report['summary']['major_destroyed_bidirectional']}")
    write_text(eval_dir / "adjacency_confusion.txt", "\n".join(lines) + "\n")
    return report


def build_severity_error_report(records: list[dict[str, Any]], ordinal_state: dict[str, Any]) -> dict[str, Any]:
    overall_errors = [float(record["severity_abs_error"]) for record in records]
    per_class: dict[str, dict[str, Any]] = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_records = [record for record in records if record["gt"] == class_idx]
        class_errors = [float(record["severity_abs_error"]) for record in class_records]
        class_expected = [float(record["expected_severity"]) for record in class_records]
        class_gt = [float(record["gt_severity"]) for record in class_records]
        per_class[class_name] = {
            "mae": float(np.mean(class_errors)) if class_errors else 0.0,
            "mean_expected_severity": float(np.mean(class_expected)) if class_expected else 0.0,
            "mean_gt_severity": float(np.mean(class_gt)) if class_gt else 0.0,
            "support": len(class_records),
        }

    return {
        "loss_mode": ordinal_state["loss_mode"],
        "mean_absolute_severity_error": float(np.mean(overall_errors)) if overall_errors else 0.0,
        "positions": ordinal_state["positions_by_class"],
        "per_class": per_class,
    }


def write_qwk_report(eval_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "quadratic_weighted_kappa": float(metrics["quadratic_weighted_kappa"]),
        "num_instances": int(metrics["num_instances"]),
        "model_type": metrics["model_type"],
        "loss_mode": metrics["loss_mode"],
    }
    write_json(eval_dir / "qwk.json", payload)
    return payload


def write_emd_report(
    eval_dir: Path,
    records: list[dict[str, Any]],
    positions: list[float],
) -> dict[str, Any]:
    probabilities = np.asarray([record["probabilities"] for record in records], dtype=np.float64)
    y_true = np.asarray([record["gt"] for record in records], dtype=np.int64)
    mean_emd, sample_emd = compute_emd_from_probabilities(probabilities, y_true, positions)

    for record, emd_value in zip(records, sample_emd.tolist()):
        record["emd_severity"] = float(emd_value)

    per_class: dict[str, dict[str, Any]] = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = y_true == class_idx
        values = sample_emd[mask]
        per_class[class_name] = {
            "support": int(mask.sum()),
            "mean_emd": float(values.mean()) if values.size else 0.0,
        }

    payload = {
        "emd_severity": float(mean_emd),
        "positions": {class_name: float(position) for class_name, position in zip(CLASS_NAMES, positions)},
        "per_class": per_class,
    }
    write_json(eval_dir / "emd.json", payload)
    return payload


def write_ordinal_error_profile(eval_dir: Path, y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    profile = compute_ordinal_error_profile(y_true, y_pred, CLASS_NAMES)
    write_json(eval_dir / "ordinal_error_profile.json", profile)
    lines = [
        "Ordinal error profile",
        f"- exact_match_rate={profile['exact_match_rate']:.4f}",
        f"- adjacent_error_rate={profile['adjacent_error_rate']:.4f}",
        f"- far_error_rate={profile['far_error_rate']:.4f}",
        f"- adjacent_error_share_among_errors={profile['adjacent_error_share_among_errors']:.4f}",
        f"- far_error_share_among_errors={profile['far_error_share_among_errors']:.4f}",
        f"- mean_absolute_class_distance={profile['mean_absolute_class_distance']:.4f}",
        "",
        "Distance histogram",
    ]
    for distance, count in profile["distance_histogram"].items():
        lines.append(f"- |pred-gt|={distance}: count={count}")
    lines.append("")
    lines.append("Per-class ordinal error")
    for class_name in CLASS_NAMES:
        payload = profile["per_class"][class_name]
        lines.append(
            f"- {class_name}: support={payload['support']} adjacent_rate={payload['adjacent_error_rate']:.4f} "
            f"far_rate={payload['far_error_rate']:.4f} mean_abs_distance={payload['mean_absolute_distance']:.4f}"
        )
    write_text(eval_dir / "ordinal_error_profile.txt", "\n".join(lines) + "\n")
    return profile


def write_expected_severity_stats(eval_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_values = [float(record["expected_severity"]) for record in records]
    abs_errors = [float(record["severity_abs_error"]) for record in records]
    payload = {
        "overall": {
            "expected_severity": summarize_distribution_values(expected_values),
            "severity_abs_error": summarize_distribution_values(abs_errors),
        },
        "per_class": {},
    }
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_records = [record for record in records if record["gt"] == class_idx]
        payload["per_class"][class_name] = {
            "support": len(class_records),
            "expected_severity": summarize_distribution_values(
                [float(record["expected_severity"]) for record in class_records]
            ),
            "severity_abs_error": summarize_distribution_values(
                [float(record["severity_abs_error"]) for record in class_records]
            ),
        }
    write_json(eval_dir / "expected_severity_stats.json", payload)
    return payload


def write_far_error_breakdown(eval_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    payload = {
        "overall_far_error_rate": float(metrics.get("far_error_rate", 0.0)),
        "pairs": {},
    }
    for gt_idx, gt_name in enumerate(CLASS_NAMES):
        support = int(confusion[gt_idx].sum())
        for pred_idx, pred_name in enumerate(CLASS_NAMES):
            if abs(gt_idx - pred_idx) < 2:
                continue
            count = int(confusion[gt_idx, pred_idx])
            key = f"{gt_name}->{pred_name}"
            payload["pairs"][key] = {
                "count": count,
                "support": support,
                "rate_within_gt_class": float(count / support) if support > 0 else 0.0,
            }
    write_json(eval_dir / "far_error_breakdown.json", payload)
    return payload


def write_minor_major_boundary_report(
    eval_dir: Path,
    records: list[dict[str, Any]],
    *,
    calibrated: bool,
    stage2_checkpoint: bool,
    top_k: int = 20,
) -> dict[str, Any]:
    boundary_records = [record for record in records if record["gt"] in {1, 2}]
    if boundary_records:
        subset_true = [int(record["gt"]) for record in boundary_records]
        subset_pred = [int(record["pred"]) for record in boundary_records]
        subset_metrics, _ = compute_classification_metrics(subset_true, subset_pred, CLASS_NAMES)
    else:
        subset_metrics = {
            "overall_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_accuracy": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "adjacent_error_rate": 0.0,
            "far_error_rate": 0.0,
            "per_class": {},
            "class_counts": {},
            "prediction_distribution": {},
            "confusion_matrix": [[0, 0, 0, 0] for _ in CLASS_NAMES],
            "ordinal_error_profile": {},
        }
    confusion_records = [
        record
        for record in boundary_records
        if {record["gt"], record["pred"]} == {1, 2} and record["gt"] != record["pred"]
    ]
    confusion_records.sort(
        key=lambda item: (
            item["confidence"],
            1.0 - item["true_probability"],
            item["severity_abs_error"],
        ),
        reverse=True,
    )
    payload = {
        "support": len(boundary_records),
        "subset_metrics": subset_metrics,
        "minor_to_major_count": int(sum(1 for record in boundary_records if record["gt"] == 1 and record["pred"] == 2)),
        "major_to_minor_count": int(sum(1 for record in boundary_records if record["gt"] == 2 and record["pred"] == 1)),
        "hardest_minor_major_confusions": [
            {
                "sample_index": int(record["sample_index"]),
                "tile_id": record["meta"]["tile_id"],
                "building_idx": record["meta"]["building_idx"],
                "gt": CLASS_NAMES[record["gt"]],
                "pred": CLASS_NAMES[record["pred"]],
                "confidence": float(record["confidence"]),
                "true_probability": float(record["true_probability"]),
                "severity_abs_error": float(record["severity_abs_error"]),
            }
            for record in confusion_records[:top_k]
        ],
        "used_calibration": bool(calibrated),
        "used_stage2_checkpoint": bool(stage2_checkpoint),
    }
    write_json(eval_dir / "minor_major_boundary_report.json", payload)
    return payload


def write_calibration_before_after(
    eval_dir: Path,
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any] | None,
    calibration_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "calibration": calibration_payload,
        "before": None if before_metrics is None else {
            "macro_f1": float(before_metrics["macro_f1"]),
            "quadratic_weighted_kappa": float(before_metrics["quadratic_weighted_kappa"]),
            "far_error_rate": float(before_metrics.get("far_error_rate", 0.0)),
            "adjacent_error_rate": float(before_metrics.get("adjacent_error_rate", 0.0)),
        },
        "after": None if after_metrics is None else {
            "macro_f1": float(after_metrics["macro_f1"]),
            "quadratic_weighted_kappa": float(after_metrics["quadratic_weighted_kappa"]),
            "far_error_rate": float(after_metrics.get("far_error_rate", 0.0)),
            "adjacent_error_rate": float(after_metrics.get("adjacent_error_rate", 0.0)),
        },
    }
    write_json(eval_dir / "calibration_before_after.json", payload)
    return payload


def write_tau_statistics(eval_dir: Path, inference_summary: dict[str, Any], ordinal_state: dict[str, Any]) -> dict[str, Any]:
    tau_stats = inference_summary.get("tau_stats")
    payload = {
        "inference_tau_statistics": tau_stats,
        "inference_difficulty_statistics": inference_summary.get("difficulty_stats"),
        "corr_tau_difficulty": inference_summary.get("corr_tau_difficulty"),
        "criterion_tau_statistics": ordinal_state.get("tau_statistics"),
        "criterion_difficulty_statistics": ordinal_state.get("difficulty_statistics"),
        "criterion_corr_tau_difficulty": ordinal_state.get("corr_tau_difficulty"),
        "tau_reference": ordinal_state.get("tau"),
        "tau_bounds": ordinal_state.get("tau_bounds"),
        "collapsed_near_constant": bool(tau_stats is not None and tau_stats["std"] < 1e-4),
    }
    write_json(eval_dir / "tau_stats.json", payload)
    return payload


def save_empty_tau_plot(save_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tau_histogram(eval_dir: Path, tau_values: list[float]) -> None:
    save_path = eval_dir / "tau_histogram.png"
    if not tau_values:
        save_empty_tau_plot(save_path, "Tau Histogram", "No tau values available")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(tau_values, dtype=np.float64), bins=24, color="#2b6cb0", alpha=0.85, edgecolor="white")
    ax.set_title("Adaptive Tau Histogram")
    ax.set_xlabel("tau")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tau_boxplot(eval_dir: Path, tau_groups: dict[str, list[float]]) -> None:
    save_path = eval_dir / "tau_boxplot.png"
    valid_groups = [(name, values) for name, values in tau_groups.items() if values]
    if not valid_groups:
        save_empty_tau_plot(save_path, "Tau Boxplot", "No tau values available")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([values for _, values in valid_groups], labels=[name for name, _ in valid_groups], patch_artist=True)
    ax.set_title("Adaptive Tau by GT Class")
    ax.set_xlabel("GT class")
    ax.set_ylabel("tau")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tau_difficulty_scatter(eval_dir: Path, tau_values: list[float], difficulty_values: list[float]) -> None:
    save_path = eval_dir / "tau_vs_difficulty_scatter.png"
    if not tau_values or not difficulty_values or len(tau_values) != len(difficulty_values):
        save_empty_tau_plot(save_path, "Tau vs Difficulty", "No tau/difficulty pairs available")
        return

    x = np.asarray(difficulty_values, dtype=np.float64)
    y = np.asarray(tau_values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(x, y, s=14, alpha=0.35, color="#c05621", edgecolors="none")
    if x.size >= 2 and np.std(x) > 1e-8:
        coeffs = np.polyfit(x, y, deg=1)
        trend_x = np.linspace(float(x.min()), float(x.max()), 100)
        trend_y = coeffs[0] * trend_x + coeffs[1]
        ax.plot(trend_x, trend_y, color="#1a202c", linewidth=2.0)
    ax.set_title("Adaptive Tau vs Difficulty")
    ax.set_xlabel("difficulty = 1 - p(gt)")
    ax.set_ylabel("tau")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_tau_group_payload(records: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = {}
    for record in records:
        tau = record.get("tau")
        if tau is None:
            continue
        group_key = str(key_fn(record))
        groups.setdefault(group_key, []).append(float(tau))

    payload: dict[str, dict[str, Any]] = {}
    for key, values in groups.items():
        payload[key] = summarize_tau_values(values) or {"support": 0}
    return payload


def export_tau_diagnostics(eval_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    tau_values = [float(record["tau"]) for record in records if record.get("tau") is not None]
    difficulty_values = [float(record["difficulty"]) for record in records if record.get("tau") is not None]
    tau_by_gt_class = {
        class_name: summarize_tau_values(
            [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] == class_idx]
        ) or {"support": 0}
        for class_idx, class_name in enumerate(CLASS_NAMES)
    }
    tau_by_correctness = {
        "correct": summarize_tau_values(
            [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] == record["pred"]]
        ) or {"support": 0},
        "wrong": summarize_tau_values(
            [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] != record["pred"]]
        ) or {"support": 0},
    }

    write_json(eval_dir / "tau_by_gt_class.json", tau_by_gt_class)
    write_json(eval_dir / "tau_by_correctness.json", tau_by_correctness)
    save_tau_histogram(eval_dir, tau_values)
    save_tau_boxplot(
        eval_dir,
        {
            class_name: [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] == class_idx]
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
    )
    tau_difficulty_points = [
        {
            "sample_index": int(record["sample_index"]),
            "gt": int(record["gt"]),
            "pred": int(record["pred"]),
            "correct": bool(record["gt"] == record["pred"]),
            "tau": float(record["tau"]),
            "difficulty": float(record["difficulty"]),
        }
        for record in records
        if record.get("tau") is not None
    ]
    tau_vs_difficulty_payload = {
        "summary": {
            "tau_stats": summarize_tau_values(tau_values),
            "difficulty_stats": summarize_distribution_values(difficulty_values),
            "corr_tau_difficulty": compute_scalar_correlation(tau_values, difficulty_values),
            "support": len(tau_difficulty_points),
        },
        "points": tau_difficulty_points,
    }
    write_json(eval_dir / "tau_vs_difficulty.json", tau_vs_difficulty_payload)
    save_tau_difficulty_scatter(eval_dir, tau_values, difficulty_values)
    return {
        "all_tau_values": tau_values,
        "all_difficulty_values": difficulty_values,
        "tau_by_gt_class": tau_by_gt_class,
        "tau_by_correctness": tau_by_correctness,
        "tau_vs_difficulty": tau_vs_difficulty_payload,
    }


def write_adaptive_tau_analysis(
    eval_dir: Path,
    loss_mode: str,
    tau_report: dict[str, Any],
    tau_diag: dict[str, Any],
) -> None:
    overall = tau_report.get("inference_tau_statistics")
    lines = ["# Adaptive Tau Analysis", ""]

    if overall is None or loss_mode not in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}:
        lines.append("- Current run does not expose adaptive sample-level tau diagnostics.")
        write_text(eval_dir / "adaptive_tau_analysis.md", "\n".join(lines) + "\n")
        return

    tau_difficulty_summary = tau_diag.get("tau_vs_difficulty", {}).get("summary", {})
    lines.extend(
        [
            "## Overall",
            (
                f"- mean={overall['mean']:.4f}, std={overall['std']:.6f}, "
                f"min={overall['min']:.4f}, p10={overall['p10']:.4f}, "
                f"p50={overall['p50']:.4f}, p90={overall['p90']:.4f}, max={overall['max']:.4f}"
            ),
        ]
    )
    if overall["std"] < 1e-4:
        lines.append("- adaptive tau remains near-constant and is effectively collapsed")
    else:
        lines.append("- adaptive tau remains non-constant across evaluation samples")

    corr_tau_difficulty = tau_difficulty_summary.get("corr_tau_difficulty")
    if corr_tau_difficulty is None:
        lines.append("- tau vs difficulty correlation: unavailable")
    elif corr_tau_difficulty > 0.10:
        lines.append(f"- tau is positively correlated with difficulty (corr={corr_tau_difficulty:.4f})")
    elif corr_tau_difficulty < -0.10:
        lines.append(f"- tau is negatively correlated with difficulty (corr={corr_tau_difficulty:.4f})")
    else:
        lines.append(f"- tau shows only weak correlation with difficulty (corr={corr_tau_difficulty:.4f})")

    lines.extend(["", "## By Correctness"])
    tau_by_correctness = tau_diag["tau_by_correctness"]
    for key in ["correct", "wrong"]:
        payload = tau_by_correctness.get(key, {"support": 0})
        if payload.get("support", 0) <= 0:
            lines.append(f"- {key}: no samples")
            continue
        lines.append(
            f"- {key}: support={payload['support']} mean={payload['mean']:.4f} std={payload['std']:.4f} "
            f"p10={payload['p10']:.4f} p50={payload['p50']:.4f} p90={payload['p90']:.4f}"
        )

    correct_payload = tau_by_correctness.get("correct")
    wrong_payload = tau_by_correctness.get("wrong")
    if correct_payload and wrong_payload and correct_payload.get("support", 0) > 0 and wrong_payload.get("support", 0) > 0:
        mean_gap = float(wrong_payload["mean"]) - float(correct_payload["mean"])
        abs_gap = abs(mean_gap)
        if mean_gap > 0.02:
            lines.append(f"- wrong samples have clearly larger tau on average (mean gap={mean_gap:.4f})")
        elif mean_gap > 0.0:
            lines.append(f"- wrong samples have slightly larger tau on average (mean gap={mean_gap:.4f})")
        elif abs_gap > 0.02:
            lines.append(f"- wrong samples are not larger; tau separates in the opposite direction (mean gap={mean_gap:.4f})")
        else:
            lines.append(f"- tau shows only weak correctness-dependent separation (mean gap={mean_gap:.4f})")

    lines.extend(["", "## By GT Class"])
    tau_by_gt_class = tau_diag["tau_by_gt_class"]
    gt_means: list[float] = []
    for class_name in CLASS_NAMES:
        payload = tau_by_gt_class.get(class_name, {"support": 0})
        if payload.get("support", 0) <= 0:
            lines.append(f"- {class_name}: no samples")
            continue
        gt_means.append(float(payload["mean"]))
        lines.append(
            f"- {class_name}: support={payload['support']} mean={payload['mean']:.4f} std={payload['std']:.4f} "
            f"p10={payload['p10']:.4f} p50={payload['p50']:.4f} p90={payload['p90']:.4f}"
        )
    if gt_means:
        class_spread = max(gt_means) - min(gt_means)
        if class_spread > 0.02:
            lines.append(f"- tau varies meaningfully across GT classes (mean spread={class_spread:.4f})")
        else:
            lines.append(f"- tau varies only weakly across GT classes (mean spread={class_spread:.4f})")

    write_text(eval_dir / "adaptive_tau_analysis.md", "\n".join(lines) + "\n")


def load_loss_mode_reports(current_run_dir: Path) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    exp_root = current_run_dir.parent.parent
    candidates = [
        *((loss_mode, exp_root / "oracle_mcd" / loss_mode / "eval") for loss_mode in STANDARD_ORACLE_LOSS_MODES),
        ("corn", exp_root / "oracle_mcd_corn" / "corn" / "eval"),
        ("corn_adaptive_tau_safe", exp_root / "oracle_mcd_corn" / "corn_adaptive_tau_safe" / "eval"),
        ("corn_ordinal_multitask_v1", exp_root / "oracle_mcd_corn" / "corn_ordinal_multitask_v1" / "eval"),
    ]

    for loss_mode, eval_dir in candidates:
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        payload: dict[str, Any] = {}
        with metrics_path.open("r", encoding="utf-8") as f:
            payload["metrics"] = json.load(f)

        for key, filename in [
            ("adjacency", "adjacency_confusion.json"),
            ("severity", "severity_error.json"),
            ("ordinal", "ordinal_positions.json"),
            ("qwk", "qwk.json"),
            ("emd", "emd.json"),
            ("ordinal_error", "ordinal_error_profile.json"),
            ("tau", "tau_stats.json"),
        ]:
            file_path = eval_dir / filename
            if file_path.exists():
                with file_path.open("r", encoding="utf-8") as f:
                    payload[key] = json.load(f)
        comparisons[loss_mode] = payload
    return comparisons


def summarize_comparison(before_name: str, after_name: str, comparisons: dict[str, dict[str, Any]]) -> str:
    before = comparisons.get(before_name)
    after = comparisons.get(after_name)
    if before is None or after is None:
        return f"- {after_name} vs {before_name}: comparison unavailable until both evaluations exist."

    before_adj = before["adjacency"]["summary"]["minor_major_bidirectional"]
    after_adj = after["adjacency"]["summary"]["minor_major_bidirectional"]
    before_f1 = before["metrics"]["macro_f1"]
    after_f1 = after["metrics"]["macro_f1"]
    before_qwk = before.get("qwk", {}).get("quadratic_weighted_kappa", before["metrics"].get("quadratic_weighted_kappa", 0.0))
    after_qwk = after.get("qwk", {}).get("quadratic_weighted_kappa", after["metrics"].get("quadratic_weighted_kappa", 0.0))

    if after_adj < before_adj:
        verdict = "improved"
    elif after_adj > before_adj:
        verdict = "worsened"
    else:
        verdict = "held flat"

    return (
        f"- {after_name} vs {before_name}: minor/major bidirectional confusion {verdict} "
        f"({before_adj} -> {after_adj}), macro_f1 {before_f1:.4f} -> {after_f1:.4f}, "
        f"QWK {before_qwk:.4f} -> {after_qwk:.4f}."
    )


def summarize_learned_gaps(comparisons: dict[str, dict[str, Any]]) -> list[str]:
    for key in ["adaptive_ucl_cda_v3", "adaptive_ucl_cda_v2", "adaptive_ucl_cda", "learnable_cda"]:
        payload = comparisons.get(key)
        if payload is None or "ordinal" not in payload:
            continue
        ordinal = payload["ordinal"]
        gaps = ordinal["gaps"]
        adjacent_gaps = {
            "no-damage <-> minor-damage": float(gaps["gap_01"]),
            "minor-damage <-> major-damage": float(gaps["gap_12"]),
            "major-damage <-> destroyed": float(gaps["gap_23"]),
        }
        smallest_pair = min(adjacent_gaps, key=adjacent_gaps.get)
        largest_pair = max(adjacent_gaps, key=adjacent_gaps.get)
        tau_stats = ordinal.get("tau_statistics")
        lines = [
            (
                f"- {key} gaps: gap_01={float(gaps['gap_01']):.4f}, "
                f"gap_12={float(gaps['gap_12']):.4f}, gap_23={float(gaps['gap_23']):.4f}."
            ),
            f"- Smallest adjacent gap: {smallest_pair}.",
            f"- Largest adjacent gap: {largest_pair}.",
        ]
        if tau_stats is not None:
            lines.append(
                f"- Tau stats: mean={tau_stats['mean']:.4f}, std={tau_stats['std']:.4f}, "
                f"p10={tau_stats['p10']:.4f}, p50={tau_stats['p50']:.4f}, p90={tau_stats['p90']:.4f}."
            )
        return lines
    return ["- Learnable ordinal gaps: not available yet."]


def write_analysis_report(
    eval_dir: Path,
    metrics: dict[str, Any],
    adjacency_report: dict[str, Any],
    severity_report: dict[str, Any],
    ordinal_state: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    emd_report: dict[str, Any],
) -> None:
    top_confusion, confusion_count = find_top_confusions(metrics["confusion_matrix"])
    lines = [
        "# Oracle Upper-Bound Analysis",
        "",
        "## Core Reading",
        f"- Strongest confusion pair: {top_confusion} ({confusion_count} samples).",
        f"- minor-damage vs major-damage cross-confusions: {adjacency_report['summary']['minor_major_bidirectional']}.",
        f"- Mean absolute severity error: {severity_report['mean_absolute_severity_error']:.4f}.",
        f"- Quadratic weighted kappa: {metrics['quadratic_weighted_kappa']:.4f}.",
        f"- EMD / Wasserstein-1 severity error: {emd_report['emd_severity']:.4f}.",
        "",
        "## Learned Severity Axis",
    ]
    for class_name, position in ordinal_state["positions_by_class"].items():
        lines.append(f"- {class_name}: {position:.4f}")
    if ordinal_state["tau"] is not None:
        lines.append(f"- tau reference: {ordinal_state['tau']:.4f}")
    if ordinal_state.get("tau_statistics") is not None:
        tau_stats = ordinal_state["tau_statistics"]
        lines.append(
            f"- tau stats: mean={tau_stats['mean']:.4f}, std={tau_stats['std']:.4f}, "
            f"p10={tau_stats['p10']:.4f}, p50={tau_stats['p50']:.4f}, p90={tau_stats['p90']:.4f}"
        )
    if ordinal_state.get("difficulty_statistics") is not None:
        difficulty_stats = ordinal_state["difficulty_statistics"]
        corr_tau_difficulty = ordinal_state.get("corr_tau_difficulty")
        corr_text = "n/a" if corr_tau_difficulty is None else f"{corr_tau_difficulty:.4f}"
        lines.append(
            f"- difficulty stats: mean={difficulty_stats['mean']:.4f}, std={difficulty_stats['std']:.4f}, "
            f"corr_tau_difficulty={corr_text}"
        )

    lines.extend(["", "## Loss Comparison"])
    if comparisons:
        for loss_mode in STANDARD_ORACLE_LOSS_MODES + ["corn", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"]:
            payload = comparisons.get(loss_mode)
            if payload is None:
                lines.append(f"- {loss_mode}: metrics not available yet.")
                continue
            metrics_payload = payload["metrics"]
            adjacency_payload = payload.get("adjacency", {"summary": {"minor_major_bidirectional": 0}})
            severity_payload = payload.get("severity", {"mean_absolute_severity_error": 0.0})
            qwk_payload = payload.get("qwk", {"quadratic_weighted_kappa": metrics_payload.get("quadratic_weighted_kappa", 0.0)})
            emd_payload = payload.get("emd", {"emd_severity": 0.0})
            lines.append(
                f"- {loss_mode}: macro_f1={metrics_payload['macro_f1']:.4f}, "
                f"balanced_accuracy={metrics_payload['balanced_accuracy']:.4f}, "
                f"qwk={qwk_payload['quadratic_weighted_kappa']:.4f}, "
                f"emd={emd_payload['emd_severity']:.4f}, "
                f"minor_major_bidirectional={adjacency_payload['summary']['minor_major_bidirectional']}, "
                f"severity_mae={severity_payload['mean_absolute_severity_error']:.4f}"
            )
        lines.append("")
        lines.append(summarize_comparison("weighted_ce", "fixed_cda", comparisons))
        lines.append(summarize_comparison("fixed_cda", "learnable_cda", comparisons))
        lines.append(summarize_comparison("learnable_cda", "adaptive_ucl_cda", comparisons))
        lines.append(summarize_comparison("adaptive_ucl_cda", "adaptive_ucl_cda_v2", comparisons))
        lines.append(summarize_comparison("adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", comparisons))
        lines.append(summarize_comparison("corn_adaptive_tau_safe", "corn_ordinal_multitask_v1", comparisons))
        lines.extend(summarize_learned_gaps(comparisons))
    else:
        lines.append("- Only the current loss mode has been evaluated so far.")

    write_text(eval_dir / "oracle_upper_bound_analysis.md", "\n".join(lines) + "\n")


def print_ordinal_state(ordinal_state: dict[str, Any]) -> None:
    print("Ordinal severity axis:")
    for class_name, position in ordinal_state["positions_by_class"].items():
        print(f"  {class_name} -> {position:.4f}")
    if ordinal_state["tau"] is not None:
        print(f"Reference tau: {ordinal_state['tau']:.4f}")
    if ordinal_state.get("tau_statistics") is not None:
        tau_stats = ordinal_state["tau_statistics"]
        print(
            "Tau stats: "
            f"mean={tau_stats['mean']:.4f}, std={tau_stats['std']:.4f}, "
            f"p10={tau_stats['p10']:.4f}, p50={tau_stats['p50']:.4f}, p90={tau_stats['p90']:.4f}"
        )


def main() -> None:
    args = parse_args()
    direct_run = len(sys.argv) == 1
    config = apply_overrides(load_config(args.config), args, default_loss_mode="corn_adaptive_tau_safe")

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        checkpoint_default_loss_mode = "corn_adaptive_tau_safe"
        training_cfg = checkpoint["config"].get("training", {})
        if "loss_mode" not in training_cfg and "loss_state_dict" not in checkpoint:
            checkpoint_default_loss_mode = "weighted_ce"
        config = apply_overrides(copy.deepcopy(checkpoint["config"]), args, default_loss_mode=checkpoint_default_loss_mode)

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    loader, dataset = make_dataloader(config)
    model = build_model(config)
    resolved_encoder_in_channels = int(
        getattr(model, "encoder_in_channels", config.get("geometry_prior", {}).get("encoder_in_channels", 4))
    )
    model_load_status = load_state_dict_with_fallback(
        model,
        checkpoint["model_state_dict"],
        strict=True,
        description="evaluation model",
    )
    if model_load_status["error"] is not None:
        print(model_load_status["error"])
    if model_load_status["missing_keys"] or model_load_status["unexpected_keys"]:
        print(
            "Evaluation model load compatibility "
            f"missing={model_load_status['missing_keys']} "
            f"unexpected={model_load_status['unexpected_keys']}"
        )
    model = model.to(device)
    criterion = build_loss_module(config, checkpoint, device=device)

    run_dir = checkpoint_path.parent.parent
    eval_dir = Path(args.save_dir) if args.save_dir is not None else run_dir / "eval"
    ensure_dir(eval_dir)
    write_yaml(eval_dir / "eval_config.yaml", config)

    print(f"DIRECT_RUN={direct_run}")
    print(f"Device: {device}")
    print(f"Dataset instances: {len(dataset)}")
    print(f"Cache: {dataset.cache_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model type: {config['model']['model_type']}")
    print(f"Head type: {config['model'].get('head_type', 'standard')}")
    print(f"Backbone: {config['model'].get('backbone', 'convnextv2_tiny')}")
    print(f"Pretrained: {bool(config['model'].get('pretrained', True))}")
    print(f"encoder_in_channels: {resolved_encoder_in_channels}")
    print(f"Eval source: {config['data'].get('eval_source', 'xbd_only')}")
    print(f"Source ratio: {config['data'].get('source_ratio', {'xbd': 1, 'bright': 1})}")
    print(f"Loss mode: {config['training']['loss_mode']}")
    print(f"Dual scale enabled: {bool(config.get('input_mode', {}).get('use_dual_scale', False))}")
    print(f"Context branch enabled: {bool(config['model'].get('use_context_branch', False))}")
    print(f"Boundary prior enabled: {bool(config.get('geometry_prior', {}).get('use_boundary_prior', False))}")
    if str(config["model"].get("backbone", "")).startswith("convnextv2"):
        print("Backbone family: ConvNeXt V2 (timm features_only)")
    print(f"Source counts: {getattr(dataset, 'source_counts', {})}")

    y_true, y_pred, prediction_records, inference_summary, inference_cache = run_inference(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    uncalibrated_metrics, uncalibrated_report = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)
    for payload in [uncalibrated_metrics]:
        payload["model_type"] = config["model"]["model_type"]
        payload["head_type"] = config["model"].get("head_type", "standard")
        payload["backbone"] = config["model"].get("backbone", "convnextv2_tiny")
        payload["loss_mode"] = config["training"]["loss_mode"]
        payload["checkpoint"] = str(checkpoint_path)
        payload["num_instances"] = len(dataset)
        payload["eval_source"] = config["data"].get("eval_source", "xbd_only")
        payload["dual_scale_enabled"] = bool(config.get("input_mode", {}).get("use_dual_scale", False))
        payload["boundary_prior_enabled"] = bool(config.get("geometry_prior", {}).get("use_boundary_prior", False))

    calibration_requested = bool(config["evaluation"].get("enable_threshold_calibration", False))
    calibration_fit_requested = bool(config["evaluation"].get("fit_calibration_on_val", False))
    calibration_file_path = args.calibration_file
    calibration_payload = None
    calibration_source = None
    if calibration_requested or calibration_fit_requested or calibration_file_path is not None:
        if config["model"].get("head_type", "standard") != "corn":
            raise ValueError("Threshold calibration currently supports CORN checkpoints only.")
        if calibration_file_path is not None:
            with Path(calibration_file_path).open("r", encoding="utf-8") as f:
                calibration_payload = json.load(f)
            calibration_source = "file"
        elif calibration_fit_requested or calibration_requested:
            calibration_payload = fit_corn_threshold_calibration(
                inference_cache["logits"],
                torch.tensor(y_true, dtype=torch.long),
                calibration_type=str(config["evaluation"].get("calibration_type", "temperature_plus_bias")),
                max_iter=int(config["evaluation"].get("calibration_max_iter", 200)),
                lr=float(config["evaluation"].get("calibration_lr", 0.01)),
            )
            calibration_source = "fit_on_val"
        if calibration_payload is not None:
            write_json(eval_dir / "calibration.json", calibration_payload)

    calibrated_metrics = None
    calibrated_report = None
    calibrated_records = None
    calibrated_true = None
    calibrated_pred = None
    if calibration_payload is not None:
        calibrated_true, calibrated_pred, calibrated_records = rebuild_prediction_records_with_calibration(
            inference_cache,
            calibration_payload,
        )
        calibrated_metrics, calibrated_report = compute_classification_metrics(calibrated_true, calibrated_pred, CLASS_NAMES)
        calibrated_metrics["model_type"] = config["model"]["model_type"]
        calibrated_metrics["head_type"] = config["model"].get("head_type", "standard")
        calibrated_metrics["loss_mode"] = config["training"]["loss_mode"]
        calibrated_metrics["checkpoint"] = str(checkpoint_path)
        calibrated_metrics["num_instances"] = len(dataset)

    metrics = calibrated_metrics if calibrated_metrics is not None else uncalibrated_metrics
    report = calibrated_report if calibrated_report is not None else uncalibrated_report
    prediction_records = calibrated_records if calibrated_records is not None else prediction_records
    y_true = calibrated_true if calibrated_true is not None else y_true
    y_pred = calibrated_pred if calibrated_pred is not None else y_pred
    metrics = json.loads(json.dumps(metrics))
    metrics["current_metrics_source"] = "calibrated" if calibrated_metrics is not None else "uncalibrated"
    metrics["calibration"] = {
        "enabled": calibration_payload is not None,
        "source": calibration_source,
        "file": calibration_file_path,
    }
    metrics["uncalibrated_metrics"] = json.loads(json.dumps(uncalibrated_metrics))
    metrics["calibrated_metrics"] = None if calibrated_metrics is None else json.loads(json.dumps(calibrated_metrics))
    used_stage2_checkpoint = bool(checkpoint.get("stage_name") == "stage2" or checkpoint_path.parent.parent.name == "stage2")
    metrics["used_stage2_checkpoint"] = used_stage2_checkpoint
    metrics["distribution_head_acc"] = inference_summary.get("distribution_head_acc")
    metrics["severity_regression_mae"] = inference_summary.get("severity_regression_mae")
    metrics["corn_expected_severity_mae"] = inference_summary.get("corn_expected_severity_mae")

    ordinal_state = save_ordinal_exports(eval_dir, criterion)
    adjacency_report = write_adjacency_confusion_report(eval_dir, metrics)
    severity_report = build_severity_error_report(prediction_records, ordinal_state)
    write_json(eval_dir / "severity_error.json", severity_report)
    qwk_report = write_qwk_report(eval_dir, metrics)
    emd_report = write_emd_report(eval_dir, prediction_records, ordinal_state["positions"])
    metrics["emd_severity"] = float(emd_report["emd_severity"])
    ordinal_error_profile = write_ordinal_error_profile(eval_dir, y_true, y_pred)
    metrics["adjacent_error_rate"] = float(ordinal_error_profile["adjacent_error_rate"])
    metrics["far_error_rate"] = float(ordinal_error_profile["far_error_rate"])
    tau_report = write_tau_statistics(eval_dir, inference_summary, ordinal_state)
    tau_diag = export_tau_diagnostics(eval_dir, prediction_records)
    write_adaptive_tau_analysis(eval_dir, str(config["training"]["loss_mode"]), tau_report, tau_diag)
    contrastive_embedding_stats = compute_contrastive_embedding_stats(inference_cache.get("contrastive_embeddings"), y_true)
    write_json(eval_dir / "contrastive_embedding_stats.json", contrastive_embedding_stats)
    far_error_breakdown = write_far_error_breakdown(eval_dir, metrics)
    expected_severity_stats = write_expected_severity_stats(eval_dir, prediction_records)
    minor_major_boundary_report = write_minor_major_boundary_report(
        eval_dir,
        prediction_records,
        calibrated=calibration_payload is not None,
        stage2_checkpoint=used_stage2_checkpoint,
        top_k=int(config["evaluation"]["num_visuals_per_class"]),
    )
    write_json(
        eval_dir / "multitask_diagnostics.json",
        {
            "distribution_head_acc": metrics.get("distribution_head_acc"),
            "severity_regression_mae": metrics.get("severity_regression_mae"),
            "corn_expected_severity_mae": metrics.get("corn_expected_severity_mae"),
        },
    )
    if calibration_requested or calibration_fit_requested or calibration_payload is not None:
        write_calibration_before_after(eval_dir, uncalibrated_metrics, calibrated_metrics, calibration_payload)

    write_json(eval_dir / "metrics.json", metrics)
    write_text(eval_dir / "classification_report.txt", report)
    save_confusion_matrix_plot(metrics["confusion_matrix"], CLASS_NAMES, eval_dir / "confusion_matrix.png")
    save_prediction_records(eval_dir / "predictions.jsonl", prediction_records)
    save_visual_examples(
        dataset=dataset,
        records=prediction_records,
        eval_dir=eval_dir,
        num_visuals_per_class=int(config["evaluation"]["num_visuals_per_class"]),
    )
    save_hardest_minor_major_visuals(
        dataset=dataset,
        records=prediction_records,
        eval_dir=eval_dir,
        top_k=int(config["evaluation"]["num_visuals_per_class"]),
    )

    comparisons = load_loss_mode_reports(run_dir)
    comparisons[config["training"]["loss_mode"]] = {
        "metrics": metrics,
        "adjacency": adjacency_report,
        "severity": severity_report,
        "ordinal": json.loads(json.dumps(ordinal_state)),
        "qwk": qwk_report,
        "emd": emd_report,
        "ordinal_error": ordinal_error_profile,
        "tau": tau_report,
        "far_error_breakdown": far_error_breakdown,
        "expected_severity": expected_severity_stats,
        "minor_major_boundary": minor_major_boundary_report,
    }
    write_analysis_report(eval_dir, metrics, adjacency_report, severity_report, ordinal_state, comparisons, emd_report)

    print("Evaluation complete.")
    print(
        f"overall_accuracy={metrics['overall_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"qwk={metrics['quadratic_weighted_kappa']:.4f} "
        f"emd={metrics['emd_severity']:.4f}"
    )
    print(
        f"adjacent_error_rate={metrics['adjacent_error_rate']:.4f} "
        f"far_error_rate={metrics['far_error_rate']:.4f}"
    )
    print(
        f"metrics_source={metrics['current_metrics_source']} "
        f"calibration_enabled={metrics['calibration']['enabled']} "
        f"stage2_checkpoint={metrics['used_stage2_checkpoint']}"
    )
    if metrics.get("distribution_head_acc") is not None:
        print(f"distribution_head_acc={float(metrics['distribution_head_acc']):.4f}")
    if metrics.get("severity_regression_mae") is not None:
        print(f"severity_regression_mae={float(metrics['severity_regression_mae']):.4f}")
    if metrics.get("corn_expected_severity_mae") is not None:
        print(f"corn_expected_severity_mae={float(metrics['corn_expected_severity_mae']):.4f}")
    if tau_report.get("inference_tau_statistics") is not None:
        tau_stats = tau_report["inference_tau_statistics"]
        print(
            f"tau_mean={tau_stats['mean']:.4f} tau_std={tau_stats['std']:.6f} "
            f"tau_p10={tau_stats['p10']:.4f} tau_p50={tau_stats['p50']:.4f} tau_p90={tau_stats['p90']:.4f}"
        )
    if tau_report.get("inference_difficulty_statistics") is not None:
        difficulty_stats = tau_report["inference_difficulty_statistics"]
        corr_tau_difficulty = tau_report.get("corr_tau_difficulty")
        corr_text = "n/a" if corr_tau_difficulty is None else f"{corr_tau_difficulty:.4f}"
        print(
            f"difficulty_mean={difficulty_stats['mean']:.4f} difficulty_std={difficulty_stats['std']:.4f} "
            f"corr_tau_difficulty={corr_text}"
        )
    print(f"Mean absolute severity error: {severity_report['mean_absolute_severity_error']:.4f}")
    print_ordinal_state(ordinal_state)
    print(f"Artifacts saved to: {eval_dir}")


if __name__ == "__main__":
    main()
