from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from utils.io import read_yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

MODEL_TYPE = "damage_instance_classifier"
BACKBONE_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
CLASSIFIER_TYPE = "corn"
INPUT_MODE_CHOICES = {"rgb", "rgbm"}
AMP_DTYPE_CHOICES = {"auto", "fp16", "bf16", "float16", "bfloat16"}


def default_config() -> dict[str, Any]:
    return {
        "project_name": "calibrated-building-damage-classifier",
        "seed": 42,
        "dataset": {
            "root_dir": "/home/lky/data/xBD",
            "train_list": "/home/lky/data/xBD/xBD_list/train_all.txt",
            "val_list": "/home/lky/data/xBD/xBD_list/val_all.txt",
            "test_list": "/home/lky/data/xBD/xBD_list/test_all.txt",
            "instance_source": "gt_json",
            "allow_tier3": False,
            "use_multi_context": True,
            "image_size": 224,
            "image_size_tight": 224,
            "image_size_context": 224,
            "image_size_neighborhood": 256,
            "tight_padding": 0.10,
            "context_scale": 1.75,
            "neighborhood_crop_size": 512,
            "context_ratio": 0.25,
            "min_polygon_area": 16.0,
            "min_mask_pixels": 16,
            "max_out_of_bound_ratio": 0.4,
            "cache_dir": "./cache",
        },
        "augmentation": {
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rotate90_prob": 0.5,
            "random_resized_crop_prob": 0.5,
            "random_resized_crop_scale": [0.9, 1.0],
            "random_resized_crop_ratio": [0.95, 1.05],
            "min_mask_retention": 0.75,
            "color_jitter_prob": 0.6,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.1,
            "hue": 0.02,
            "context_dropout_prob": 0.2,
            "context_blur_prob": 0.2,
            "context_grayscale_prob": 0.1,
            "context_noise_prob": 0.05,
            "context_mix_prob": 0.0,
            "context_edge_soften_pixels": 4,
            "context_dilate_pixels": 3,
            "context_apply_to_pre_and_post_independently": False,
            "context_preserve_instance_strictly": True,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "model": {
            "model_type": MODEL_TYPE,
            "backbone_name": BACKBONE_NAME,
            "classifier_type": CLASSIFIER_TYPE,
            "pretrained": True,
            "input_mode": "rgbm",
            "feature_dim": 256,
            "alignment_type": "residual_gate",
            "use_multi_context_model": True,
            "token_count": 8,
            "token_mixer_layers": 1,
            "token_mixer_heads": 4,
            "tight_token_count": 8,
            "context_token_count": 8,
            "neighborhood_token_count": 12,
            "local_attention_heads": 4,
            "local_attention_layers": 2,
            "tight_window_size": 7,
            "context_window_size": 7,
            "neighborhood_window_size": 8,
            "use_tight_branch": True,
            "use_context_branch": True,
            "use_neighborhood_scale": True,
            "use_local_attention": True,
            "use_cross_scale_attention": True,
            "cross_scale_heads": 4,
            "cross_scale_layers": 2,
            "cross_scale_dropout": 0.1,
            "cross_scale_layerscale_init": 1.0e-3,
            "cross_scale_residual_max": 0.3,
            "context_dropout_prob": 0.20,
            "neighborhood_dropout_prob": 0.40,
            "safe_multicontext_mode": True,
            "enable_neighborhood_graph": False,
            "graph_k_neighbors": 6,
            "graph_layers": 1,
            "graph_hidden_dim": 256,
            "graph_attention_heads": 4,
            "graph_use_distance_bias": True,
            "neighborhood_branch_gate_init": -4.0,
            "neighborhood_residual_scale_init": 0.02,
            "neighborhood_residual_scale_max": 0.20,
            "neighborhood_aux_enabled": False,
            "dropout": 0.2,
            "use_change_suppression": True,
            "change_block_channels": 256,
            "enable_pseudo_suppression": True,
            "fuse_change_to_tokens": True,
            "change_residual_scale": 0.2,
            "change_gate_init_gamma": 0.1,
            "gate_temperature": 2.0,
            "gate_bias_init": -2.0,
            "neighborhood_gate_bias_init": -4.0,
            "neighborhood_gate_temperature": 3.0,
            "enable_severity_aux": True,
            "freeze_neighborhood_strength_after_epoch": 7,
            "neighborhood_branch_gate_max": 0.04,
        },
        "loss": {
            "name": "corn",
            "probability_eps": 1e-6,
            "loss_damage_aux_weight": 0.05,
            "loss_damage_tight_weight": 0.02,
            "loss_damage_context_weight": 0.01,
            "loss_damage_neighborhood_weight": 0.00,
            "loss_severity_aux_weight": 0.02,
            "loss_unchanged_weight": 0.02,
            "loss_gate_bg_weight": 0.01,
            "loss_gate_contrast_weight": 0.02,
            "tight_gate_bg_weight": 0.002,
            "tight_gate_contrast_weight": 0.005,
            "context_gate_bg_weight": 0.001,
            "context_gate_contrast_weight": 0.002,
            "neighborhood_gate_bg_weight": 0.000,
            "neighborhood_gate_contrast_weight": 0.000,
            "loss_corn_mono_weight": 0.0,
            "gate_contrast_margin": 0.15,
            "tight_gate_contrast_margin": 0.15,
            "context_gate_contrast_margin": 0.08,
            "neighborhood_gate_contrast_margin": 0.03,
            "damage_aux_mode": "instance_weak",
            "no_damage_label_id": 0,
            "graph_consistency_weight": 0.00,
            "enable_ordinal_adjacent_smoothing": True,
            "ordinal_smoothing": 0.05,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-4,
            "backbone_lr": 3e-5,
            "new_module_lr": 1e-4,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "name": "warmup_cosine",
            "warmup_epochs": 5,
            "min_lr_ratio": 0.01,
        },
        "training": {
            "batch_size": 32,
            "epochs": 50,
            "num_workers": 8,
            "amp_enabled": True,
            "amp_dtype": "auto",
            "grad_scaler_init_scale": 4096.0,
            "grad_scaler_growth_interval": 2000,
            "max_grad_norm": 0.5,
            "finite_check_enabled": True,
            "early_stop_metric": "macro_f1",
            "early_stop_patience": 8,
            "save_best_macro_f1": True,
            "save_best_qwk": True,
            "save_best_far_error": True,
            "composite_alpha_qwk": 0.15,
            "composite_beta_far_error": 0.10,
            "ema_enabled": True,
            "ema_decay": 0.999,
        },
        "eval": {
            "split_name": "val",
            "batch_size": 32,
            "num_workers": 8,
        },
        "bridge": {
            "split_name": "val",
            "batch_size": 32,
            "num_workers": 8,
            "target_source": "png",
            "use_official_target_png": True,
            "save_pixel_predictions": True,
            "save_visuals": True,
            "save_official_style_pngs": True,
            "overlap_policy": "max_label",
            "output_subdir": "bridge_pixel_eval",
        },
        "logging": {
            "output_root": "./outputs",
            "exp_name": "default",
            "log_feature_stats": True,
            "log_gate_stats": True,
            "log_severity_aux_stats": True,
            "log_scale_metrics": True,
            "log_cross_scale_attention": True,
            "log_graph_metrics": True,
            "log_context_diagnostic": True,
        },
        "sampler": {
            "enable_class_boost": True,
            "minor_boost": 1.20,
            "major_boost": 1.05,
            "destroyed_boost": 1.00,
            "no_damage_boost": 1.00,
        },
        "ablation": {
            "run_name": "convnextv2_corn_tokenagg_cachlite",
            "keep_pixel_bridge_eval": True,
        },
    }


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _require_keys(config: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if key not in config:
            raise KeyError(f"Missing required config section '{key}'.")


def _normalize_amp_dtype(value: str) -> str:
    normalized = str(value).lower()
    if normalized == "float16":
        normalized = "fp16"
    elif normalized == "bfloat16":
        normalized = "bf16"
    if normalized not in AMP_DTYPE_CHOICES:
        raise ValueError(f"Unsupported training.amp_dtype='{value}'.")
    return normalized


def _validate_dataset_config(config: dict[str, Any]) -> None:
    dataset_cfg = config["dataset"]
    dataset_cfg["root_dir"] = str(dataset_cfg["root_dir"])
    dataset_cfg["train_list"] = str(dataset_cfg["train_list"])
    dataset_cfg["val_list"] = str(dataset_cfg["val_list"])
    dataset_cfg["test_list"] = str(dataset_cfg.get("test_list", ""))
    dataset_cfg["instance_source"] = str(dataset_cfg["instance_source"]).lower()
    dataset_cfg["allow_tier3"] = bool(dataset_cfg["allow_tier3"])
    dataset_cfg["use_multi_context"] = bool(dataset_cfg.get("use_multi_context", True))
    dataset_cfg["image_size"] = int(dataset_cfg.get("image_size", dataset_cfg.get("image_size_tight", 224)))
    dataset_cfg["image_size_tight"] = int(dataset_cfg.get("image_size_tight", dataset_cfg["image_size"]))
    dataset_cfg["image_size_context"] = int(dataset_cfg.get("image_size_context", dataset_cfg["image_size"]))
    dataset_cfg["image_size_neighborhood"] = int(dataset_cfg.get("image_size_neighborhood", dataset_cfg["image_size_context"]))
    dataset_cfg["tight_padding"] = float(dataset_cfg.get("tight_padding", 0.10))
    dataset_cfg["context_scale"] = float(dataset_cfg.get("context_scale", 1.75))
    dataset_cfg["neighborhood_crop_size"] = int(dataset_cfg.get("neighborhood_crop_size", 512))
    dataset_cfg["context_ratio"] = float(dataset_cfg["context_ratio"])
    dataset_cfg["min_polygon_area"] = float(dataset_cfg["min_polygon_area"])
    dataset_cfg["min_mask_pixels"] = int(dataset_cfg["min_mask_pixels"])
    dataset_cfg["max_out_of_bound_ratio"] = float(dataset_cfg["max_out_of_bound_ratio"])
    dataset_cfg["cache_dir"] = str(dataset_cfg["cache_dir"])
    if dataset_cfg["instance_source"] != "gt_json":
        raise ValueError("Only dataset.instance_source='gt_json' is supported.")


def _validate_model_config(config: dict[str, Any]) -> None:
    model_cfg = config["model"]
    model_cfg["model_type"] = MODEL_TYPE
    model_cfg["backbone_name"] = str(model_cfg["backbone_name"])
    model_cfg["classifier_type"] = str(model_cfg["classifier_type"]).lower()
    model_cfg["pretrained"] = bool(model_cfg["pretrained"])
    model_cfg["input_mode"] = str(model_cfg["input_mode"]).lower()
    model_cfg["feature_dim"] = int(model_cfg["feature_dim"])
    model_cfg["alignment_type"] = str(model_cfg["alignment_type"]).lower()
    model_cfg["use_multi_context_model"] = bool(model_cfg.get("use_multi_context_model", True))
    model_cfg["token_count"] = int(model_cfg["token_count"])
    model_cfg["token_mixer_layers"] = int(model_cfg["token_mixer_layers"])
    model_cfg["token_mixer_heads"] = int(model_cfg["token_mixer_heads"])
    model_cfg["tight_token_count"] = int(model_cfg.get("tight_token_count", model_cfg["token_count"]))
    model_cfg["context_token_count"] = int(model_cfg.get("context_token_count", model_cfg["token_count"]))
    model_cfg["neighborhood_token_count"] = int(model_cfg.get("neighborhood_token_count", max(model_cfg["token_count"], 12)))
    model_cfg["local_attention_heads"] = int(model_cfg.get("local_attention_heads", model_cfg["token_mixer_heads"]))
    model_cfg["local_attention_layers"] = int(model_cfg.get("local_attention_layers", 2))
    model_cfg["tight_window_size"] = int(model_cfg.get("tight_window_size", 7))
    model_cfg["context_window_size"] = int(model_cfg.get("context_window_size", 7))
    model_cfg["neighborhood_window_size"] = int(model_cfg.get("neighborhood_window_size", 8))
    model_cfg["use_tight_branch"] = bool(model_cfg.get("use_tight_branch", True))
    model_cfg["use_context_branch"] = bool(model_cfg.get("use_context_branch", True))
    model_cfg["use_neighborhood_scale"] = bool(model_cfg.get("use_neighborhood_scale", True))
    model_cfg["use_local_attention"] = bool(model_cfg.get("use_local_attention", True))
    model_cfg["use_cross_scale_attention"] = bool(model_cfg.get("use_cross_scale_attention", True))
    model_cfg["cross_scale_heads"] = int(model_cfg.get("cross_scale_heads", model_cfg["token_mixer_heads"]))
    model_cfg["cross_scale_layers"] = int(model_cfg.get("cross_scale_layers", 2))
    model_cfg["cross_scale_dropout"] = float(model_cfg.get("cross_scale_dropout", model_cfg.get("dropout", 0.1)))
    model_cfg["cross_scale_layerscale_init"] = float(model_cfg.get("cross_scale_layerscale_init", 1.0e-3))
    model_cfg["cross_scale_residual_max"] = float(model_cfg.get("cross_scale_residual_max", 0.3))
    model_cfg["context_dropout_prob"] = float(model_cfg.get("context_dropout_prob", 0.20))
    model_cfg["neighborhood_dropout_prob"] = float(model_cfg.get("neighborhood_dropout_prob", 0.40))
    model_cfg["safe_multicontext_mode"] = bool(model_cfg.get("safe_multicontext_mode", True))
    model_cfg["enable_neighborhood_graph"] = bool(model_cfg.get("enable_neighborhood_graph", False))
    model_cfg["graph_k_neighbors"] = int(model_cfg.get("graph_k_neighbors", 6))
    model_cfg["graph_layers"] = int(model_cfg.get("graph_layers", 1))
    model_cfg["graph_hidden_dim"] = int(model_cfg.get("graph_hidden_dim", model_cfg["feature_dim"]))
    model_cfg["graph_attention_heads"] = int(model_cfg.get("graph_attention_heads", model_cfg["token_mixer_heads"]))
    model_cfg["graph_use_distance_bias"] = bool(model_cfg.get("graph_use_distance_bias", True))
    model_cfg["neighborhood_branch_gate_init"] = float(model_cfg.get("neighborhood_branch_gate_init", -4.0))
    model_cfg["neighborhood_residual_scale_init"] = float(model_cfg.get("neighborhood_residual_scale_init", 0.02))
    model_cfg["neighborhood_residual_scale_max"] = float(model_cfg.get("neighborhood_residual_scale_max", 0.20))
    model_cfg["neighborhood_aux_enabled"] = bool(model_cfg.get("neighborhood_aux_enabled", False))
    model_cfg["dropout"] = float(model_cfg["dropout"])
    model_cfg["use_change_suppression"] = bool(model_cfg["use_change_suppression"])
    model_cfg["change_block_channels"] = int(model_cfg.get("change_block_channels", model_cfg["feature_dim"]))
    model_cfg["enable_pseudo_suppression"] = bool(model_cfg["enable_pseudo_suppression"])
    model_cfg["fuse_change_to_tokens"] = bool(model_cfg["fuse_change_to_tokens"])
    model_cfg["change_residual_scale"] = float(model_cfg["change_residual_scale"])
    model_cfg["change_gate_init_gamma"] = float(model_cfg["change_gate_init_gamma"])
    model_cfg["gate_temperature"] = float(model_cfg.get("gate_temperature", 2.0))
    model_cfg["gate_bias_init"] = float(model_cfg.get("gate_bias_init", -2.0))
    model_cfg["neighborhood_gate_bias_init"] = float(model_cfg.get("neighborhood_gate_bias_init", -4.0))
    model_cfg["neighborhood_gate_temperature"] = float(model_cfg.get("neighborhood_gate_temperature", 3.0))
    model_cfg["enable_severity_aux"] = bool(model_cfg.get("enable_severity_aux", True))
    model_cfg["freeze_neighborhood_strength_after_epoch"] = int(model_cfg.get("freeze_neighborhood_strength_after_epoch", 7))
    model_cfg["neighborhood_branch_gate_max"] = float(model_cfg.get("neighborhood_branch_gate_max", 0.04))

    if model_cfg["safe_multicontext_mode"]:
        model_cfg["enable_neighborhood_graph"] = False

    if model_cfg["input_mode"] not in INPUT_MODE_CHOICES:
        raise ValueError(f"Unsupported model.input_mode='{model_cfg['input_mode']}'.")
    if model_cfg["classifier_type"] != CLASSIFIER_TYPE:
        raise ValueError("The cleaned mainline only supports model.classifier_type='corn'.")
    if model_cfg["alignment_type"] != "residual_gate":
        raise ValueError("The cleaned mainline only supports model.alignment_type='residual_gate'.")
    if model_cfg["feature_dim"] <= 0:
        raise ValueError("model.feature_dim must be positive.")
    if model_cfg["token_count"] <= 0:
        raise ValueError("model.token_count must be positive.")
    if not model_cfg["use_tight_branch"]:
        raise ValueError("model.use_tight_branch must remain true for the current classifier.")
    for key in [
        "tight_token_count",
        "context_token_count",
        "neighborhood_token_count",
        "local_attention_heads",
        "local_attention_layers",
        "tight_window_size",
        "context_window_size",
        "neighborhood_window_size",
        "cross_scale_heads",
        "cross_scale_layers",
        "graph_k_neighbors",
        "graph_layers",
        "graph_hidden_dim",
        "graph_attention_heads",
    ]:
        if int(model_cfg[key]) <= 0:
            raise ValueError(f"model.{key} must be positive.")
    if model_cfg["token_mixer_layers"] <= 0:
        raise ValueError("model.token_mixer_layers must be positive.")
    if model_cfg["token_mixer_heads"] <= 0:
        raise ValueError("model.token_mixer_heads must be positive.")
    if model_cfg["change_block_channels"] <= 0:
        model_cfg["change_block_channels"] = model_cfg["feature_dim"]
    if model_cfg["change_residual_scale"] < 0.0:
        raise ValueError("model.change_residual_scale must be non-negative.")
    if model_cfg["gate_temperature"] <= 0.0:
        raise ValueError("model.gate_temperature must be positive.")
    if model_cfg["neighborhood_gate_temperature"] <= 0.0:
        raise ValueError("model.neighborhood_gate_temperature must be positive.")
    if model_cfg["cross_scale_layerscale_init"] < 0.0:
        raise ValueError("model.cross_scale_layerscale_init must be non-negative.")
    if model_cfg["cross_scale_residual_max"] <= 0.0:
        raise ValueError("model.cross_scale_residual_max must be positive.")
    if model_cfg["neighborhood_residual_scale_init"] < 0.0 or model_cfg["neighborhood_residual_scale_max"] <= 0.0:
        raise ValueError("Neighborhood residual scale values must be positive.")
    if model_cfg["neighborhood_branch_gate_max"] <= 0.0:
        raise ValueError("model.neighborhood_branch_gate_max must be positive.")


def _validate_loss_config(config: dict[str, Any]) -> None:
    loss_cfg = config["loss"]
    loss_cfg["name"] = str(loss_cfg["name"]).lower()
    loss_cfg["probability_eps"] = float(loss_cfg["probability_eps"])
    loss_cfg["loss_damage_aux_weight"] = float(loss_cfg.get("loss_damage_aux_weight", loss_cfg.get("lambda_damage_aux", 0.0)))
    loss_cfg["loss_damage_tight_weight"] = float(loss_cfg.get("loss_damage_tight_weight", loss_cfg["loss_damage_aux_weight"]))
    loss_cfg["loss_damage_context_weight"] = float(loss_cfg.get("loss_damage_context_weight", 0.0))
    loss_cfg["loss_damage_neighborhood_weight"] = float(loss_cfg.get("loss_damage_neighborhood_weight", 0.0))
    loss_cfg["loss_severity_aux_weight"] = float(loss_cfg.get("loss_severity_aux_weight", 0.0))
    loss_cfg["loss_unchanged_weight"] = float(loss_cfg.get("loss_unchanged_weight", loss_cfg.get("lambda_unchanged", 0.0)))
    loss_cfg["loss_gate_bg_weight"] = float(loss_cfg.get("loss_gate_bg_weight", 0.0))
    loss_cfg["loss_gate_contrast_weight"] = float(loss_cfg.get("loss_gate_contrast_weight", 0.0))
    loss_cfg["tight_gate_bg_weight"] = float(loss_cfg.get("tight_gate_bg_weight", loss_cfg["loss_gate_bg_weight"]))
    loss_cfg["tight_gate_contrast_weight"] = float(loss_cfg.get("tight_gate_contrast_weight", loss_cfg["loss_gate_contrast_weight"]))
    loss_cfg["context_gate_bg_weight"] = float(loss_cfg.get("context_gate_bg_weight", 0.0))
    loss_cfg["context_gate_contrast_weight"] = float(loss_cfg.get("context_gate_contrast_weight", 0.0))
    loss_cfg["neighborhood_gate_bg_weight"] = float(loss_cfg.get("neighborhood_gate_bg_weight", 0.0))
    loss_cfg["neighborhood_gate_contrast_weight"] = float(loss_cfg.get("neighborhood_gate_contrast_weight", 0.0))
    loss_cfg["loss_corn_mono_weight"] = float(loss_cfg.get("loss_corn_mono_weight", loss_cfg.get("lambda_corn_mono", 0.0)))
    loss_cfg["gate_contrast_margin"] = float(loss_cfg.get("gate_contrast_margin", 0.15))
    loss_cfg["tight_gate_contrast_margin"] = float(loss_cfg.get("tight_gate_contrast_margin", loss_cfg["gate_contrast_margin"]))
    loss_cfg["context_gate_contrast_margin"] = float(loss_cfg.get("context_gate_contrast_margin", 0.08))
    loss_cfg["neighborhood_gate_contrast_margin"] = float(loss_cfg.get("neighborhood_gate_contrast_margin", 0.03))
    loss_cfg["damage_aux_mode"] = str(loss_cfg.get("damage_aux_mode", "instance_weak")).lower()
    loss_cfg["no_damage_label_id"] = int(loss_cfg.get("no_damage_label_id", 0))
    loss_cfg["graph_consistency_weight"] = float(loss_cfg.get("graph_consistency_weight", 0.0))
    loss_cfg["enable_ordinal_adjacent_smoothing"] = bool(loss_cfg.get("enable_ordinal_adjacent_smoothing", False))
    loss_cfg["ordinal_smoothing"] = float(loss_cfg.get("ordinal_smoothing", 0.05))
    if loss_cfg["name"] != "corn":
        raise ValueError("The cleaned mainline only supports loss.name='corn'.")
    if not 0.0 < loss_cfg["probability_eps"] < 0.1:
        raise ValueError("loss.probability_eps must be in (0, 0.1).")
    if (
        loss_cfg["loss_damage_aux_weight"] < 0.0
        or loss_cfg["loss_damage_tight_weight"] < 0.0
        or loss_cfg["loss_damage_context_weight"] < 0.0
        or loss_cfg["loss_damage_neighborhood_weight"] < 0.0
        or loss_cfg["loss_severity_aux_weight"] < 0.0
        or loss_cfg["loss_unchanged_weight"] < 0.0
        or loss_cfg["loss_gate_bg_weight"] < 0.0
        or loss_cfg["loss_gate_contrast_weight"] < 0.0
        or loss_cfg["tight_gate_bg_weight"] < 0.0
        or loss_cfg["tight_gate_contrast_weight"] < 0.0
        or loss_cfg["context_gate_bg_weight"] < 0.0
        or loss_cfg["context_gate_contrast_weight"] < 0.0
        or loss_cfg["neighborhood_gate_bg_weight"] < 0.0
        or loss_cfg["neighborhood_gate_contrast_weight"] < 0.0
        or loss_cfg["loss_corn_mono_weight"] < 0.0
        or loss_cfg["graph_consistency_weight"] < 0.0
    ):
        raise ValueError("Auxiliary loss weights must be non-negative.")
    if loss_cfg["damage_aux_mode"] not in {"instance_weak", "pixel_if_available"}:
        raise ValueError("loss.damage_aux_mode must be 'instance_weak' or 'pixel_if_available'.")
    if loss_cfg["no_damage_label_id"] < 0:
        raise ValueError("loss.no_damage_label_id must be non-negative.")
    if loss_cfg["gate_contrast_margin"] < 0.0:
        raise ValueError("loss.gate_contrast_margin must be non-negative.")
    for key in [
        "tight_gate_contrast_margin",
        "context_gate_contrast_margin",
        "neighborhood_gate_contrast_margin",
    ]:
        if loss_cfg[key] < 0.0:
            raise ValueError(f"loss.{key} must be non-negative.")
    if not 0.0 <= loss_cfg["ordinal_smoothing"] < 0.5:
        raise ValueError("loss.ordinal_smoothing must be in [0, 0.5).")


def _validate_ablation_config(config: dict[str, Any]) -> None:
    ablation_cfg = config["ablation"]
    ablation_cfg["run_name"] = str(ablation_cfg.get("run_name", "")).strip() or "default_ablation"
    ablation_cfg["keep_pixel_bridge_eval"] = bool(ablation_cfg.get("keep_pixel_bridge_eval", True))


def _validate_common_runtime_config(config: dict[str, Any]) -> None:
    optimizer_cfg = config["optimizer"]
    optimizer_cfg["name"] = str(optimizer_cfg["name"]).lower()
    optimizer_cfg["lr"] = float(optimizer_cfg["lr"])
    optimizer_cfg["backbone_lr"] = float(optimizer_cfg.get("backbone_lr", optimizer_cfg["lr"]))
    optimizer_cfg["new_module_lr"] = float(optimizer_cfg.get("new_module_lr", optimizer_cfg["lr"]))
    optimizer_cfg["weight_decay"] = float(optimizer_cfg["weight_decay"])
    if optimizer_cfg["name"] != "adamw":
        raise ValueError("The cleaned mainline only supports optimizer.name='adamw'.")
    if optimizer_cfg["lr"] <= 0.0 or optimizer_cfg["backbone_lr"] <= 0.0 or optimizer_cfg["new_module_lr"] <= 0.0:
        raise ValueError("Optimizer learning rates must be positive.")

    scheduler_cfg = config["scheduler"]
    scheduler_cfg["name"] = str(scheduler_cfg["name"]).lower()
    scheduler_cfg["warmup_epochs"] = int(scheduler_cfg["warmup_epochs"])
    scheduler_cfg["min_lr_ratio"] = float(scheduler_cfg["min_lr_ratio"])
    if scheduler_cfg["name"] != "warmup_cosine":
        raise ValueError("The cleaned mainline only supports scheduler.name='warmup_cosine'.")

    training_cfg = config["training"]
    training_cfg["batch_size"] = int(training_cfg["batch_size"])
    training_cfg["epochs"] = int(training_cfg["epochs"])
    training_cfg["num_workers"] = int(training_cfg["num_workers"])
    training_cfg["amp_enabled"] = bool(training_cfg["amp_enabled"])
    training_cfg["amp_dtype"] = _normalize_amp_dtype(training_cfg["amp_dtype"])
    training_cfg["grad_scaler_init_scale"] = float(training_cfg["grad_scaler_init_scale"])
    training_cfg["grad_scaler_growth_interval"] = int(training_cfg["grad_scaler_growth_interval"])
    training_cfg["max_grad_norm"] = float(training_cfg["max_grad_norm"])
    training_cfg["finite_check_enabled"] = bool(training_cfg["finite_check_enabled"])
    training_cfg["early_stop_metric"] = str(training_cfg["early_stop_metric"]).lower()
    training_cfg["early_stop_patience"] = int(training_cfg["early_stop_patience"])
    training_cfg["save_best_macro_f1"] = bool(training_cfg["save_best_macro_f1"])
    training_cfg["save_best_qwk"] = bool(training_cfg["save_best_qwk"])
    training_cfg["save_best_far_error"] = bool(training_cfg["save_best_far_error"])
    training_cfg["composite_alpha_qwk"] = float(training_cfg["composite_alpha_qwk"])
    training_cfg["composite_beta_far_error"] = float(training_cfg["composite_beta_far_error"])
    training_cfg["ema_enabled"] = bool(training_cfg.get("ema_enabled", False))
    training_cfg["ema_decay"] = float(training_cfg.get("ema_decay", 0.999))
    if training_cfg["early_stop_metric"] not in {"macro_f1", "composite_score"}:
        raise ValueError("training.early_stop_metric must be 'macro_f1' or 'composite_score'.")
    if training_cfg["ema_enabled"] and not 0.0 < training_cfg["ema_decay"] < 1.0:
        raise ValueError("training.ema_decay must be in (0, 1).")

    eval_cfg = config["eval"]
    eval_cfg["split_name"] = str(eval_cfg["split_name"])
    eval_cfg["batch_size"] = int(eval_cfg["batch_size"])
    eval_cfg["num_workers"] = int(eval_cfg["num_workers"])

    bridge_cfg = config["bridge"]
    bridge_cfg["split_name"] = str(bridge_cfg["split_name"])
    bridge_cfg["batch_size"] = int(bridge_cfg["batch_size"])
    bridge_cfg["num_workers"] = int(bridge_cfg["num_workers"])
    bridge_cfg["target_source"] = str(bridge_cfg["target_source"]).lower()
    bridge_cfg["use_official_target_png"] = bool(bridge_cfg["use_official_target_png"])
    bridge_cfg["save_pixel_predictions"] = bool(bridge_cfg["save_pixel_predictions"])
    bridge_cfg["save_visuals"] = bool(bridge_cfg["save_visuals"])
    bridge_cfg["save_official_style_pngs"] = bool(bridge_cfg["save_official_style_pngs"])
    bridge_cfg["overlap_policy"] = str(bridge_cfg["overlap_policy"]).lower()
    bridge_cfg["output_subdir"] = str(bridge_cfg["output_subdir"])
    if bridge_cfg["target_source"] not in {"png", "polygon_rasterize"}:
        raise ValueError("bridge.target_source must be 'png' or 'polygon_rasterize'.")
    if bridge_cfg["overlap_policy"] not in {"max_label", "last_wins"}:
        raise ValueError("bridge.overlap_policy must be 'max_label' or 'last_wins'.")

    logging_cfg = config["logging"]
    logging_cfg["output_root"] = str(logging_cfg["output_root"])
    logging_cfg["exp_name"] = str(logging_cfg["exp_name"]).strip() or "default"
    logging_cfg["log_feature_stats"] = bool(logging_cfg.get("log_feature_stats", True))
    logging_cfg["log_gate_stats"] = bool(logging_cfg.get("log_gate_stats", True))
    logging_cfg["log_severity_aux_stats"] = bool(logging_cfg.get("log_severity_aux_stats", True))
    logging_cfg["log_scale_metrics"] = bool(logging_cfg.get("log_scale_metrics", True))
    logging_cfg["log_cross_scale_attention"] = bool(logging_cfg.get("log_cross_scale_attention", True))
    logging_cfg["log_graph_metrics"] = bool(logging_cfg.get("log_graph_metrics", True))
    logging_cfg["log_context_diagnostic"] = bool(logging_cfg.get("log_context_diagnostic", True))

    sampler_cfg = config.setdefault("sampler", {})
    sampler_cfg["enable_class_boost"] = bool(sampler_cfg.get("enable_class_boost", False))
    sampler_cfg["minor_boost"] = float(sampler_cfg.get("minor_boost", 1.0))
    sampler_cfg["major_boost"] = float(sampler_cfg.get("major_boost", 1.0))
    sampler_cfg["destroyed_boost"] = float(sampler_cfg.get("destroyed_boost", 1.0))
    sampler_cfg["no_damage_boost"] = float(sampler_cfg.get("no_damage_boost", 1.0))


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _require_keys(
        config,
        "dataset",
        "augmentation",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "eval",
        "bridge",
        "logging",
        "ablation",
    )
    _validate_dataset_config(config)
    _validate_model_config(config)
    _validate_loss_config(config)
    _validate_common_runtime_config(config)
    _validate_ablation_config(config)
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = default_config()
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if config_path.exists():
        raw = read_yaml(config_path)
        if raw:
            _deep_update(config, raw)
    return validate_config(config)


def apply_overrides(config: dict[str, Any], args: Any, *, is_eval: bool = False) -> dict[str, Any]:
    config = copy.deepcopy(config)

    if getattr(args, "seed", None) is not None:
        config["seed"] = int(args.seed)

    dataset_cfg = config["dataset"]
    for key in ["root_dir", "train_list", "val_list", "test_list", "instance_source", "cache_dir"]:
        value = getattr(args, key, None)
        if value is not None:
            dataset_cfg[key] = value
    if getattr(args, "allow_tier3", False):
        dataset_cfg["allow_tier3"] = True
    if getattr(args, "image_size", None) is not None:
        dataset_cfg["image_size"] = int(args.image_size)
        dataset_cfg["image_size_tight"] = int(args.image_size)
        dataset_cfg["image_size_context"] = int(args.image_size)
        dataset_cfg["image_size_neighborhood"] = int(args.image_size)

    model_cfg = config["model"]
    if getattr(args, "input_mode", None) is not None:
        model_cfg["input_mode"] = str(args.input_mode)
    if getattr(args, "no_pretrained", False):
        model_cfg["pretrained"] = False
    if getattr(args, "feature_dim", None) is not None:
        model_cfg["feature_dim"] = int(args.feature_dim)
    if getattr(args, "token_count", None) is not None:
        model_cfg["token_count"] = int(args.token_count)
        model_cfg["tight_token_count"] = int(args.token_count)
        model_cfg["context_token_count"] = int(args.token_count)
        model_cfg["neighborhood_token_count"] = int(args.token_count)

    training_cfg = config["training"]
    eval_cfg = config["eval"]
    bridge_cfg = config["bridge"]
    optimizer_cfg = config["optimizer"]
    logging_cfg = config["logging"]

    if getattr(args, "batch_size", None) is not None:
        if is_eval:
            eval_cfg["batch_size"] = int(args.batch_size)
            bridge_cfg["batch_size"] = int(args.batch_size)
        else:
            training_cfg["batch_size"] = int(args.batch_size)
            eval_cfg["batch_size"] = int(args.batch_size)
            bridge_cfg["batch_size"] = int(args.batch_size)
    if getattr(args, "epochs", None) is not None:
        training_cfg["epochs"] = int(args.epochs)
    if getattr(args, "lr", None) is not None:
        optimizer_cfg["lr"] = float(args.lr)
        optimizer_cfg["backbone_lr"] = float(args.lr)
        optimizer_cfg["new_module_lr"] = float(args.lr)
    if getattr(args, "amp", None) is not None:
        training_cfg["amp_enabled"] = bool(args.amp)
    if getattr(args, "amp_dtype", None) is not None:
        training_cfg["amp_dtype"] = str(args.amp_dtype)
    if getattr(args, "num_workers", None) is not None:
        if is_eval:
            eval_cfg["num_workers"] = int(args.num_workers)
            bridge_cfg["num_workers"] = int(args.num_workers)
        else:
            training_cfg["num_workers"] = int(args.num_workers)
            eval_cfg["num_workers"] = int(args.num_workers)
            bridge_cfg["num_workers"] = int(args.num_workers)
    if getattr(args, "output_root", None) is not None:
        logging_cfg["output_root"] = str(args.output_root)
    if getattr(args, "exp_name", None) is not None:
        logging_cfg["exp_name"] = str(args.exp_name)
    if getattr(args, "split_name", None) is not None:
        eval_cfg["split_name"] = str(args.split_name)
        bridge_cfg["split_name"] = str(args.split_name)

    return validate_config(config)


def get_run_dir(config: dict[str, Any]) -> Path:
    logging_cfg = config["logging"]
    return Path(logging_cfg["output_root"]) / logging_cfg["exp_name"]


def get_split_list_path(config: dict[str, Any], split_name: str) -> str:
    split_name = str(split_name)
    dataset_cfg = config["dataset"]
    if split_name == "train":
        return str(dataset_cfg["train_list"])
    if split_name == "val":
        return str(dataset_cfg["val_list"])
    if split_name == "test":
        test_list = str(dataset_cfg.get("test_list", "")).strip()
        if not test_list:
            raise ValueError("dataset.test_list is not configured.")
        return test_list
    raise ValueError(f"Unsupported split_name='{split_name}'.")
