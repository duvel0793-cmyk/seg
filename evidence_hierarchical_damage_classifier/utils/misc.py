from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}
SEVERITY_TARGETS = torch.tensor([0.0, 0.33, 0.66, 1.0], dtype=torch.float32)
SCALE_NAMES = ("tight", "context", "neighborhood")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def default_config() -> dict[str, Any]:
    return {
        "project": {
            "name": "evidence_hierarchical_damage_classifier",
            "output_dir": "outputs/evidence_hier_corn",
        },
        "data": {
            "root": "/home/ubuntu/code/lky/data/xBD",
            "train_list": "/home/ubuntu/code/lky/data/xBD/xBD_list/train_all.txt",
            "val_list": "/home/ubuntu/code/lky/data/xBD/xBD_list/val_all.txt",
            "test_list": "/home/ubuntu/code/lky/data/xBD/xBD_list/test_all.txt",
            "use_pixel_targets": True,
            "ignore_index": 255,
            "num_workers": None,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        "dataset": {
            "crop_scales": {
                "tight": {"enabled": True, "output_size": 224, "context_factor": 1.05},
                "context": {"enabled": True, "output_size": 224, "context_factor": 2.0},
                "neighborhood": {"enabled": True, "output_size": 256, "context_factor": 4.0},
            },
            "input_mode": "rgbm",
            "num_classes": 4,
            "min_polygon_area": 16.0,
            "min_mask_pixels": 16,
            "max_out_of_bound_ratio": 0.4,
            "allow_tier3": False,
            "debug_subset": 0,
            "debug_max_samples": 0,
        },
        "augmentation": {
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rotate90_prob": 0.5,
            "random_resized_crop_prob": 0.5,
            "random_resized_crop_scale": [0.9, 1.0],
            "random_resized_crop_ratio": [0.95, 1.05],
            "color_jitter_prob": 0.5,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.10,
            "hue": 0.02,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "model": {
            "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "pretrained": True,
            "input_mode": "rgbm",
            "use_tight_branch": True,
            "use_context_branch": True,
            "use_neighborhood_branch": True,
            "enable_alignment": True,
            "enable_prepost_fusion": True,
            "fusion_mode": "diff_prod_concat",
            "enable_damage_aware_block": True,
            "enable_change_gate": True,
            "use_cross_scale_attention": True,
            "cross_scale_fusion_mode": "tight_query_all_kv",
            "use_evidence_head": True,
            "use_hierarchical_head": True,
            "use_global_corn_aux": True,
            "enable_minor_boundary_aux": False,
            "head_type": "two_stage_hierarchical",
            "feature_dim": 768,
            "token_dim": 256,
            "evidence_dim": 128,
            "dropout": 0.1,
            "evidence_fusion_mode": "concat_mlp_original",
            "evidence_gate_init_logit": -4.0,
            "enable_feature_calibration": True,
            "evidence_scales": ["tight", "context", "neighborhood"],
            "tight_token_count": 8,
            "context_token_count": 8,
            "neighborhood_token_count": 12,
            "local_attention_heads": 4,
            "local_attention_layers": 2,
            "local_attention_layers_tight": None,
            "local_attention_layers_context": None,
            "local_attention_layers_neighborhood": None,
            "tight_window_size": 7,
            "context_window_size": 7,
            "neighborhood_window_size": 8,
            "cross_scale_heads": 4,
            "cross_scale_layers": 2,
            "cross_scale_dropout": 0.1,
            "context_dropout_prob": 0.1,
            "neighborhood_dropout_prob": 0.2,
            "evidence_topk_ratio": 0.10,
            "evidence_threshold": 0.5,
            "damage_decision_threshold": 0.5,
        },
        "loss": {
            "corn_weight": 0.0,
            "binary_damage_weight": 1.0,
            "severity_corn_weight": 1.0,
            "final_ce_weight": 0.5,
            "final_class_weight_mode": "auto",
            "dataset_class_counts": [115085, 14712, 13845, 13029],
            "global_corn_aux_weight": 0.2,
            "evidence_pixel_weight": 0.1,
            "evidence_mil_weight": 0.1,
            "severity_map_weight": 0.05,
            "evidence_pixel_class_mode": "dense_all",
            "evidence_pixel_every_n_steps": 1,
            "minor_no_aux_weight": 0.0,
            "minor_major_aux_weight": 0.0,
            "evidence_schedule": {
                "enabled": False,
                "warmup_epochs": 0,
                "ramp_epochs": 0,
            },
            "damage_aux_weight": 0.1,
            "unchanged_weight": 0.02,
            "gate_bg_weight": 0.01,
            "gate_contrast_weight": 0.01,
            "ignore_index": 255,
            "gate_contrast_margin": 0.15,
            "balanced_binary_loss": True,
            "balanced_final_ce": True,
        },
        "training": {
            "epochs": 20,
            "batch_size": 16,
            "num_workers": 8,
            "optimizer": "adamw",
            "lr": 2.0e-4,
            "backbone_lr": None,
            "new_module_lr": None,
            "weight_decay": 0.05,
            "warmup_epochs": 2,
            "scheduler": "warmup_cosine",
            "amp": True,
            "amp_dtype": "bf16",
            "grad_clip_norm": 1.0,
            "ema_enabled": True,
            "ema_decay": 0.999,
            "channels_last": False,
            "compile_model": False,
            "early_stop_metric": "ema_macro_f1",
            "early_stop_patience": 6,
            "seed": 42,
        },
        "eval": {
            "batch_size": 8,
            "num_workers": 8,
            "validate_ema": True,
            "validate_raw": True,
            "validate_ema_every": 1,
            "validate_raw_every": 1,
            "save_best_by": "ema_macro_f1",
            "split": "val",
            "damage_threshold_sweep": False,
            "damage_threshold_values": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            "apply_best_damage_threshold": "none",
            "save_diagnostics": False,
            "save_visuals": False,
        },
        "logging": {
            "log_interval": 50,
            "save_history_jsonl": True,
            "save_evidence_preview": False,
            "log_heavy_diagnostics": False,
        },
        "bridge": {
            "target_source": "png",
            "overlap_policy": "max_label",
            "save_pixel_predictions": True,
            "save_visuals": True,
        },
    }


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def _apply_modern_schema_overrides(config: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    data_raw = raw.get("data", {}) if isinstance(raw.get("data"), dict) else {}
    dataset_raw = raw.get("dataset", {}) if isinstance(raw.get("dataset"), dict) else {}
    model_raw = raw.get("model", {}) if isinstance(raw.get("model"), dict) else {}
    loss_raw = raw.get("loss", {}) if isinstance(raw.get("loss"), dict) else {}
    optimizer_raw = raw.get("optimizer", {}) if isinstance(raw.get("optimizer"), dict) else {}
    scheduler_raw = raw.get("scheduler", {}) if isinstance(raw.get("scheduler"), dict) else {}
    training_raw = raw.get("training", {}) if isinstance(raw.get("training"), dict) else {}
    eval_raw = raw.get("eval", {}) if isinstance(raw.get("eval"), dict) else {}
    logging_raw = raw.get("logging", {}) if isinstance(raw.get("logging"), dict) else {}
    bridge_raw = raw.get("bridge", {}) if isinstance(raw.get("bridge"), dict) else {}
    ablation_raw = raw.get("ablation", {}) if isinstance(raw.get("ablation"), dict) else {}

    if "project_name" in raw:
        _set_nested(config, ("project", "name"), raw["project_name"])
    if "output_dir" in raw:
        _set_nested(config, ("project", "output_dir"), raw["output_dir"])
    elif "output_root" in logging_raw:
        exp_name = (
            logging_raw.get("exp_name")
            or ablation_raw.get("run_name")
            or raw.get("project_name")
            or config["project"].get("name")
        )
        _set_nested(config, ("project", "output_dir"), str(Path(logging_raw["output_root"]) / str(exp_name)))

    dataset_to_data = {
        "root_dir": ("data", "root"),
        "train_list": ("data", "train_list"),
        "val_list": ("data", "val_list"),
        "test_list": ("data", "test_list"),
    }
    for source_key, target_path in dataset_to_data.items():
        if source_key in dataset_raw:
            _set_nested(config, target_path, dataset_raw[source_key])

    if "seed" in raw:
        _set_nested(config, ("training", "seed"), raw["seed"])
    if "seed" in training_raw:
        _set_nested(config, ("training", "seed"), training_raw["seed"])
    if "batch_size" in training_raw:
        _set_nested(config, ("training", "batch_size"), training_raw["batch_size"])
    if "epochs" in training_raw:
        _set_nested(config, ("training", "epochs"), training_raw["epochs"])
    if "num_workers" in training_raw:
        _set_nested(config, ("training", "num_workers"), training_raw["num_workers"])
        _set_nested(config, ("eval", "num_workers"), training_raw["num_workers"])
    elif "num_workers" in data_raw:
        _set_nested(config, ("training", "num_workers"), data_raw["num_workers"])
        _set_nested(config, ("eval", "num_workers"), data_raw["num_workers"])
    if "amp_enabled" in training_raw:
        _set_nested(config, ("training", "amp"), training_raw["amp_enabled"])
    if "amp_dtype" in training_raw:
        _set_nested(config, ("training", "amp_dtype"), training_raw["amp_dtype"])
    if "ema_enabled" in training_raw:
        _set_nested(config, ("training", "ema_enabled"), training_raw["ema_enabled"])
    if "ema" in training_raw:
        _set_nested(config, ("training", "ema_enabled"), training_raw["ema"])
    if "ema_decay" in training_raw:
        _set_nested(config, ("training", "ema_decay"), training_raw["ema_decay"])
    if "early_stop_metric" in training_raw:
        metric_name = str(training_raw["early_stop_metric"])
        if metric_name == "macro_f1":
            metric_name = "ema_macro_f1"
        _set_nested(config, ("training", "early_stop_metric"), metric_name)
    if "early_stop_patience" in training_raw:
        _set_nested(config, ("training", "early_stop_patience"), training_raw["early_stop_patience"])
    if "max_grad_norm" in training_raw:
        _set_nested(config, ("training", "grad_clip_norm"), training_raw["max_grad_norm"])

    if "lr" in optimizer_raw:
        _set_nested(config, ("training", "lr"), optimizer_raw["lr"])
    if "weight_decay" in optimizer_raw:
        _set_nested(config, ("training", "weight_decay"), optimizer_raw["weight_decay"])
    if "backbone_lr" in optimizer_raw:
        _set_nested(config, ("training", "backbone_lr"), optimizer_raw["backbone_lr"])
    if "new_module_lr" in optimizer_raw:
        _set_nested(config, ("training", "new_module_lr"), optimizer_raw["new_module_lr"])

    if "warmup_epochs" in scheduler_raw:
        _set_nested(config, ("training", "warmup_epochs"), scheduler_raw["warmup_epochs"])
    if "min_lr_ratio" in scheduler_raw:
        _set_nested(config, ("training", "min_lr_ratio"), scheduler_raw["min_lr_ratio"])

    if "batch_size" in eval_raw:
        _set_nested(config, ("eval", "batch_size"), eval_raw["batch_size"])
    if "num_workers" in eval_raw:
        _set_nested(config, ("eval", "num_workers"), eval_raw["num_workers"])

    if "allow_tier3" in dataset_raw:
        _set_nested(config, ("dataset", "allow_tier3"), dataset_raw["allow_tier3"])
    if "min_polygon_area" in dataset_raw:
        _set_nested(config, ("dataset", "min_polygon_area"), dataset_raw["min_polygon_area"])
    if "min_mask_pixels" in dataset_raw:
        _set_nested(config, ("dataset", "min_mask_pixels"), dataset_raw["min_mask_pixels"])
    if "max_out_of_bound_ratio" in dataset_raw:
        _set_nested(config, ("dataset", "max_out_of_bound_ratio"), dataset_raw["max_out_of_bound_ratio"])

    tight_size = dataset_raw.get("image_size_tight", dataset_raw.get("image_size"))
    context_size = dataset_raw.get("image_size_context", dataset_raw.get("image_size"))
    neighborhood_size = dataset_raw.get("image_size_neighborhood")
    if tight_size is not None:
        _set_nested(config, ("dataset", "crop_scales", "tight", "output_size"), tight_size)
    if context_size is not None:
        _set_nested(config, ("dataset", "crop_scales", "context", "output_size"), context_size)
    if neighborhood_size is not None:
        _set_nested(config, ("dataset", "crop_scales", "neighborhood", "output_size"), neighborhood_size)
    if "tight_padding" in dataset_raw:
        tight_factor = 1.0 + (2.0 * float(dataset_raw["tight_padding"]))
        _set_nested(config, ("dataset", "crop_scales", "tight", "context_factor"), tight_factor)
    if "context_scale" in dataset_raw:
        _set_nested(config, ("dataset", "crop_scales", "context", "context_factor"), dataset_raw["context_scale"])
    if "crop_scales" in dataset_raw and isinstance(dataset_raw["crop_scales"], dict):
        for scale_name in SCALE_NAMES:
            scale_entry = dataset_raw["crop_scales"].get(scale_name)
            if not isinstance(scale_entry, dict):
                continue
            if "enabled" in scale_entry:
                _set_nested(config, ("dataset", "crop_scales", scale_name, "enabled"), scale_entry["enabled"])

    model_aliases = {
        "backbone_name": ("model", "backbone"),
        "pretrained": ("model", "pretrained"),
        "input_mode": ("model", "input_mode"),
        "dropout": ("model", "dropout"),
        "tight_token_count": ("model", "tight_token_count"),
        "context_token_count": ("model", "context_token_count"),
        "neighborhood_token_count": ("model", "neighborhood_token_count"),
        "local_attention_heads": ("model", "local_attention_heads"),
        "local_attention_layers": ("model", "local_attention_layers"),
        "tight_window_size": ("model", "tight_window_size"),
        "context_window_size": ("model", "context_window_size"),
        "neighborhood_window_size": ("model", "neighborhood_window_size"),
        "cross_scale_heads": ("model", "cross_scale_heads"),
        "cross_scale_layers": ("model", "cross_scale_layers"),
        "cross_scale_dropout": ("model", "cross_scale_dropout"),
        "context_dropout_prob": ("model", "context_dropout_prob"),
        "neighborhood_dropout_prob": ("model", "neighborhood_dropout_prob"),
    }
    for source_key, target_path in model_aliases.items():
        if source_key in model_raw:
            _set_nested(config, target_path, model_raw[source_key])
    if "feature_dim" in model_raw:
        _set_nested(config, ("model", "feature_dim"), model_raw["feature_dim"])
        if "token_dim" not in model_raw:
            _set_nested(config, ("model", "token_dim"), model_raw["feature_dim"])
    if "token_dim" in model_raw:
        _set_nested(config, ("model", "token_dim"), model_raw["token_dim"])
    if "evidence_dim" in model_raw:
        _set_nested(config, ("model", "evidence_dim"), model_raw["evidence_dim"])
    if "use_tight_branch" in model_raw:
        _set_nested(config, ("model", "use_tight_branch"), model_raw["use_tight_branch"])
    if "use_context_branch" in model_raw:
        _set_nested(config, ("model", "use_context_branch"), model_raw["use_context_branch"])
    if "use_neighborhood_scale" in model_raw:
        _set_nested(config, ("model", "use_neighborhood_branch"), model_raw["use_neighborhood_scale"])
    if "use_neighborhood_branch" in model_raw:
        _set_nested(config, ("model", "use_neighborhood_branch"), model_raw["use_neighborhood_branch"])
    if "use_cross_scale_attention" in model_raw:
        _set_nested(config, ("model", "use_cross_scale_attention"), model_raw["use_cross_scale_attention"])
    for field_name in (
        "enable_alignment",
        "enable_prepost_fusion",
        "fusion_mode",
        "enable_damage_aware_block",
        "enable_change_gate",
        "head_type",
        "cross_scale_fusion_mode",
        "enable_feature_calibration",
        "enable_minor_boundary_aux",
        "use_evidence_head",
        "use_hierarchical_head",
        "use_global_corn_aux",
        "local_attention_layers_tight",
        "local_attention_layers_context",
        "local_attention_layers_neighborhood",
    ):
        if field_name in model_raw:
            _set_nested(config, ("model", field_name), model_raw[field_name])
    if "classifier_type" in model_raw:
        classifier_type = str(model_raw["classifier_type"]).lower()
        _set_nested(config, ("model", "use_hierarchical_head"), classifier_type == "corn")
        _set_nested(config, ("model", "use_global_corn_aux"), True)
    if "use_hierarchical_head" in model_raw and "head_type" not in model_raw:
        inferred_head_type = "two_stage_hierarchical" if bool(model_raw["use_hierarchical_head"]) else "flat_corn"
        _set_nested(config, ("model", "head_type"), inferred_head_type)

    loss_aliases = {
        "loss_damage_aux_weight": ("loss", "damage_aux_weight"),
        "loss_gate_bg_weight": ("loss", "gate_bg_weight"),
        "loss_gate_contrast_weight": ("loss", "gate_contrast_weight"),
    }
    for source_key, target_path in loss_aliases.items():
        if source_key in loss_raw:
            _set_nested(config, target_path, loss_raw[source_key])
    if "ignore_index" in loss_raw:
        _set_nested(config, ("loss", "ignore_index"), loss_raw["ignore_index"])
        _set_nested(config, ("data", "ignore_index"), loss_raw["ignore_index"])
    for field_name in (
        "corn_weight",
        "binary_damage_weight",
        "severity_corn_weight",
        "final_ce_weight",
        "final_class_weight_mode",
        "global_corn_aux_weight",
        "evidence_pixel_weight",
        "evidence_mil_weight",
        "severity_map_weight",
        "damage_aux_weight",
        "unchanged_weight",
        "gate_bg_weight",
        "gate_contrast_weight",
        "minor_no_aux_weight",
        "minor_major_aux_weight",
    ):
        if field_name in loss_raw:
            _set_nested(config, ("loss", field_name), loss_raw[field_name])

    if "save_pixel_predictions" in bridge_raw:
        _set_nested(config, ("bridge", "save_pixel_predictions"), bridge_raw["save_pixel_predictions"])
    if "save_visuals" in bridge_raw:
        _set_nested(config, ("bridge", "save_visuals"), bridge_raw["save_visuals"])
    if "overlap_policy" in bridge_raw:
        _set_nested(config, ("bridge", "overlap_policy"), bridge_raw["overlap_policy"])

    return config


def _remap_path_if_missing(path_value: str | Path, *, source_root: str | Path | None, target_root: str | Path | None) -> str:
    path = Path(path_value)
    if path.exists() or source_root is None or target_root is None:
        return str(path)
    try:
        relative = path.relative_to(Path(source_root))
    except Exception:
        return str(path)
    candidate = Path(target_root) / relative
    return str(candidate if candidate.exists() else path)


def merge_config(raw: dict[str, Any]) -> dict[str, Any]:
    default_cfg = default_config()
    cfg = default_config()
    deep_update(cfg, raw)
    _apply_modern_schema_overrides(cfg, raw)
    default_root = default_cfg["data"]["root"]
    current_root = cfg["data"]["root"]
    for split_key in ("train_list", "val_list", "test_list"):
        cfg["data"][split_key] = _remap_path_if_missing(
            cfg["data"][split_key],
            source_root=current_root,
            target_root=default_root,
        )
    if not Path(cfg["data"]["root"]).exists() and Path(default_root).exists():
        cfg["data"]["root"] = str(default_root)
    for scale_name in SCALE_NAMES:
        cfg["dataset"]["crop_scales"].setdefault(scale_name, {})
        cfg["dataset"]["crop_scales"][scale_name].setdefault("enabled", True)
    if "head_type" not in cfg["model"] or not cfg["model"]["head_type"]:
        cfg["model"]["head_type"] = "two_stage_hierarchical" if bool(cfg["model"].get("use_hierarchical_head", True)) else "flat_corn"
    if not bool(cfg["dataset"]["crop_scales"]["context"].get("enabled", True)):
        cfg["model"]["use_context_branch"] = False
    if not bool(cfg["dataset"]["crop_scales"]["neighborhood"].get("enabled", True)):
        cfg["model"]["use_neighborhood_branch"] = False
    cfg["project"]["output_dir"] = str(cfg["project"]["output_dir"])
    cfg["data"]["root"] = str(cfg["data"]["root"])
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    raw = read_yaml(path)
    return merge_config(raw)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_dtype(config: dict[str, Any], device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or not bool(config["training"].get("amp", True)):
        return None
    requested = str(config["training"].get("amp_dtype", "bf16")).lower()
    if requested == "bf16" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        return torch.bfloat16
    if requested in {"bf16", "fp16"}:
        return torch.float16 if requested == "fp16" or not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.bfloat16
    return torch.float16


def get_output_dir(config: dict[str, Any]) -> Path:
    return ensure_dir(Path(config["project"]["output_dir"]))


def get_enabled_scale_names(config: dict[str, Any]) -> list[str]:
    crop_scales = config.get("dataset", {}).get("crop_scales", {})
    return [scale_name for scale_name in SCALE_NAMES if bool(crop_scales.get(scale_name, {}).get("enabled", True))]


def get_split_list(config: dict[str, Any], split: str) -> str:
    split = str(split)
    if split == "train":
        return str(config["data"]["train_list"])
    if split == "val":
        return str(config["data"]["val_list"])
    if split == "test":
        return str(config["data"]["test_list"])
    raise ValueError(f"Unsupported split '{split}'.")


def tensor_to_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def summarize_tensor(tensor: torch.Tensor) -> dict[str, float | int | None]:
    tensor = tensor.detach().float()
    finite = torch.isfinite(tensor)
    result: dict[str, float | int | None] = {
        "numel": int(tensor.numel()),
        "num_finite": int(finite.sum().item()),
    }
    if finite.any():
        values = tensor[finite]
        result["mean"] = float(values.mean().item())
        result["std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
        result["min"] = float(values.min().item())
        result["max"] = float(values.max().item())
    else:
        result["mean"] = None
        result["std"] = None
        result["min"] = None
        result["max"] = None
    return result


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    text = str(wkt).strip()
    upper = text.upper()
    if upper.startswith("POLYGON"):
        start = text.find("((")
        end = text.rfind("))")
        ring_text = text[start + 2 : end]
    elif upper.startswith("MULTIPOLYGON"):
        start = text.find("(((")
        end = text.rfind(")))")
        ring_text = text[start + 3 : end].split("),")[0]
    else:
        raise ValueError(f"Unsupported WKT: {wkt[:32]}")
    ring_text = ring_text.replace("(", "").replace(")", "")
    points: list[tuple[float, float]] = []
    for pair in ring_text.split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("Invalid polygon.")
    return points


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    x = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bbox(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def is_valid_polygon(points: Iterable[tuple[float, float]], min_area: float = 1.0) -> bool:
    pts = list(points)
    if len(pts) < 3:
        return False
    return polygon_area(pts) >= float(min_area)


def clip_bbox_to_image(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1_i = max(0, int(math.floor(x1)))
    y1_i = max(0, int(math.floor(y1)))
    x2_i = min(width, int(math.ceil(x2)))
    y2_i = min(height, int(math.ceil(y2)))
    if x2_i <= x1_i:
        x2_i = min(width, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(height, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def out_of_bounds_fraction(bbox: tuple[float, float, float, float], width: int, height: int) -> float:
    original_area = bbox_area(bbox)
    if original_area <= 0:
        return 1.0
    x1, y1, x2, y2 = bbox
    clipped_w = max(0.0, min(x2, width) - max(x1, 0.0))
    clipped_h = max(0.0, min(y2, height) - max(y1, 0.0))
    return float(max(0.0, 1.0 - ((clipped_w * clipped_h) / original_area)))


def polygon_to_mask(
    points: Iterable[tuple[float, float]],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)
    ox, oy = offset
    shifted = [(float(x - ox), float(y - oy)) for x, y in points]
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(shifted, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)


def infer_disaster_name(tile_id: str, payload: dict[str, Any] | None = None) -> str:
    if payload is not None:
        metadata = payload.get("metadata", {})
        if metadata.get("disaster"):
            return str(metadata["disaster"])
        if metadata.get("disaster_type"):
            return str(metadata["disaster_type"])
    token = str(tile_id)
    if "_" in token:
        return token.rsplit("_", 1)[0]
    return token


def load_label_png(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def save_label_png(path: str | Path, label_map: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(np.asarray(label_map, dtype=np.uint8), mode="L").save(path)
