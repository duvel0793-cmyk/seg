from __future__ import annotations

import copy
from typing import Any


DEFAULT_AUX_SOFT_LABEL_TARGET_DISTRIBUTION = [
    [0.90, 0.10, 0.00, 0.00],
    [0.15, 0.65, 0.20, 0.00],
    [0.00, 0.20, 0.65, 0.15],
    [0.00, 0.00, 0.10, 0.90],
]

DEFAULT_ORDINAL_MULTITASK_TARGET_DISTRIBUTION = [
    [0.90, 0.10, 0.00, 0.00],
    [0.10, 0.60, 0.25, 0.05],
    [0.05, 0.25, 0.60, 0.10],
    [0.00, 0.00, 0.10, 0.90],
]

DEFAULT_ORDINAL_MULTITASK_DIST_CLASS_WEIGHTS = [1.0, 1.35, 1.30, 1.0]
DEFAULT_ORDINAL_MULTITASK_REG_CLASS_WEIGHTS = [1.0, 1.40, 1.35, 1.0]


def _ensure_positive_ratio_dict(source_ratio: dict[str, Any] | None) -> dict[str, int]:
    payload = copy.deepcopy(source_ratio or {})
    resolved = {
        "xbd": max(1, int(payload.get("xbd", 1))),
        "bright": max(1, int(payload.get("bright", 1))),
    }
    return resolved


def ensure_input_pipeline_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.setdefault("data", {})
    root_dir = data_cfg.get("root_dir", "/home/lky/data/xBD")
    data_cfg.setdefault("root_dir", root_dir)
    data_cfg.setdefault("xbd_root", root_dir)
    data_cfg.setdefault("bright_root", "/home/lky/data/BRIGHT")
    data_cfg.setdefault("train_source", "xbd_only")
    data_cfg.setdefault("eval_source", "xbd_only")
    data_cfg["source_ratio"] = _ensure_positive_ratio_dict(data_cfg.get("source_ratio"))

    bright_cfg = data_cfg.setdefault("bright", {})
    bright_cfg.setdefault("train_list", None)
    bright_cfg.setdefault("val_list", None)
    bright_cfg.setdefault("test_list", None)
    bright_cfg.setdefault("train_ratio", 0.8)
    bright_cfg.setdefault("val_ratio", 0.1)
    bright_cfg.setdefault("split_seed", int(config.get("seed", 42)))
    bright_cfg.setdefault("target_mode", "binary_damage_mask")
    bright_cfg.setdefault("positive_class_name", None)
    bright_cfg.setdefault("label_mapping", None)
    bright_cfg.setdefault("post_convert_mode", "grayscale_to_rgb")
    bright_cfg.setdefault("component_connectivity", 2)

    input_mode_cfg = config.setdefault("input_mode", {})
    input_mode_cfg.setdefault("use_dual_scale", False)
    input_mode_cfg.setdefault("local_size", int(data_cfg.get("image_size", 224)))
    input_mode_cfg.setdefault("context_size", int(data_cfg.get("image_size", 224)))
    input_mode_cfg.setdefault("local_margin_ratio", 0.15)
    input_mode_cfg.setdefault("context_scale", 2.5)
    input_mode_cfg.setdefault("min_crop_size", 32)

    geometry_cfg = config.setdefault("geometry_prior", {})
    geometry_cfg.setdefault("use_instance_mask", True)
    geometry_cfg.setdefault("use_boundary_prior", False)
    geometry_cfg.setdefault("encoder_in_channels", 5)
    geometry_cfg.setdefault("boundary_width_px", 3)
    geometry_cfg.setdefault("dilation_radius", 3)
    geometry_cfg.setdefault("erosion_radius", 1)

    diff_cfg = config.setdefault("diff_input", {})
    diff_cfg.setdefault("return_abs_diff", False)
    diff_cfg.setdefault("feed_abs_diff_to_model", False)

    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("use_context_branch", False)
    model_cfg.setdefault("fuse_local_context_mode", "concat_absdiff")
    return config


def _validate_target_distribution(
    target_distribution: list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    num_classes: int = 4,
) -> list[list[float]]:
    rows = [[float(value) for value in row] for row in target_distribution]
    if len(rows) != int(num_classes):
        raise ValueError(
            f"Aux soft target distribution must contain {int(num_classes)} rows, got {len(rows)}."
        )

    normalized_rows: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if len(row) != int(num_classes):
            raise ValueError(
                f"Aux soft target distribution row {row_index} must contain {int(num_classes)} columns, "
                f"got {len(row)}."
            )
        row_sum = sum(row)
        if row_sum <= 0.0:
            raise ValueError(f"Aux soft target distribution row {row_index} must sum to a positive value.")
        normalized_rows.append([float(value) / float(row_sum) for value in row])
    return normalized_rows


def _validate_class_weights(
    class_weights: list[float] | tuple[float, ...],
    *,
    num_classes: int = 4,
    label: str,
) -> list[float]:
    weights = [float(value) for value in class_weights]
    if len(weights) != int(num_classes):
        raise ValueError(
            f"{label} must contain {int(num_classes)} values, got {len(weights)}."
        )
    if any(value <= 0.0 for value in weights):
        raise ValueError(f"{label} values must be > 0.")
    return weights


def ensure_auxiliary_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    config = ensure_input_pipeline_config_defaults(config)
    model_cfg = config.setdefault("model", {})
    train_cfg = config.setdefault("training", {})
    is_multitask_mode = str(train_cfg.get("loss_mode", "")) == "corn_ordinal_multitask_v1"

    aux_cfg = model_cfg.setdefault("aux_soft_label_head", {})
    aux_cfg.setdefault("enabled", False)
    aux_cfg.setdefault("hidden_dim", None)
    aux_cfg.setdefault("dropout", float(model_cfg.get("dropout", 0.2)))
    aux_cfg.setdefault("weight", 0.0)
    aux_cfg.setdefault(
        "target_distribution",
        copy.deepcopy(DEFAULT_AUX_SOFT_LABEL_TARGET_DISTRIBUTION),
    )
    aux_cfg["target_distribution"] = _validate_target_distribution(aux_cfg["target_distribution"])

    ordinal_multitask_cfg = config.setdefault("ordinal_multitask", {})
    dist_cfg = ordinal_multitask_cfg.setdefault("distribution_head", {})
    dist_cfg.setdefault("enabled", bool(is_multitask_mode))
    dist_cfg.setdefault("hidden_dim", 512)
    dist_cfg.setdefault("dropout", float(model_cfg.get("dropout", 0.2)))
    dist_cfg.setdefault("weight", 0.18)
    dist_cfg.setdefault(
        "target_distribution",
        copy.deepcopy(DEFAULT_ORDINAL_MULTITASK_TARGET_DISTRIBUTION),
    )
    dist_cfg.setdefault("class_weights", copy.deepcopy(DEFAULT_ORDINAL_MULTITASK_DIST_CLASS_WEIGHTS))
    dist_cfg["target_distribution"] = _validate_target_distribution(dist_cfg["target_distribution"])
    dist_cfg["class_weights"] = _validate_class_weights(
        dist_cfg["class_weights"],
        label="ordinal_multitask.distribution_head.class_weights",
    )

    reg_cfg = ordinal_multitask_cfg.setdefault("severity_regression", {})
    reg_cfg.setdefault("enabled", bool(is_multitask_mode))
    reg_cfg.setdefault("hidden_dim", 256)
    reg_cfg.setdefault("dropout", float(model_cfg.get("dropout", 0.2)))
    reg_cfg.setdefault("weight", 0.08)
    reg_cfg.setdefault("loss", "smooth_l1")
    reg_cfg.setdefault("class_weights", copy.deepcopy(DEFAULT_ORDINAL_MULTITASK_REG_CLASS_WEIGHTS))
    reg_cfg["class_weights"] = _validate_class_weights(
        reg_cfg["class_weights"],
        label="ordinal_multitask.severity_regression.class_weights",
    )

    rank_cfg = ordinal_multitask_cfg.setdefault("rank_contrastive", {})
    rank_cfg.setdefault("enabled", bool(train_cfg.get("enable_ordinal_contrastive", False) or is_multitask_mode))
    rank_cfg.setdefault("proj_dim", int(train_cfg.get("contrastive_proj_dim", 128)))
    rank_cfg.setdefault("hidden_dim", int(train_cfg.get("contrastive_hidden_features", 256)))
    rank_cfg.setdefault("dropout", float(train_cfg.get("contrastive_dropout", 0.10)))
    rank_cfg.setdefault("weight", float(train_cfg.get("lambda_contrastive", 0.05 if is_multitask_mode else 0.0)))
    rank_cfg.setdefault("distance", "cosine")
    rank_cfg.setdefault("margin_gap1", 0.40)
    rank_cfg.setdefault("margin_gap2", 0.80)
    rank_cfg.setdefault("margin_gap3", 1.10)
    rank_cfg.setdefault(
        "pair_weights",
        {
            "same": 1.0,
            "gap1": 1.2,
            "gap2": 1.0,
            "gap3": 1.1,
        },
    )
    rank_cfg.setdefault("boost_minor_major_pairs", 1.1)
    pair_weights = rank_cfg.setdefault("pair_weights", {})
    pair_weights.setdefault("same", 1.0)
    pair_weights.setdefault("gap1", 1.2)
    pair_weights.setdefault("gap2", 1.0)
    pair_weights.setdefault("gap3", 1.1)

    if float(rank_cfg["margin_gap3"]) <= float(rank_cfg["margin_gap2"]) or float(rank_cfg["margin_gap2"]) <= float(rank_cfg["margin_gap1"]):
        raise ValueError("ordinal_multitask.rank_contrastive requires margin_gap3 > margin_gap2 > margin_gap1.")

    consistency_cfg = ordinal_multitask_cfg.setdefault("consistency", {})
    consistency_cfg.setdefault("enabled", bool(is_multitask_mode))
    consistency_cfg.setdefault("weight_distribution", 0.03)
    consistency_cfg.setdefault("weight_severity", 0.03)

    contrastive_cfg = model_cfg.setdefault("ordinal_contrastive", {})
    contrastive_cfg.setdefault("enabled", bool(rank_cfg["enabled"]))
    contrastive_cfg.setdefault("proj_dim", int(rank_cfg["proj_dim"]))
    contrastive_cfg.setdefault("hidden_dim", int(rank_cfg["hidden_dim"]))
    contrastive_cfg.setdefault("dropout", float(rank_cfg["dropout"]))
    contrastive_cfg.setdefault("weight", float(rank_cfg["weight"]))
    contrastive_cfg.setdefault("distance", str(rank_cfg["distance"]))
    contrastive_cfg.setdefault("margin_adjacent", float(rank_cfg["margin_gap1"]))
    contrastive_cfg.setdefault("margin_far", float(rank_cfg["margin_gap2"]))
    contrastive_cfg.setdefault("far_weight", float(rank_cfg["pair_weights"]["gap2"]))
    contrastive_cfg.setdefault("margin_gap1", float(rank_cfg["margin_gap1"]))
    contrastive_cfg.setdefault("margin_gap2", float(rank_cfg["margin_gap2"]))
    contrastive_cfg.setdefault("margin_gap3", float(rank_cfg["margin_gap3"]))
    contrastive_cfg.setdefault("pair_weight_same", float(rank_cfg["pair_weights"]["same"]))
    contrastive_cfg.setdefault("pair_weight_gap1", float(rank_cfg["pair_weights"]["gap1"]))
    contrastive_cfg.setdefault("pair_weight_gap2", float(rank_cfg["pair_weights"]["gap2"]))
    contrastive_cfg.setdefault("pair_weight_gap3", float(rank_cfg["pair_weights"]["gap3"]))
    contrastive_cfg.setdefault("minor_major_pair_boost", float(rank_cfg["boost_minor_major_pairs"]))

    if float(contrastive_cfg["margin_far"]) <= float(contrastive_cfg["margin_adjacent"]):
        raise ValueError(
            "model.ordinal_contrastive.margin_far must be larger than margin_adjacent."
        )

    train_cfg.setdefault("save_epoch_checkpoints", False)
    train_cfg.setdefault("save_every_n_epochs", 1)
    train_cfg.setdefault("lambda_distribution", float(dist_cfg["weight"]))
    train_cfg.setdefault("lambda_severity_reg", float(reg_cfg["weight"]))
    train_cfg.setdefault("lambda_consistency_dist", float(consistency_cfg["weight_distribution"]))
    train_cfg.setdefault("lambda_consistency_severity", float(consistency_cfg["weight_severity"]))

    # Keep legacy training keys synchronized so older code paths and logs remain readable.
    train_cfg["enable_ordinal_contrastive"] = bool(rank_cfg["enabled"])
    train_cfg["lambda_contrastive"] = float(rank_cfg["weight"])
    train_cfg["contrastive_proj_dim"] = int(rank_cfg["proj_dim"])
    train_cfg["contrastive_hidden_features"] = int(rank_cfg["hidden_dim"])
    train_cfg["contrastive_dropout"] = float(rank_cfg["dropout"])
    train_cfg["contrastive_far_weight"] = float(rank_cfg["pair_weights"]["gap2"])
    train_cfg["contrastive_margin_adjacent"] = float(rank_cfg["margin_gap1"])
    train_cfg["contrastive_margin_far"] = float(rank_cfg["margin_gap2"])
    train_cfg["contrastive_margin_gap1"] = float(rank_cfg["margin_gap1"])
    train_cfg["contrastive_margin_gap2"] = float(rank_cfg["margin_gap2"])
    train_cfg["contrastive_margin_gap3"] = float(rank_cfg["margin_gap3"])
    train_cfg["contrastive_pair_weight_same"] = float(rank_cfg["pair_weights"]["same"])
    train_cfg["contrastive_pair_weight_gap1"] = float(rank_cfg["pair_weights"]["gap1"])
    train_cfg["contrastive_pair_weight_gap2"] = float(rank_cfg["pair_weights"]["gap2"])
    train_cfg["contrastive_pair_weight_gap3"] = float(rank_cfg["pair_weights"]["gap3"])
    train_cfg["contrastive_minor_major_pair_boost"] = float(rank_cfg["boost_minor_major_pairs"])

    model_multitask_cfg = model_cfg.setdefault("ordinal_multitask", {})
    model_multitask_cfg["enabled"] = bool(is_multitask_mode)
    model_multitask_cfg["distribution_head"] = copy.deepcopy(dist_cfg)
    model_multitask_cfg["severity_regression"] = copy.deepcopy(reg_cfg)
    model_multitask_cfg["consistency"] = copy.deepcopy(consistency_cfg)
    return config
