"""Mainline model builder for oracle instance upper-bound damage classification."""

from __future__ import annotations

from models.oracle_mc_damage_model import OracleMCDamageClassifier


MAINLINE_BACKBONE = "hybrid_vmamba"
MAINLINE_MODEL_TYPE = "oracle_mcd_corn"
MAINLINE_HEAD_TYPE = "corn"
MAINLINE_LOSS_MODE = "corn_adaptive_tau_safe"


def build_model(config: dict) -> OracleMCDamageClassifier:
    model_cfg = config["model"]
    training_cfg = config["training"]

    backbone = str(model_cfg.get("backbone", MAINLINE_BACKBONE))
    model_type = str(model_cfg.get("model_type", MAINLINE_MODEL_TYPE))
    head_type = str(model_cfg.get("head_type", MAINLINE_HEAD_TYPE))
    loss_mode = str(training_cfg.get("loss_mode", MAINLINE_LOSS_MODE))

    if backbone != MAINLINE_BACKBONE:
        raise ValueError(f"Only backbone='{MAINLINE_BACKBONE}' is supported, got '{backbone}'.")
    if model_type != MAINLINE_MODEL_TYPE:
        raise ValueError(f"Only model_type='{MAINLINE_MODEL_TYPE}' is supported, got '{model_type}'.")
    if head_type != MAINLINE_HEAD_TYPE:
        raise ValueError(f"Only head_type='{MAINLINE_HEAD_TYPE}' is supported, got '{head_type}'.")
    if loss_mode != MAINLINE_LOSS_MODE:
        raise ValueError(f"Only loss_mode='{MAINLINE_LOSS_MODE}' is supported, got '{loss_mode}'.")

    return OracleMCDamageClassifier(
        backbone=backbone,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        attention_reduction=int(model_cfg.get("channel_attention_reduction", 16)),
        num_classes=4,
        ambiguity_hidden_features=int(model_cfg.get("ambiguity_hidden_features", 256)),
        tau_min=float(training_cfg.get("tau_min", 0.12)),
        tau_max=float(training_cfg.get("tau_max", 0.45)),
        tau_base=float(training_cfg.get("tau_base", training_cfg.get("tau_target", 0.22))),
        delta_scale=float(training_cfg.get("delta_scale", 0.10)),
        tau_init=float(training_cfg.get("tau_init", 0.22)),
        tau_target=float(training_cfg.get("tau_target", 0.22)),
        tau_logit_scale=float(training_cfg.get("tau_logit_scale", 2.0)),
        tau_parameterization=str(training_cfg.get("tau_parameterization", "bounded_sigmoid")),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
        vmamba_pretrained_weight_path=str(model_cfg.get("vmamba_pretrained_weight_path", "")),
        conv_stage_depths=model_cfg.get("conv_stage_depths", [2, 2]),
        vmamba_stage_depths=model_cfg.get("vmamba_stage_depths", [6, 2]),
        dims=model_cfg.get("dims", [96, 192, 384, 768]),
        deep_scan_backend=str(model_cfg.get("deep_scan_backend", "official_ss2d")),
        vss_d_state=int(model_cfg.get("vss_d_state", 16)),
        vss_d_conv=int(model_cfg.get("vss_d_conv", 4)),
        vss_expand=int(model_cfg.get("vss_expand", 2)),
    )