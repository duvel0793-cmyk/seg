from __future__ import annotations

from models.model import DamageInstanceModel
from models.multi_context_model import MultiContextDamageModel


def build_model(config: dict) -> DamageInstanceModel | MultiContextDamageModel:
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    enable_damage_aux = bool(
        max(
            float(loss_cfg.get("loss_damage_tight_weight", 0.0)),
            float(loss_cfg.get("loss_damage_context_weight", 0.0)),
            float(loss_cfg.get("loss_damage_neighborhood_weight", 0.0)),
        )
        > 0.0
    )
    enable_severity_aux = bool(
        bool(model_cfg.get("enable_severity_aux", True))
        and float(loss_cfg.get("loss_severity_aux_weight", 0.0)) > 0.0
    )
    if bool(model_cfg.get("use_multi_context_model", False)):
        return MultiContextDamageModel(
            backbone_name=str(model_cfg["backbone_name"]),
            pretrained=bool(model_cfg["pretrained"]),
            input_mode=str(model_cfg["input_mode"]),
            feature_dim=int(model_cfg["feature_dim"]),
            dropout=float(model_cfg["dropout"]),
            use_change_suppression=bool(model_cfg["use_change_suppression"]),
            change_block_channels=int(model_cfg["change_block_channels"]),
            enable_pseudo_suppression=bool(model_cfg["enable_pseudo_suppression"]),
            fuse_change_to_tokens=bool(model_cfg["fuse_change_to_tokens"]),
            change_residual_scale=float(model_cfg["change_residual_scale"]),
            change_gate_init_gamma=float(model_cfg["change_gate_init_gamma"]),
            gate_temperature=float(model_cfg["gate_temperature"]),
            gate_bias_init=float(model_cfg["gate_bias_init"]),
            enable_damage_aux=enable_damage_aux,
            enable_severity_aux=enable_severity_aux,
            tight_token_count=int(model_cfg["tight_token_count"]),
            context_token_count=int(model_cfg["context_token_count"]),
            neighborhood_token_count=int(model_cfg["neighborhood_token_count"]),
            local_attention_heads=int(model_cfg["local_attention_heads"]),
            local_attention_layers=int(model_cfg["local_attention_layers"]),
            tight_window_size=int(model_cfg["tight_window_size"]),
            context_window_size=int(model_cfg["context_window_size"]),
            neighborhood_window_size=int(model_cfg["neighborhood_window_size"]),
            use_cross_scale_attention=bool(model_cfg["use_cross_scale_attention"]),
            cross_scale_heads=int(model_cfg["cross_scale_heads"]),
            cross_scale_layers=int(model_cfg["cross_scale_layers"]),
            cross_scale_dropout=float(model_cfg["cross_scale_dropout"]),
            context_dropout_prob=float(model_cfg["context_dropout_prob"]),
            neighborhood_dropout_prob=float(model_cfg["neighborhood_dropout_prob"]),
            enable_neighborhood_graph=bool(model_cfg["enable_neighborhood_graph"]),
            graph_k_neighbors=int(model_cfg["graph_k_neighbors"]),
            graph_layers=int(model_cfg["graph_layers"]),
            graph_hidden_dim=int(model_cfg["graph_hidden_dim"]),
            graph_attention_heads=int(model_cfg["graph_attention_heads"]),
            graph_use_distance_bias=bool(model_cfg["graph_use_distance_bias"]),
            use_tight_branch=bool(model_cfg.get("use_tight_branch", True)),
            use_context_branch=bool(model_cfg.get("use_context_branch", True)),
            use_neighborhood_scale=bool(model_cfg.get("use_neighborhood_scale", True)),
            use_local_attention=bool(model_cfg.get("use_local_attention", True)),
            safe_multicontext_mode=bool(model_cfg.get("safe_multicontext_mode", True)),
            cross_scale_layerscale_init=float(model_cfg.get("cross_scale_layerscale_init", 1.0e-3)),
            cross_scale_residual_max=float(model_cfg.get("cross_scale_residual_max", 0.3)),
            neighborhood_branch_gate_init=float(model_cfg.get("neighborhood_branch_gate_init", -4.0)),
            neighborhood_residual_scale_init=float(model_cfg.get("neighborhood_residual_scale_init", 0.02)),
            neighborhood_residual_scale_max=float(model_cfg.get("neighborhood_residual_scale_max", 0.20)),
            neighborhood_aux_enabled=bool(model_cfg.get("neighborhood_aux_enabled", False)),
            neighborhood_gate_bias_init=float(model_cfg.get("neighborhood_gate_bias_init", -4.0)),
            neighborhood_gate_temperature=float(model_cfg.get("neighborhood_gate_temperature", 3.0)),
            freeze_neighborhood_strength_after_epoch=int(model_cfg.get("freeze_neighborhood_strength_after_epoch", 7)),
            neighborhood_branch_gate_max=float(model_cfg.get("neighborhood_branch_gate_max", 0.04)),
            num_classes=4,
        )
    return DamageInstanceModel(
        backbone_name=str(model_cfg["backbone_name"]),
        pretrained=bool(model_cfg["pretrained"]),
        input_mode=str(model_cfg["input_mode"]),
        feature_dim=int(model_cfg["feature_dim"]),
        token_count=int(model_cfg["token_count"]),
        token_mixer_layers=int(model_cfg["token_mixer_layers"]),
        token_mixer_heads=int(model_cfg["token_mixer_heads"]),
        dropout=float(model_cfg["dropout"]),
        use_change_suppression=bool(model_cfg["use_change_suppression"]),
        change_block_channels=int(model_cfg["change_block_channels"]),
        enable_pseudo_suppression=bool(model_cfg["enable_pseudo_suppression"]),
        fuse_change_to_tokens=bool(model_cfg["fuse_change_to_tokens"]),
        change_residual_scale=float(model_cfg["change_residual_scale"]),
        change_gate_init_gamma=float(model_cfg["change_gate_init_gamma"]),
        gate_temperature=float(model_cfg["gate_temperature"]),
        gate_bias_init=float(model_cfg["gate_bias_init"]),
        enable_damage_aux=enable_damage_aux,
        enable_severity_aux=enable_severity_aux,
        num_classes=4,
    )
