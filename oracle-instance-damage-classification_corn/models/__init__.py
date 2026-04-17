"""Model package for xBD oracle instance damage classification."""

from models.baselines import PostOnlyDamageClassifier, SiameseSimpleDamageClassifier
from models.oracle_mc_damage_model import OracleMCDamageClassifier
from utils.config_utils import ensure_auxiliary_config_defaults


def _resolve_encoder_input_channels(config: dict) -> int:
    geometry_cfg = config.get("geometry_prior", {})
    use_boundary_prior = bool(geometry_cfg.get("use_boundary_prior", False))
    requested = int(geometry_cfg.get("encoder_in_channels", 5 if use_boundary_prior else 4))
    if use_boundary_prior:
        return max(5, requested)
    return max(4, requested)


def build_model(config: dict):
    config = ensure_auxiliary_config_defaults(config)
    model_cfg = config["model"]
    training_cfg = config["training"]
    model_type = model_cfg["model_type"]
    head_type = str(model_cfg.get("head_type", "standard"))
    loss_mode = str(training_cfg["loss_mode"])
    aux_head_cfg = model_cfg.get("aux_soft_label_head", {})
    multitask_cfg = config.get("ordinal_multitask", {})
    distribution_cfg = multitask_cfg.get("distribution_head", {})
    severity_cfg = multitask_cfg.get("severity_regression", {})
    rank_cfg = multitask_cfg.get("rank_contrastive", {})
    contrastive_cfg = model_cfg.get("ordinal_contrastive", {})
    encoder_in_channels = _resolve_encoder_input_channels(config)
    use_boundary_prior = bool(config.get("geometry_prior", {}).get("use_boundary_prior", False))
    use_context_branch = bool(model_cfg.get("use_context_branch", False))
    fuse_local_context_mode = str(model_cfg.get("fuse_local_context_mode", "concat_absdiff"))

    if model_type == "post_only":
        if loss_mode in {"corn", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"}:
            raise ValueError("CORN is only supported for the oracle_mcd backbone path.")
        return PostOnlyDamageClassifier(
            backbone=model_cfg["backbone"],
            pretrained=bool(model_cfg["pretrained"]),
            dropout=float(model_cfg["dropout"]),
            encoder_in_channels=encoder_in_channels,
        )

    if model_type == "siamese_simple":
        if loss_mode in {"corn", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"}:
            raise ValueError("CORN is only supported for the oracle_mcd backbone path.")
        return SiameseSimpleDamageClassifier(
            backbone=model_cfg["backbone"],
            pretrained=bool(model_cfg["pretrained"]),
            dropout=float(model_cfg["dropout"]),
            encoder_in_channels=encoder_in_channels,
        )

    if model_type in {"oracle_mcd", "oracle_mcd_corn"}:
        if model_type == "oracle_mcd_corn":
            head_type = "corn"
        use_ambiguity_head = bool(
            loss_mode
            in {
                "adaptive_ucl_cda",
                "adaptive_ucl_cda_v2",
                "adaptive_ucl_cda_v3",
                "corn_adaptive_tau_safe",
                "corn_ordinal_multitask_v1",
            }
        )
        detach_ambiguity_input = bool(loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"})
        default_tau_parameterization = (
            "bounded_sigmoid"
            if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"}
            else "sigmoid"
        )
        tau_parameterization = str(training_cfg.get("tau_parameterization", default_tau_parameterization))
        return OracleMCDamageClassifier(
            backbone=model_cfg["backbone"],
            pretrained=bool(model_cfg["pretrained"]),
            dropout=float(model_cfg["dropout"]),
            attention_reduction=int(model_cfg["channel_attention_reduction"]),
            num_classes=4,
            head_type=head_type,
            use_ambiguity_head=use_ambiguity_head,
            ambiguity_hidden_features=int(model_cfg.get("ambiguity_hidden_features", 256)),
            tau_min=float(training_cfg.get("tau_min", 0.10)),
            tau_max=float(training_cfg.get("tau_max", 0.60)),
            tau_base=float(training_cfg.get("tau_base", training_cfg.get("tau_target", 0.22))),
            delta_scale=float(training_cfg.get("delta_scale", 0.12)),
            tau_init=float(training_cfg.get("tau_init", 0.27)),
            tau_target=float(training_cfg.get("tau_target", training_cfg.get("tau_init", 0.22))),
            tau_logit_scale=float(training_cfg.get("tau_logit_scale", 2.0)),
            tau_parameterization=tau_parameterization,
            detach_ambiguity_input=detach_ambiguity_input,
            enable_aux_soft_label_head=bool(aux_head_cfg.get("enabled", False)),
            aux_soft_label_hidden_dim=(
                None if aux_head_cfg.get("hidden_dim", None) is None else int(aux_head_cfg["hidden_dim"])
            ),
            aux_soft_label_dropout=float(aux_head_cfg.get("dropout", model_cfg.get("dropout", 0.2))),
            enable_ordinal_distribution_head=bool(distribution_cfg.get("enabled", False)),
            ordinal_distribution_hidden_dim=int(distribution_cfg.get("hidden_dim", 512)),
            ordinal_distribution_dropout=float(distribution_cfg.get("dropout", model_cfg.get("dropout", 0.2))),
            enable_severity_regression_head=bool(severity_cfg.get("enabled", False)),
            severity_regression_hidden_dim=int(severity_cfg.get("hidden_dim", 256)),
            severity_regression_dropout=float(severity_cfg.get("dropout", model_cfg.get("dropout", 0.2))),
            enable_ordinal_contrastive=bool(rank_cfg.get("enabled", contrastive_cfg.get("enabled", False))),
            contrastive_hidden_features=int(
                rank_cfg.get(
                    "hidden_dim",
                    contrastive_cfg.get("hidden_dim", training_cfg.get("contrastive_hidden_features", 256)),
                )
            ),
            contrastive_proj_dim=int(
                rank_cfg.get(
                    "proj_dim",
                    contrastive_cfg.get("proj_dim", training_cfg.get("contrastive_proj_dim", 128)),
                )
            ),
            contrastive_dropout=float(
                rank_cfg.get(
                    "dropout",
                    contrastive_cfg.get("dropout", training_cfg.get("contrastive_dropout", 0.1)),
                )
            ),
            encoder_in_channels=encoder_in_channels,
            use_boundary_prior=use_boundary_prior,
            use_context_branch=use_context_branch,
            fuse_local_context_mode=fuse_local_context_mode,
        )

    raise ValueError(f"Unsupported model_type='{model_type}'.")
