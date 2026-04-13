"""Model package for xBD oracle instance upper-bound damage classification."""

from models.baselines import PostOnlyDamageClassifier, SiameseSimpleDamageClassifier
from models.oracle_mc_damage_model import OracleMCDamageClassifier


def build_model(config: dict):
    model_cfg = config["model"]
    training_cfg = config["training"]
    model_type = str(model_cfg["model_type"])
    head_type = str(model_cfg.get("head_type", "standard"))
    backbone = str(model_cfg.get("backbone", "hybrid_vmamba"))
    pretrained = bool(model_cfg.get("pretrained", True))
    drop_path_rate = float(model_cfg.get("drop_path_rate", 0.1))
    vmamba_pretrained_weight_path = str(model_cfg.get("vmamba_pretrained_weight_path", ""))
    loss_mode = str(training_cfg["loss_mode"])

    if model_type == "post_only":
        if loss_mode in {"corn", "corn_adaptive_tau_safe"}:
            raise ValueError("CORN is only supported for oracle multi-temporal instance models.")
        return PostOnlyDamageClassifier(
            backbone=backbone,
            pretrained=pretrained,
            dropout=float(model_cfg["dropout"]),
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )

    if model_type == "siamese_simple":
        if loss_mode in {"corn", "corn_adaptive_tau_safe"}:
            raise ValueError("CORN is only supported for oracle multi-temporal instance models.")
        return SiameseSimpleDamageClassifier(
            backbone=backbone,
            pretrained=pretrained,
            dropout=float(model_cfg["dropout"]),
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )

    if model_type in {"oracle_mcd", "oracle_mcd_corn"}:
        if model_type == "oracle_mcd_corn":
            head_type = "corn"
        use_ambiguity_head = bool(
            loss_mode in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
        )
        detach_ambiguity_input = bool(loss_mode == "corn_adaptive_tau_safe")
        default_tau_parameterization = (
            "bounded_sigmoid"
            if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
            else "sigmoid"
        )
        tau_parameterization = str(training_cfg.get("tau_parameterization", default_tau_parameterization))
        return OracleMCDamageClassifier(
            backbone=backbone,
            pretrained=pretrained,
            dropout=float(model_cfg["dropout"]),
            attention_reduction=int(model_cfg.get("channel_attention_reduction", 16)),
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
            enable_ordinal_contrastive=bool(training_cfg.get("enable_ordinal_contrastive", False)),
            contrastive_hidden_features=int(training_cfg.get("contrastive_hidden_features", 512)),
            contrastive_proj_dim=int(training_cfg.get("contrastive_proj_dim", 128)),
            contrastive_dropout=float(training_cfg.get("contrastive_dropout", 0.1)),
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )

    raise ValueError(f"Unsupported model_type='{model_type}'.")
