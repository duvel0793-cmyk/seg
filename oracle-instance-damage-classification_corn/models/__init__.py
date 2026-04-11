"""Model package for xBD oracle instance damage classification."""

from models.baselines import PostOnlyDamageClassifier, SiameseSimpleDamageClassifier
from models.oracle_mc_damage_model import OracleMCDamageClassifier


def build_model(config: dict):
    model_cfg = config["model"]
    training_cfg = config["training"]
    model_type = model_cfg["model_type"]
    head_type = str(model_cfg.get("head_type", "standard"))

    if model_type == "post_only":
        if training_cfg["loss_mode"] in {"corn", "corn_adaptive_tau_safe"}:
            raise ValueError("CORN is only supported for the oracle_mcd backbone path.")
        return PostOnlyDamageClassifier(
            backbone=model_cfg["backbone"],
            pretrained=bool(model_cfg["pretrained"]),
            dropout=float(model_cfg["dropout"]),
        )

    if model_type == "siamese_simple":
        if training_cfg["loss_mode"] in {"corn", "corn_adaptive_tau_safe"}:
            raise ValueError("CORN is only supported for the oracle_mcd backbone path.")
        return SiameseSimpleDamageClassifier(
            backbone=model_cfg["backbone"],
            pretrained=bool(model_cfg["pretrained"]),
            dropout=float(model_cfg["dropout"]),
        )

    if model_type in {"oracle_mcd", "oracle_mcd_corn"}:
        if model_type == "oracle_mcd_corn":
            head_type = "corn"
        loss_mode = str(training_cfg["loss_mode"])
        use_ambiguity_head = bool(
            loss_mode in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"}
        )
        detach_ambiguity_input = bool(loss_mode == "corn_adaptive_tau_safe")
        default_tau_parameterization = "bounded_sigmoid" if loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe"} else "sigmoid"
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
            enable_ordinal_contrastive=bool(training_cfg.get("enable_ordinal_contrastive", False)),
            contrastive_hidden_features=int(training_cfg.get("contrastive_hidden_features", 512)),
            contrastive_proj_dim=int(training_cfg.get("contrastive_proj_dim", 128)),
            contrastive_dropout=float(training_cfg.get("contrastive_dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model_type='{model_type}'.")
