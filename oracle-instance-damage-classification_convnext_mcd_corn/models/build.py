from __future__ import annotations

from models.convnext_mcd_damage_model import ConvNeXtMCDDamageClassifier


def build_model(config: dict) -> ConvNeXtMCDDamageClassifier:
    model_cfg = config["model"]
    if str(model_cfg["model_type"]) != "convnext_mcd_corn":
        raise ValueError("This project only supports model_type=convnext_mcd_corn.")
    return ConvNeXtMCDDamageClassifier(
        backbone=str(model_cfg.get("backbone", "convnext_tiny")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        pretrained_path=str(model_cfg.get("pretrained_path", "")),
        pretrained_url=str(model_cfg.get("pretrained_url", "")),
        auto_download_pretrained=bool(model_cfg.get("auto_download_pretrained", True)),
        use_4ch_stem=bool(model_cfg.get("use_4ch_stem", False)),
        use_mask_gating=bool(model_cfg.get("use_mask_gating", True)),
        mask_gate_strength=float(model_cfg.get("mask_gate_strength", 0.2)),
        return_multiscale=bool(model_cfg.get("return_multiscale", True)),
        num_classes=int(model_cfg.get("num_classes", 4)),
        tau_min=float(model_cfg.get("tau_min", 0.85)),
        tau_max=float(model_cfg.get("tau_max", 1.15)),
        tau_init=float(model_cfg.get("tau_init", 1.0)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 512)),
        detach_tau_input=bool(model_cfg.get("detach_tau_input", True)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
    )

