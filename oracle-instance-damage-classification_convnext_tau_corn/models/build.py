"""Model factory."""

from __future__ import annotations

from typing import Any, Dict

from models.convnext_oracle_tau_corn_model import ConvNeXtOracleTauCORNModel


def build_model(config: Dict[str, Any]) -> ConvNeXtOracleTauCORNModel:
    """Build the oracle damage classifier from config."""
    model_cfg = config["model"]
    model = ConvNeXtOracleTauCORNModel(**model_cfg)
    report = model.get_backbone_pretrained_report()
    print(
        "[build_model] "
        f"backbone_variant={model_cfg.get('backbone_variant')} "
        f"load_pretrained={model_cfg.get('load_pretrained', False)} "
        f"auto_download_pretrained={model_cfg.get('auto_download_pretrained', False)} "
        f"pretrained_path={model_cfg.get('pretrained_path', '')} "
        f"pretrained_url={model_cfg.get('pretrained_url', '')} "
        f"freeze_backbone={model_cfg.get('freeze_backbone', False)} "
        f"tau_mode={model_cfg.get('tau_mode', 'per_threshold')} "
        f"pretrained_loaded={report.get('pretrained_loaded', False)}"
    )
    return model
