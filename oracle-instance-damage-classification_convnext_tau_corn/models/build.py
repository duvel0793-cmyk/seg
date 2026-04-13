"""Model factory."""

from __future__ import annotations

from typing import Any, Dict

from models.convnext_oracle_tau_corn_model import ConvNeXtOracleTauCORNModel


def build_model(config: Dict[str, Any]) -> ConvNeXtOracleTauCORNModel:
    """Build the oracle damage classifier from config."""
    model_cfg = config["model"]
    return ConvNeXtOracleTauCORNModel(**model_cfg)

