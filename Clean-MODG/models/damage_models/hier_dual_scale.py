"""Hierarchical dual-scale variant with binary auxiliary supervision."""

from __future__ import annotations

from typing import Any, Dict

from models.damage_models.clean_dual_scale import CleanDualScaleDamageNet


class HierDualScaleDamageNet(CleanDualScaleDamageNet):
    def __init__(self, model_cfg: Dict[str, Any], loss_cfg: Dict[str, Any]) -> None:
        model_cfg = dict(model_cfg)
        model_cfg["use_binary_aux"] = True
        super().__init__(model_cfg=model_cfg, loss_cfg=loss_cfg)
