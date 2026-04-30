"""Model builders for Clean-MODG."""

from __future__ import annotations

import copy
from typing import Any, Dict

from models.damage_models.clean_dual_scale import CleanDualScaleDamageNet
from models.damage_models.hier_dual_scale import HierDualScaleDamageNet
from models.damage_models.single_scale import SingleScaleDamageNet


def build_model(config: Dict[str, Any]):
    model_cfg = copy.deepcopy(config.get("model", {}))
    loss_cfg = config.get("loss", {})
    loss_type = str(loss_cfg.get("type", "corn")).lower()
    model_cfg["use_ce_head"] = bool(model_cfg.get("use_ce_head", False) or loss_type in {"ce", "focal"})

    name = str(model_cfg.get("name", "clean_dual_scale")).lower()
    if model_cfg.get("use_neighborhood", False):
        raise NotImplementedError(
            "Neighborhood scale ablation is declared in config but not implemented in Clean-MODG mainline. "
            "Please add a dedicated neighborhood data branch before enabling it."
        )

    if name == "clean_dual_scale":
        return CleanDualScaleDamageNet(model_cfg=model_cfg, loss_cfg=loss_cfg)
    if name == "hier_dual_scale":
        return HierDualScaleDamageNet(model_cfg=model_cfg, loss_cfg=loss_cfg)
    if name == "single_scale":
        return SingleScaleDamageNet(model_cfg=model_cfg, loss_cfg=loss_cfg)
    raise ValueError(f"Unsupported model name: {name}")
