"""Single-scale baseline model for ablations."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from models.backbones import build_backbone
from models.heads import BinaryHead, CEHead, CORNHead
from models.modules import ExplicitChangeBuilder, GlobalAvgPooling, MaskGuidedPooling


class SingleScaleDamageNet(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any], loss_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.loss_cfg = dict(loss_cfg)
        self.num_classes = int(loss_cfg.get("num_classes", 4))
        self.scale = str(model_cfg.get("scale", "tight"))
        self.use_binary_aux = bool(model_cfg.get("use_binary_aux", False))
        self.use_ce_head = bool(model_cfg.get("use_ce_head", False))

        self.backbone = build_backbone(
            name=str(model_cfg.get("backbone", "convnextv2_tiny")),
            pretrained=bool(model_cfg.get("pretrained", True)),
            in_chans=int(model_cfg.get("in_chans", 3)),
            checkpoint_path=model_cfg.get("pretrained_path"),
        )
        self.feature_dim = int(self.backbone.get_feature_dim())
        self.change_builder = ExplicitChangeBuilder(self.feature_dim)
        self.pool = MaskGuidedPooling() if bool(model_cfg.get("use_mask_pooling", True)) else GlobalAvgPooling()
        hidden_dim = int(model_cfg.get("fusion_dim", 512))
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg.get("dropout", 0.0))),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.corn_head = CORNHead(hidden_dim, num_classes=self.num_classes)
        self.ce_head = CEHead(hidden_dim, num_classes=self.num_classes) if self.use_ce_head else None
        self.binary_head = BinaryHead(hidden_dim) if self.use_binary_aux else None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | None]:
        pre = batch[f"pre_{self.scale}"]
        post = batch[f"post_{self.scale}"]
        mask = batch.get(f"mask_{self.scale}")
        pre_feat = self.backbone(pre)
        post_feat = self.backbone(post)
        change_feat = self.change_builder(pre_feat, post_feat)
        pooled = self.pool(change_feat, mask)
        fused = self.projector(pooled)
        return {
            "corn_logits": self.corn_head(fused),
            "logits": self.ce_head(fused) if self.ce_head is not None else None,
            "binary_logits": self.binary_head(fused) if self.binary_head is not None else None,
            "features": fused,
        }
