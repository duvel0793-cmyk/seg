"""Mainline clean dual-scale damage network."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from models.backbones import build_backbone
from models.heads import BinaryHead, CEHead, CORNHead
from models.modules import (
    ConcatMLPFusion,
    ExplicitChangeBuilder,
    GatedScaleFusion,
    GlobalAvgPooling,
    LocalWindowAttention,
    MaskGuidedPooling,
)


class CleanDualScaleDamageNet(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any], loss_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.loss_cfg = dict(loss_cfg)
        self.num_classes = int(loss_cfg.get("num_classes", 4))
        self.use_local_window_attention = bool(model_cfg.get("use_local_window_attention", False))
        self.use_binary_aux = bool(model_cfg.get("use_binary_aux", False))
        self.use_ce_head = bool(model_cfg.get("use_ce_head", False))

        self.backbone = build_backbone(
            name=str(model_cfg.get("backbone", "convnextv2_tiny")),
            pretrained=bool(model_cfg.get("pretrained", True)),
            in_chans=int(model_cfg.get("in_chans", 3)),
            checkpoint_path=model_cfg.get("pretrained_path"),
        )
        self.feature_dim = int(self.backbone.get_feature_dim())

        self.tight_change = ExplicitChangeBuilder(self.feature_dim)
        self.context_change = ExplicitChangeBuilder(self.feature_dim)

        use_mask_pooling = bool(model_cfg.get("use_mask_pooling", True))
        self.pool_tight = MaskGuidedPooling() if use_mask_pooling else GlobalAvgPooling()
        self.pool_context = MaskGuidedPooling() if use_mask_pooling else GlobalAvgPooling()

        fusion_name = str(model_cfg.get("fusion", "concat_mlp")).lower()
        fusion_dim = int(model_cfg.get("fusion_dim", 512))
        dropout = float(model_cfg.get("dropout", 0.0))
        if fusion_name == "concat_mlp":
            self.fusion = ConcatMLPFusion(self.feature_dim, fusion_dim, dropout=dropout)
        elif fusion_name == "gated":
            self.fusion = GatedScaleFusion(self.feature_dim, fusion_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported fusion module: {fusion_name}")

        self.local_window_attention = (
            LocalWindowAttention(dim=self.feature_dim)
            if self.use_local_window_attention
            else nn.Identity()
        )
        self.corn_head = CORNHead(self.fusion.output_dim, num_classes=self.num_classes)
        self.ce_head = CEHead(self.fusion.output_dim, num_classes=self.num_classes) if self.use_ce_head else None
        self.binary_head = BinaryHead(self.fusion.output_dim) if self.use_binary_aux else None

    def _encode_scale(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        mask: torch.Tensor | None,
        change_builder: nn.Module,
        pooler: nn.Module,
        attention: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_feat = self.backbone(pre_image)
        post_feat = self.backbone(post_image)
        change_feat = change_builder(pre_feat, post_feat)
        if attention is not None:
            change_feat = attention(change_feat)
        pooled = pooler(change_feat, mask)
        return change_feat, pooled

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | None]:
        pre_tight = batch["pre_tight"]
        post_tight = batch["post_tight"]
        pre_context = batch["pre_context"]
        post_context = batch["post_context"]
        mask_tight = batch.get("mask_tight")
        mask_context = batch.get("mask_context")

        tight_change_map, tight_vec = self._encode_scale(
            pre_tight,
            post_tight,
            mask_tight,
            self.tight_change,
            self.pool_tight,
            attention=None,
        )
        context_change_map, context_vec = self._encode_scale(
            pre_context,
            post_context,
            mask_context,
            self.context_change,
            self.pool_context,
            attention=self.local_window_attention if self.use_local_window_attention else None,
        )

        fused = self.fusion(tight_vec, context_vec)
        outputs: Dict[str, torch.Tensor | None] = {
            "corn_logits": self.corn_head(fused),
            "logits": self.ce_head(fused) if self.ce_head is not None else None,
            "binary_logits": self.binary_head(fused) if self.binary_head is not None else None,
            "features": fused,
            "tight_change_map": tight_change_map,
            "context_change_map": context_change_map,
        }
        return outputs
