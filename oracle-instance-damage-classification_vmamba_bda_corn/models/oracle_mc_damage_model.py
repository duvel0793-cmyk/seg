from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.encoder_factory import build_encoder
from models.fusion import BidirectionalFusionBlock, ChannelAttentionGate
from models.heads import (
    InstanceClassifierHead,
    OrdinalAmbiguityHead,
    OrdinalContrastiveProjector,
    OrdinalCORNHead,
)
from models.pooling import MaskedMultiScalePooling


class OracleMCDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        attention_reduction: int = 16,
        num_classes: int = 4,
        head_type: str = "standard",
        use_ambiguity_head: bool = False,
        ambiguity_hidden_features: int = 256,
        tau_min: float = 0.10,
        tau_max: float = 0.60,
        tau_base: float = 0.22,
        delta_scale: float = 0.12,
        tau_init: float = 0.27,
        tau_target: float = 0.22,
        tau_logit_scale: float = 2.0,
        tau_parameterization: str = "sigmoid",
        detach_ambiguity_input: bool = False,
        enable_ordinal_contrastive: bool = False,
        contrastive_hidden_features: int = 512,
        contrastive_proj_dim: int = 128,
        contrastive_dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        vmamba_pretrained_weight_path: str = "",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.head_type = str(head_type)
        self.use_ambiguity_head = bool(use_ambiguity_head)
        self.detach_ambiguity_input = bool(detach_ambiguity_input)
        self.enable_ordinal_contrastive = bool(enable_ordinal_contrastive)

        if self.head_type not in {"standard", "corn"}:
            raise ValueError(f"Unsupported head_type='{self.head_type}'.")

        self.encoder = build_encoder(
            backbone,
            in_channels=4,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )
        channels = self.encoder.feature_channels

        self.fusion_blocks = nn.ModuleDict(
            {
                name: BidirectionalFusionBlock(in_channels=channel, out_channels=channel)
                for name, channel in channels.items()
            }
        )
        self.attention_gates = nn.ModuleDict(
            {
                name: ChannelAttentionGate(channels=channel, reduction=attention_reduction)
                for name, channel in channels.items()
            }
        )
        self.pool = MaskedMultiScalePooling(feature_channels=channels, pool_modes=("avg", "max"))
        self.pooled_feature_dim = MaskedMultiScalePooling.compute_output_dim(
            channels,
            pool_modes=("avg", "max"),
        )

        if self.head_type == "corn":
            self.classifier = OrdinalCORNHead(
                in_features=self.pooled_feature_dim,
                hidden_features=512,
                dropout=dropout,
                num_classes=self.num_classes,
            )
        else:
            self.classifier = InstanceClassifierHead(
                in_features=self.pooled_feature_dim,
                hidden_features=512,
                dropout=dropout,
                num_classes=self.num_classes,
            )

        self.contrastive_projector: OrdinalContrastiveProjector | None = None
        if self.enable_ordinal_contrastive:
            self.contrastive_projector = OrdinalContrastiveProjector(
                in_features=self.pooled_feature_dim,
                hidden_features=contrastive_hidden_features,
                proj_dim=contrastive_proj_dim,
                dropout=contrastive_dropout,
            )

        self.ambiguity_head: OrdinalAmbiguityHead | None = None
        if self.use_ambiguity_head:
            self.ambiguity_head = OrdinalAmbiguityHead(
                in_features=self.pooled_feature_dim,
                hidden_features=ambiguity_hidden_features,
                dropout=max(0.0, min(float(dropout) * 0.5, 0.2)),
                tau_min=tau_min,
                tau_max=tau_max,
                tau_base=tau_base,
                delta_scale=delta_scale,
                tau_init=tau_init,
                tau_target=tau_target,
                tau_logit_scale=tau_logit_scale,
                tau_parameterization=tau_parameterization,
            )

    def get_classifier_head_parameters(self) -> list[nn.Parameter]:
        params = [param for param in self.classifier.parameters() if param.requires_grad]
        params.extend(self.get_contrastive_head_parameters())
        return params

    def get_primary_classifier_head_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.classifier.parameters() if param.requires_grad]

    def get_trunk_parameters(self) -> list[nn.Parameter]:
        excluded_ids = {
            id(param)
            for param in [
                *self.get_primary_classifier_head_parameters(),
                *self.get_contrastive_head_parameters(),
                *self.get_ambiguity_head_parameters(),
            ]
        }
        return [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in excluded_ids
        ]

    def get_ambiguity_head_parameters(self) -> list[nn.Parameter]:
        if self.ambiguity_head is None:
            return []
        return [param for param in self.ambiguity_head.parameters() if param.requires_grad]

    def get_contrastive_head_parameters(self) -> list[nn.Parameter]:
        if self.contrastive_projector is None:
            return []
        return [param for param in self.contrastive_projector.parameters() if param.requires_grad]

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> dict[str, Any]:
        pre_input = torch.cat([pre_image, instance_mask], dim=1)
        post_input = torch.cat([post_image, instance_mask], dim=1)

        pre_features = self.encoder(pre_input)
        post_features = self.encoder(post_input)

        fused_features = {}
        for name in ("c2", "c3", "c4", "c5"):
            fused = self.fusion_blocks[name](pre_features[name], post_features[name])
            fused = self.attention_gates[name](fused)
            fused_features[name] = fused

        pooled = self.pool(fused_features, instance_mask)
        logits = self.classifier(pooled)
        corn_logits = logits if self.head_type == "corn" else None
        contrastive_embedding = None
        if self.contrastive_projector is not None:
            contrastive_embedding = self.contrastive_projector(pooled)
        ambiguity_input = pooled.detach() if self.detach_ambiguity_input else pooled
        tau_outputs = self.ambiguity_head(ambiguity_input) if self.ambiguity_head is not None else None
        tau = None if tau_outputs is None else tau_outputs["tau"]
        raw_tau = None if tau_outputs is None else tau_outputs["raw_tau"]
        raw_delta_tau = None if tau_outputs is None else tau_outputs.get("raw_delta_tau")
        return {
            "logits": logits,
            "corn_logits": corn_logits,
            "aux_logits": None,
            "pooled_feature": pooled,
            "contrastive_embedding": contrastive_embedding,
            "tau": tau,
            "raw_tau": raw_tau,
            "raw_delta_tau": raw_delta_tau,
            "head_type": self.head_type,
            "ambiguity_input_detached": self.detach_ambiguity_input,
            "pre_features": pre_features,
            "post_features": post_features,
            "fused_features": fused_features,
            "encoder_feature_channels": dict(self.encoder.feature_channels),
            "pretrained_load_summary": getattr(self.encoder, "pretrained_load_summary", None),
        }
