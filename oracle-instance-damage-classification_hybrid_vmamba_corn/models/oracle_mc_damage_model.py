from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.encoder_hybrid_vmamba import HybridConvVMambaEncoder
from models.fusion import BidirectionalFusionBlock, ChannelAttentionGate
from models.heads import OrdinalAmbiguityHead, OrdinalCORNHead
from models.pooling import MaskedMultiScalePooling


class OracleMCDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "hybrid_vmamba",
        pretrained: bool = True,
        dropout: float = 0.2,
        attention_reduction: int = 16,
        num_classes: int = 4,
        ambiguity_hidden_features: int = 256,
        tau_min: float = 0.12,
        tau_max: float = 0.45,
        tau_base: float = 0.22,
        delta_scale: float = 0.10,
        tau_init: float = 0.22,
        tau_target: float = 0.22,
        tau_logit_scale: float = 2.0,
        tau_parameterization: str = "bounded_sigmoid",
        drop_path_rate: float = 0.1,
        vmamba_pretrained_weight_path: str = "",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.head_type = "corn"
        self.detach_ambiguity_input = True
        if str(backbone) != "hybrid_vmamba":
            raise ValueError("OracleMCDamageClassifier only supports backbone='hybrid_vmamba'.")

        self.encoder = HybridConvVMambaEncoder(
            backbone="hybrid_vmamba",
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
        pooled_feature_dim = MaskedMultiScalePooling.compute_output_dim(channels, pool_modes=("avg", "max"))

        self.classifier = OrdinalCORNHead(
            in_features=pooled_feature_dim,
            hidden_features=512,
            dropout=dropout,
            num_classes=self.num_classes,
        )
        self.ambiguity_head = OrdinalAmbiguityHead(
            in_features=pooled_feature_dim,
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
        return [param for param in self.classifier.parameters() if param.requires_grad]

    def get_primary_classifier_head_parameters(self) -> list[nn.Parameter]:
        return self.get_classifier_head_parameters()

    def get_ambiguity_head_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.ambiguity_head.parameters() if param.requires_grad]

    def get_trunk_parameters(self) -> list[nn.Parameter]:
        excluded_ids = {
            id(param)
            for param in [
                *self.get_primary_classifier_head_parameters(),
                *self.get_ambiguity_head_parameters(),
            ]
        }
        return [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in excluded_ids
        ]

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
        tau_outputs = self.ambiguity_head(pooled.detach())

        return {
            "logits": logits,
            "tau": tau_outputs["tau"],
            "raw_tau": tau_outputs["raw_tau"],
            "head_type": self.head_type,
            "ambiguity_input_detached": self.detach_ambiguity_input,
        }
