from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from models.backbones.convnext_encoder import ConvNeXtFeatureEncoder
from models.fusion import SimpleBidirectionalFusionBlock
from models.heads import AdaptiveTauSafeHead, ClassificationHead, OrdinalCORNHead
from models.pooling import MaskedMultiScalePooling


class ConvNeXtMCDDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        pretrained_path: str = "",
        pretrained_url: str = "",
        auto_download_pretrained: bool = True,
        use_4ch_stem: bool = False,
        use_mask_gating: bool = True,
        mask_gate_strength: float = 0.2,
        return_multiscale: bool = True,
        num_classes: int = 4,
        tau_min: float = 0.85,
        tau_max: float = 1.15,
        tau_init: float = 1.0,
        dropout: float = 0.2,
        mlp_hidden_dim: int = 512,
        detach_tau_input: bool = True,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.detach_tau_input = bool(detach_tau_input)
        self.encoder = ConvNeXtFeatureEncoder(
            backbone=backbone,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            pretrained_url=pretrained_url,
            auto_download_pretrained=auto_download_pretrained,
            use_4ch_stem=use_4ch_stem,
            use_mask_gating=use_mask_gating,
            mask_gate_strength=mask_gate_strength,
            return_multiscale=return_multiscale,
            drop_path_rate=drop_path_rate,
        )
        self.feature_channels = self.encoder.feature_channels
        self.fusion_blocks = nn.ModuleDict(
            {
                name: SimpleBidirectionalFusionBlock(in_channels=channels, out_channels=channels, use_attention=True)
                for name, channels in self.feature_channels.items()
            }
        )
        self.pool = MaskedMultiScalePooling()
        self.pooled_feature_dim = sum(channels * 2 for channels in self.feature_channels.values())
        self.ce_head = ClassificationHead(
            in_features=self.pooled_feature_dim,
            hidden_features=mlp_hidden_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )
        self.corn_head = OrdinalCORNHead(
            in_features=self.pooled_feature_dim,
            hidden_features=mlp_hidden_dim,
            num_classes=self.num_classes,
            dropout=dropout,
        )
        self.tau_head = AdaptiveTauSafeHead(
            in_features=self.pooled_feature_dim,
            hidden_features=max(128, mlp_hidden_dim // 2),
            dropout=min(0.2, dropout),
            tau_min=tau_min,
            tau_max=tau_max,
            tau_init=tau_init,
        )

    def get_backbone_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.encoder.parameters() if parameter.requires_grad]

    def get_head_parameters(self) -> list[nn.Parameter]:
        excluded = {id(parameter) for parameter in self.get_backbone_parameters()}
        return [parameter for parameter in self.parameters() if parameter.requires_grad and id(parameter) not in excluded]

    def get_tau_head_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.tau_head.parameters() if parameter.requires_grad]

    def get_pretrained_report(self) -> dict[str, Any]:
        return self.encoder.get_pretrained_report()

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor, instance_mask: torch.Tensor) -> dict[str, Any]:
        pre_features = self.encoder(pre_image, instance_mask)
        post_features = self.encoder(post_image, instance_mask)

        fused_features = OrderedDict()
        for name in self.feature_channels.keys():
            fused_features[name] = self.fusion_blocks[name](pre_features[name], post_features[name])

        pooled_features = self.pool(fused_features, instance_mask)
        ce_logits = self.ce_head(pooled_features)
        corn_logits = self.corn_head(pooled_features)
        tau_input = pooled_features.detach() if self.detach_tau_input else pooled_features
        tau_outputs = self.tau_head(tau_input)
        tau = tau_outputs["tau"]
        tau_adjusted_logits = corn_logits / tau.unsqueeze(1)
        pred_labels = (torch.sigmoid(tau_adjusted_logits) > 0.5).sum(dim=1).clamp_max(self.num_classes - 1)

        return {
            "ce_logits": ce_logits,
            "corn_logits": corn_logits,
            "tau": tau,
            "raw_tau": tau_outputs["raw_tau"],
            "raw_delta_tau": tau_outputs["raw_delta_tau"],
            "tau_adjusted_logits": tau_adjusted_logits,
            "pred_labels": pred_labels,
            "features": pooled_features,
            "pre_features": pre_features,
            "post_features": post_features,
            "fused_features": fused_features,
        }
