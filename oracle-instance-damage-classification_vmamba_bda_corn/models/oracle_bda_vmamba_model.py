from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from models.encoder_factory import build_encoder
from models.heads import (
    InstanceClassifierHead,
    OrdinalAmbiguityHead,
    OrdinalContrastiveProjector,
    OrdinalCORNHead,
)
from models.pooling import MaskedMultiScalePooling
from models.temporal_fusion import BDATemporalFusionBlock


class OracleBDAVMambaDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "vmamba_tiny",
        pretrained: bool = True,
        vmamba_pretrained_weight_path: str = "",
        drop_path_rate: float = 0.1,
        dropout: float = 0.2,
        num_classes: int = 4,
        head_type: str = "corn",
        temporal_fusion_type: str = "bda_fusion",
        temporal_reduction_ratio: int = 4,
        use_abs_diff: bool = True,
        use_sum_feat: bool = True,
        use_prod_feat: bool = False,
        temporal_gate: bool = True,
        pooling_modes: tuple[str, ...] = ("avg", "max", "attention"),
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
    ) -> None:
        super().__init__()
        if head_type not in {"standard", "corn"}:
            raise ValueError(f"Unsupported head_type='{head_type}'.")
        if temporal_fusion_type != "bda_fusion":
            raise ValueError(f"Unsupported temporal_fusion_type='{temporal_fusion_type}'.")

        self.num_classes = int(num_classes)
        self.head_type = str(head_type)
        self.temporal_fusion_type = str(temporal_fusion_type)
        self.detach_ambiguity_input = bool(detach_ambiguity_input)
        self.enable_ordinal_contrastive = bool(enable_ordinal_contrastive)

        self.encoder = build_encoder(
            backbone,
            in_channels=4,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            vmamba_pretrained_weight_path=vmamba_pretrained_weight_path,
        )
        self.feature_channels = OrderedDict(self.encoder.feature_channels)
        self.fusion_blocks = nn.ModuleDict(
            {
                name: BDATemporalFusionBlock(
                    channels=channels,
                    reduction_ratio=temporal_reduction_ratio,
                    use_abs_diff=use_abs_diff,
                    use_sum_feat=use_sum_feat,
                    use_prod_feat=use_prod_feat,
                    temporal_gate=temporal_gate,
                )
                for name, channels in self.feature_channels.items()
            }
        )
        self.pool = MaskedMultiScalePooling(
            feature_channels=self.feature_channels,
            pool_modes=pooling_modes,
        )
        self.pooled_feature_dim = MaskedMultiScalePooling.compute_output_dim(
            self.feature_channels,
            pool_modes=pooling_modes,
        )

        if self.head_type == "corn":
            self.classifier = OrdinalCORNHead(
                in_features=self.pooled_feature_dim,
                hidden_features=512,
                num_classes=self.num_classes,
                dropout=dropout,
            )
        else:
            self.classifier = InstanceClassifierHead(
                in_features=self.pooled_feature_dim,
                hidden_features=512,
                num_classes=self.num_classes,
                dropout=dropout,
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
        if use_ambiguity_head:
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

        fused_features: OrderedDict[str, torch.Tensor] = OrderedDict()
        change_hints: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name in self.feature_channels.keys():
            fusion_outputs = self.fusion_blocks[name](pre_features[name], post_features[name])
            fused_features[name] = fusion_outputs["fused"]
            change_hints[name] = fusion_outputs["change_hint"]

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
            "change_hints": change_hints,
            "encoder_feature_channels": dict(self.feature_channels),
            "temporal_fusion_type": self.temporal_fusion_type,
            "pretrained_load_summary": getattr(self.encoder, "pretrained_load_summary", None),
        }
