from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.encoder import build_feature_encoder
from models.fusion import BidirectionalFusionBlock, ChannelAttentionGate
from models.heads import (
    AuxiliaryDistributionHead,
    InstanceClassifierHead,
    OrdinalAmbiguityHead,
    OrdinalContrastiveProjector,
    OrdinalCORNHead,
    SeverityRegressionHead,
)
from models.pooling import MaskedMultiScalePooling


class OracleMCDamageClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "convnextv2_tiny",
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
        enable_aux_soft_label_head: bool = False,
        aux_soft_label_hidden_dim: int | None = None,
        aux_soft_label_dropout: float | None = None,
        enable_ordinal_distribution_head: bool = False,
        ordinal_distribution_hidden_dim: int = 512,
        ordinal_distribution_dropout: float | None = None,
        enable_severity_regression_head: bool = False,
        severity_regression_hidden_dim: int = 256,
        severity_regression_dropout: float | None = None,
        enable_ordinal_contrastive: bool = False,
        contrastive_hidden_features: int = 512,
        contrastive_proj_dim: int = 128,
        contrastive_dropout: float = 0.1,
        encoder_in_channels: int = 4,
        use_boundary_prior: bool = False,
        use_context_branch: bool = False,
        fuse_local_context_mode: str = "concat_absdiff",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.head_type = str(head_type)
        self.use_ambiguity_head = bool(use_ambiguity_head)
        self.detach_ambiguity_input = bool(detach_ambiguity_input)
        self.enable_aux_soft_label_head = bool(enable_aux_soft_label_head)
        self.enable_ordinal_distribution_head = bool(enable_ordinal_distribution_head)
        self.enable_severity_regression_head = bool(enable_severity_regression_head)
        self.enable_ordinal_contrastive = bool(enable_ordinal_contrastive)
        self.encoder_in_channels = int(encoder_in_channels)
        self.use_boundary_prior = bool(use_boundary_prior)
        self.use_context_branch = bool(use_context_branch)
        self.fuse_local_context_mode = str(fuse_local_context_mode)

        if self.head_type not in {"standard", "corn"}:
            raise ValueError(f"Unsupported head_type='{self.head_type}'.")
        if self.fuse_local_context_mode not in {"concat_absdiff"}:
            raise ValueError(f"Unsupported fuse_local_context_mode='{self.fuse_local_context_mode}'.")

        self.encoder = build_feature_encoder(
            backbone=backbone,
            in_channels=self.encoder_in_channels,
            pretrained=pretrained,
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
        self.pool = MaskedMultiScalePooling()
        self.single_scale_feature_dim = sum(channel * 2 for channel in channels.values())
        self.pooled_feature_dim = (
            self.single_scale_feature_dim * 3 if self.use_context_branch else self.single_scale_feature_dim
        )

        aux_hidden_features = self._resolve_aux_hidden_dim(aux_soft_label_hidden_dim)
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

        self.aux_soft_label_head: AuxiliaryDistributionHead | None = None
        if self.enable_aux_soft_label_head:
            self.aux_soft_label_head = AuxiliaryDistributionHead(
                in_features=self.pooled_feature_dim,
                hidden_features=aux_hidden_features,
                dropout=float(dropout if aux_soft_label_dropout is None else aux_soft_label_dropout),
                num_classes=self.num_classes,
            )

        self.ordinal_distribution_head: AuxiliaryDistributionHead | None = None
        if self.enable_ordinal_distribution_head:
            self.ordinal_distribution_head = AuxiliaryDistributionHead(
                in_features=self.pooled_feature_dim,
                hidden_features=int(max(1, ordinal_distribution_hidden_dim)),
                dropout=float(dropout if ordinal_distribution_dropout is None else ordinal_distribution_dropout),
                num_classes=self.num_classes,
            )

        self.severity_regression_head: SeverityRegressionHead | None = None
        if self.enable_severity_regression_head:
            self.severity_regression_head = SeverityRegressionHead(
                in_features=self.pooled_feature_dim,
                hidden_features=int(max(1, severity_regression_hidden_dim)),
                dropout=float(dropout if severity_regression_dropout is None else severity_regression_dropout),
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

    def _resolve_aux_hidden_dim(self, configured_hidden_dim: int | None) -> int:
        if configured_hidden_dim is not None and int(configured_hidden_dim) > 0:
            return int(configured_hidden_dim)
        return min(max(128, self.pooled_feature_dim // 2), 512)

    def get_classifier_head_parameters(self) -> list[nn.Parameter]:
        params = [param for param in self.classifier.parameters() if param.requires_grad]
        params.extend(self.get_auxiliary_head_parameters())
        params.extend(self.get_contrastive_head_parameters())
        return params

    def get_primary_classifier_head_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.classifier.parameters() if param.requires_grad]

    def get_auxiliary_head_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.aux_soft_label_head is not None:
            params.extend([param for param in self.aux_soft_label_head.parameters() if param.requires_grad])
        if self.ordinal_distribution_head is not None:
            params.extend([param for param in self.ordinal_distribution_head.parameters() if param.requires_grad])
        if self.severity_regression_head is not None:
            params.extend([param for param in self.severity_regression_head.parameters() if param.requires_grad])
        return params

    def get_trunk_parameters(self) -> list[nn.Parameter]:
        excluded_ids = {
            id(param)
            for param in [
                *self.get_primary_classifier_head_parameters(),
                *self.get_auxiliary_head_parameters(),
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

    def _compose_encoder_input(
        self,
        image: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None,
    ) -> torch.Tensor:
        channels = [image, instance_mask]
        if self.encoder_in_channels >= 5:
            if instance_boundary is None:
                instance_boundary = torch.zeros_like(instance_mask)
            channels.append(instance_boundary)
        return torch.cat(channels, dim=1)

    def _encode_scale_branch(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pre_input = self._compose_encoder_input(pre_image, instance_mask, instance_boundary)
        post_input = self._compose_encoder_input(post_image, instance_mask, instance_boundary)

        pre_features = self.encoder(pre_input)
        post_features = self.encoder(post_input)

        fused_features = []
        for name in ("c2", "c3", "c4", "c5"):
            fused = self.fusion_blocks[name](pre_features[name], post_features[name])
            fused = self.attention_gates[name](fused)
            fused_features.append(fused)
        return self.pool(fused_features, instance_mask)

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None = None,
        context_pre_image: torch.Tensor | None = None,
        context_post_image: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_boundary: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        local_pooled = self._encode_scale_branch(pre_image, post_image, instance_mask, instance_boundary)
        context_pooled = None
        pooled = local_pooled
        if self.use_context_branch:
            if context_pre_image is None or context_post_image is None or context_mask is None:
                context_pooled = local_pooled
            else:
                context_pooled = self._encode_scale_branch(
                    context_pre_image,
                    context_post_image,
                    context_mask,
                    context_boundary,
                )
            pooled = torch.cat([local_pooled, context_pooled, (local_pooled - context_pooled).abs()], dim=1)

        logits = self.classifier(pooled)
        corn_logits = logits if self.head_type == "corn" else None
        aux_logits = self.aux_soft_label_head(pooled) if self.aux_soft_label_head is not None else None
        distribution_logits = self.ordinal_distribution_head(pooled) if self.ordinal_distribution_head is not None else None
        severity_score = self.severity_regression_head(pooled) if self.severity_regression_head is not None else None
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
            "aux_logits": aux_logits,
            "distribution_logits": distribution_logits,
            "severity_score": severity_score,
            "pooled_feature": pooled,
            "local_pooled_feature": local_pooled,
            "context_pooled_feature": context_pooled,
            "contrastive_embedding": contrastive_embedding,
            "tau": tau,
            "raw_tau": raw_tau,
            "raw_delta_tau": raw_delta_tau,
            "head_type": self.head_type,
            "ambiguity_input_detached": self.detach_ambiguity_input,
            "use_context_branch": self.use_context_branch,
            "use_boundary_prior": self.use_boundary_prior,
        }
