from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class PixelAuxDecoder(nn.Module):
    def __init__(self, in_channels_by_level: OrderedDict[str, int], decoder_channels: int = 128) -> None:
        super().__init__()
        self.levels = ("c2", "c3", "c4", "c5")
        self.lateral = nn.ModuleDict(
            {
                level: nn.Sequential(
                    nn.Conv2d(int(in_channels_by_level[level]), int(decoder_channels), kernel_size=1, bias=False),
                    nn.BatchNorm2d(int(decoder_channels)),
                    nn.GELU(),
                )
                for level in self.levels
            }
        )
        self.smooth = nn.ModuleDict(
            {
                level: nn.Sequential(
                    nn.Conv2d(int(decoder_channels), int(decoder_channels), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(int(decoder_channels)),
                    nn.GELU(),
                )
                for level in self.levels
            }
        )

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        p5 = self.smooth["c5"](self.lateral["c5"](features["c5"]))
        p4 = self.smooth["c4"](
            self.lateral["c4"](features["c4"])
            + F.interpolate(p5, size=features["c4"].shape[-2:], mode="bilinear", align_corners=False)
        )
        p3 = self.smooth["c3"](
            self.lateral["c3"](features["c3"])
            + F.interpolate(p4, size=features["c3"].shape[-2:], mode="bilinear", align_corners=False)
        )
        p2 = self.smooth["c2"](
            self.lateral["c2"](features["c2"])
            + F.interpolate(p3, size=features["c2"].shape[-2:], mode="bilinear", align_corners=False)
        )
        return p2


class PixelSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(int(in_channels), int(in_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(in_channels)),
            nn.GELU(),
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


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
        enable_pixel_auxiliary: bool = False,
        pixel_aux_decoder_channels: int = 128,
        pixel_aux_use_local_branch_only: bool = True,
        pixel_aux_building_head_enabled: bool = True,
        pixel_aux_damage_head_enabled: bool = True,
        pixel_aux_damage_num_classes: int = 5,
        pixel_aux_use_bridge: bool = True,
        pixel_bridge_use_boundary_stats: bool = True,
        pixel_bridge_detach: bool = False,
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
        self.enable_pixel_auxiliary = bool(enable_pixel_auxiliary)
        self.pixel_aux_decoder_channels = int(max(8, pixel_aux_decoder_channels))
        self.pixel_aux_use_local_branch_only = bool(pixel_aux_use_local_branch_only)
        self.pixel_aux_building_head_enabled = bool(pixel_aux_building_head_enabled)
        self.pixel_aux_damage_head_enabled = bool(pixel_aux_damage_head_enabled)
        self.pixel_aux_damage_num_classes = int(max(2, pixel_aux_damage_num_classes))
        self.pixel_aux_use_bridge = bool(pixel_aux_use_bridge)
        self.pixel_bridge_use_boundary_stats = bool(pixel_bridge_use_boundary_stats)
        self.pixel_bridge_detach = bool(pixel_bridge_detach)

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
        self.base_pooled_feature_dim = (
            self.single_scale_feature_dim * 3 if self.use_context_branch else self.single_scale_feature_dim
        )
        self.pixel_bridge_feature_dim = 0
        if self.enable_pixel_auxiliary and self.pixel_aux_use_bridge:
            self.pixel_bridge_feature_dim = 1 + self.pixel_aux_damage_num_classes + 1
            if self.pixel_bridge_use_boundary_stats:
                self.pixel_bridge_feature_dim += 3
        self.pooled_feature_dim = self.base_pooled_feature_dim + self.pixel_bridge_feature_dim

        self.pixel_aux_decoder: PixelAuxDecoder | None = None
        self.local_building_head: PixelSegmentationHead | None = None
        self.local_damage_head: PixelSegmentationHead | None = None
        if self.enable_pixel_auxiliary:
            self.pixel_aux_decoder = PixelAuxDecoder(
                in_channels_by_level=channels,
                decoder_channels=self.pixel_aux_decoder_channels,
            )
            if self.pixel_aux_building_head_enabled:
                self.local_building_head = PixelSegmentationHead(
                    in_channels=self.pixel_aux_decoder_channels,
                    out_channels=1,
                )
            if self.pixel_aux_damage_head_enabled:
                self.local_damage_head = PixelSegmentationHead(
                    in_channels=self.pixel_aux_decoder_channels,
                    out_channels=self.pixel_aux_damage_num_classes,
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
        if self.pixel_aux_decoder is not None:
            params.extend([param for param in self.pixel_aux_decoder.parameters() if param.requires_grad])
        if self.local_building_head is not None:
            params.extend([param for param in self.local_building_head.parameters() if param.requires_grad])
        if self.local_damage_head is not None:
            params.extend([param for param in self.local_damage_head.parameters() if param.requires_grad])
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
        *,
        return_fused_features: bool = False,
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor] | None]:
        pre_input = self._compose_encoder_input(pre_image, instance_mask, instance_boundary)
        post_input = self._compose_encoder_input(post_image, instance_mask, instance_boundary)

        pre_features = self.encoder(pre_input)
        post_features = self.encoder(post_input)

        fused_features: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name in ("c2", "c3", "c4", "c5"):
            fused = self.fusion_blocks[name](pre_features[name], post_features[name])
            fused = self.attention_gates[name](fused)
            fused_features[name] = fused
        pooled = self.pool(list(fused_features.values()), instance_mask)
        if not return_fused_features:
            return pooled, None
        return pooled, fused_features

    @staticmethod
    def _resize_binary_mask(mask: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        resized = F.interpolate(mask.float(), size=target_size, mode="nearest")
        return (resized > 0.5).float()

    @staticmethod
    def _masked_channel_average(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        fallback_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mask = mask.float()
        denominator = mask.sum(dim=(2, 3), keepdim=True)
        averaged = (values * mask).sum(dim=(2, 3), keepdim=True) / denominator.clamp_min(eps)
        if fallback_mask is not None:
            fallback_mask = fallback_mask.float()
            fallback_denominator = fallback_mask.sum(dim=(2, 3), keepdim=True)
            fallback_averaged = (values * fallback_mask).sum(dim=(2, 3), keepdim=True) / fallback_denominator.clamp_min(eps)
            has_valid_mask = (denominator > eps).expand(-1, values.size(1), -1, -1)
            averaged = torch.where(has_valid_mask, averaged, fallback_averaged)
        return averaged.squeeze(-1).squeeze(-1)

    def _build_pixel_bridge_feature(
        self,
        *,
        instance_mask: torch.Tensor,
        instance_boundary: torch.Tensor | None,
        local_building_logits: torch.Tensor | None,
        local_damage_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = int(instance_mask.size(0))
        bridge_dtype = instance_mask.dtype
        bridge_device = instance_mask.device
        feature_size = tuple(instance_mask.shape[-2:])
        if local_building_logits is not None:
            feature_size = tuple(local_building_logits.shape[-2:])
            bridge_dtype = local_building_logits.dtype
            bridge_device = local_building_logits.device
        elif local_damage_logits is not None:
            feature_size = tuple(local_damage_logits.shape[-2:])
            bridge_dtype = local_damage_logits.dtype
            bridge_device = local_damage_logits.device

        if local_building_logits is not None:
            building_probability = torch.sigmoid(local_building_logits.float())
        else:
            building_probability = torch.zeros(
                batch_size,
                1,
                feature_size[0],
                feature_size[1],
                device=bridge_device,
                dtype=torch.float32,
            )

        if local_damage_logits is not None:
            damage_probability = torch.softmax(local_damage_logits.float(), dim=1)
        else:
            damage_probability = torch.zeros(
                batch_size,
                self.pixel_aux_damage_num_classes,
                feature_size[0],
                feature_size[1],
                device=bridge_device,
                dtype=torch.float32,
            )
            damage_probability[:, 0:1] = 1.0

        resized_instance_mask = self._resize_binary_mask(
            instance_mask.to(device=bridge_device),
            feature_size,
        )
        resized_boundary_mask = torch.zeros_like(resized_instance_mask)
        if instance_boundary is not None:
            resized_boundary_mask = self._resize_binary_mask(
                instance_boundary.to(device=bridge_device),
                feature_size,
            )

        severity_values = torch.arange(
            self.pixel_aux_damage_num_classes,
            device=bridge_device,
            dtype=torch.float32,
        )
        if self.pixel_aux_damage_num_classes >= 2:
            severity_values[1:] -= 1.0
        expected_severity_map = (damage_probability * severity_values.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

        instance_building_mean = self._masked_channel_average(building_probability, resized_instance_mask)
        instance_damage_means = self._masked_channel_average(damage_probability, resized_instance_mask)
        instance_expected_mean = self._masked_channel_average(expected_severity_map, resized_instance_mask)

        bridge_parts = [instance_building_mean, instance_damage_means, instance_expected_mean]

        if self.pixel_bridge_use_boundary_stats:
            boundary_building_mean = self._masked_channel_average(
                building_probability,
                resized_boundary_mask,
                fallback_mask=resized_instance_mask,
            )
            boundary_expected_mean = self._masked_channel_average(
                expected_severity_map,
                resized_boundary_mask,
                fallback_mask=resized_instance_mask,
            )
            expected_gap = instance_expected_mean - boundary_expected_mean
            bridge_parts.extend([boundary_building_mean, boundary_expected_mean, expected_gap])

        bridge_feature = torch.cat(bridge_parts, dim=1)
        if bridge_feature.size(1) < self.pixel_bridge_feature_dim:
            bridge_feature = torch.cat(
                [
                    bridge_feature,
                    bridge_feature.new_zeros(batch_size, self.pixel_bridge_feature_dim - bridge_feature.size(1)),
                ],
                dim=1,
            )
        elif bridge_feature.size(1) > self.pixel_bridge_feature_dim:
            bridge_feature = bridge_feature[:, : self.pixel_bridge_feature_dim]

        bridge_feature = bridge_feature.to(device=bridge_device, dtype=bridge_dtype)
        if self.pixel_bridge_detach:
            bridge_feature = bridge_feature.detach()
        return bridge_feature

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
        local_pooled, local_fused_features = self._encode_scale_branch(
            pre_image,
            post_image,
            instance_mask,
            instance_boundary,
            return_fused_features=self.enable_pixel_auxiliary,
        )
        context_pooled = None
        pooled_base = local_pooled
        if self.use_context_branch:
            if context_pre_image is None or context_post_image is None or context_mask is None:
                context_pooled = local_pooled
            else:
                context_pooled, _ = self._encode_scale_branch(
                    context_pre_image,
                    context_post_image,
                    context_mask,
                    context_boundary,
                    return_fused_features=False,
                )
            pooled_base = torch.cat([local_pooled, context_pooled, (local_pooled - context_pooled).abs()], dim=1)

        local_pixel_decoder_feature = None
        local_building_logits = None
        local_damage_logits = None
        if self.enable_pixel_auxiliary and self.pixel_aux_decoder is not None and local_fused_features is not None:
            local_pixel_decoder_feature = self.pixel_aux_decoder(local_fused_features)
            local_pixel_decoder_feature = F.interpolate(
                local_pixel_decoder_feature,
                size=instance_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if self.local_building_head is not None:
                local_building_logits = self.local_building_head(local_pixel_decoder_feature)
            if self.local_damage_head is not None:
                local_damage_logits = self.local_damage_head(local_pixel_decoder_feature)

        pixel_bridge_feature = None
        pooled = pooled_base
        if self.enable_pixel_auxiliary and self.pixel_aux_use_bridge and self.pixel_bridge_feature_dim > 0:
            pixel_bridge_feature = self._build_pixel_bridge_feature(
                instance_mask=instance_mask,
                instance_boundary=instance_boundary,
                local_building_logits=local_building_logits,
                local_damage_logits=local_damage_logits,
            )
            pooled = torch.cat([pooled_base, pixel_bridge_feature], dim=1)

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
            "pooled_feature_base": pooled_base,
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
            "local_building_logits": local_building_logits,
            "local_damage_logits": local_damage_logits,
            "local_pixel_decoder_feature": local_pixel_decoder_feature,
            "pixel_bridge_feature": pixel_bridge_feature,
            "pixel_bridge_enabled": bool(self.enable_pixel_auxiliary and self.pixel_aux_use_bridge),
        }
