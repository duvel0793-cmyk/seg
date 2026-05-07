from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ConvNeXtV2Backbone, DEFAULT_CONVNEXTV2_MODEL, resolve_input_mode
from models.classifier import FlatCORNOrdinalHead, OrdinalCORNHead, ResidualFeatureCalibration, normalize_corn_outputs
from models.conditional_review_head import ConditionalReviewHead
from models.cross_scale_attention import CrossScaleTargetConditionedAttention
from models.hierarchical_head import TwoStageHierarchicalOrdinalHead
from models.pixel_ordinal_head import build_pixel_ordinal_head
from models.scale_aux_fusion_head import ScaleWiseAuxFusionHead
from models.severity_aware_scale_router import SeverityAwareScaleRouter
from models.scale_branch import ScaleBranch
from models.structural_two_stage_head import StructuralTwoStageHead


class EvidenceHierarchicalDamageModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = DEFAULT_CONVNEXTV2_MODEL,
        pretrained: bool = True,
        input_mode: str = "rgbm",
        feature_dim: int = 768,
        token_dim: int = 256,
        evidence_dim: int = 128,
        dropout: float = 0.1,
        use_tight_branch: bool = True,
        use_context_branch: bool = True,
        use_neighborhood_branch: bool = True,
        enable_alignment: bool = True,
        enable_prepost_fusion: bool = True,
        fusion_mode: str = "diff_prod_concat",
        change_block_type: str = "auto",
        enable_damage_bdfm_lite: bool = False,
        enable_damage_aware_block: bool = True,
        enable_change_gate: bool = True,
        enable_edqa_lite: bool = False,
        edqa_lite_scales: list[str] | tuple[str, ...] = ("context",),
        edqa_lite_window_size: int | None = None,
        edqa_lite_heads: int | None = None,
        edqa_lite_init_alpha: float = 0.1,
        use_cross_scale_attention: bool = True,
        cross_scale_fusion_mode: str = "tight_query_all_kv",
        use_evidence_head: bool = True,
        use_hierarchical_head: bool = True,
        use_global_corn_aux: bool = True,
        enable_minor_boundary_aux: bool = False,
        head_type: str | None = None,
        tight_token_count: int = 8,
        context_token_count: int = 8,
        neighborhood_token_count: int = 12,
        local_attention_heads: int = 4,
        local_attention_layers: int = 2,
        local_attention_layers_tight: int | None = None,
        local_attention_layers_context: int | None = None,
        local_attention_layers_neighborhood: int | None = None,
        tight_window_size: int = 7,
        context_window_size: int = 7,
        neighborhood_window_size: int = 8,
        cross_scale_heads: int = 4,
        cross_scale_layers: int = 2,
        cross_scale_dropout: float = 0.1,
        context_dropout_prob: float = 0.1,
        neighborhood_dropout_prob: float = 0.2,
        evidence_topk_ratio: float = 0.1,
        evidence_threshold: float = 0.5,
        damage_decision_threshold: float = 0.5,
        evidence_fusion_mode: str = "concat_mlp_original",
        evidence_gate_init_logit: float = -4.0,
        enable_feature_calibration: bool = True,
        evidence_scales: list[str] | tuple[str, ...] | None = None,
        use_structural_two_stage_head: bool = False,
        structural_two_stage_hidden_dim: int = 768,
        structural_two_stage_dropout: float = 0.1,
        structural_two_stage_use_for_prediction: bool = False,
        structural_two_stage_prob_mix_weight: float = 0.0,
        use_conditional_review_head: bool = False,
        conditional_review_use_for_prediction: bool = False,
        conditional_review_prob_mix_weight: float = 0.0,
        conditional_review_hidden_dim: int = 512,
        conditional_review_dropout: float = 0.1,
        conditional_review_alpha_init: float = 0.05,
        conditional_review_alpha_max: float = 0.30,
        conditional_review_detach_base_probs: bool = True,
        use_severity_aware_scale_router: bool = False,
        scale_router_hidden_dim: int = 512,
        scale_router_dropout: float = 0.1,
        scale_router_use_for_prediction: bool = False,
        scale_router_prob_mix_weight: float = 0.0,
        scale_router_init_gates: list[float] | tuple[float, ...] = (0.5, 0.7, 0.3, 0.25),
        use_scale_aux_fusion_head: bool = False,
        scale_aux_use_for_prediction: bool = False,
        scale_aux_hidden_dim: int = 512,
        scale_aux_dropout: float = 0.1,
        scale_aux_fusion_mode: str = "dynamic",
        scale_aux_initial_weights: list[float] | tuple[float, float, float, float] = (0.85, 0.06, 0.06, 0.03),
        scale_aux_temperature: float = 1.0,
        scale_aux_detach_base_probs_for_gate: bool = True,
        scale_aux_gate_warmup_epochs: int = 0,
        scale_aux_use_prior_during_warmup: bool = False,
        enable_pixel_ordinal_line: bool = False,
        pixel_line_feature_source: str = "tight",
        pixel_line_head_type: str = "two_stage_ordinal",
        pixel_line_hidden_dim: int = 256,
        pixel_line_dropout: float = 0.1,
        pixel_line_aggregation: str = "mean_topk_mix",
        pixel_line_topk_ratio: float = 0.2,
        pixel_line_mean_weight: float = 0.7,
        pixel_line_topk_weight: float = 0.3,
        pixel_line_use_for_final_prediction: bool = False,
        pixel_line_prob_mix_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not use_tight_branch:
            raise ValueError("tight branch is required")
        self.input_spec = resolve_input_mode(input_mode)
        self.append_mask_channel = bool(self.input_spec["append_mask_channel"])
        self.use_context_branch = bool(use_context_branch)
        self.use_neighborhood_branch = bool(use_neighborhood_branch)
        self.enable_alignment = bool(enable_alignment)
        self.enable_prepost_fusion = bool(enable_prepost_fusion)
        self.change_block_type = str(change_block_type)
        self.enable_damage_bdfm_lite = bool(enable_damage_bdfm_lite)
        self.enable_damage_aware_block = bool(enable_damage_aware_block)
        self.enable_change_gate = bool(enable_change_gate)
        self.enable_edqa_lite = bool(enable_edqa_lite)
        self.edqa_lite_scales = tuple(str(name) for name in edqa_lite_scales)
        self.edqa_lite_window_size = None if edqa_lite_window_size is None else int(edqa_lite_window_size)
        self.edqa_lite_heads = None if edqa_lite_heads is None else int(edqa_lite_heads)
        self.edqa_lite_init_alpha = float(edqa_lite_init_alpha)
        self.use_cross_scale_attention = bool(use_cross_scale_attention)
        self.cross_scale_fusion_mode = str(cross_scale_fusion_mode)
        self.use_hierarchical_head = bool(use_hierarchical_head)
        self.use_global_corn_aux = bool(use_global_corn_aux)
        self.enable_feature_calibration = bool(enable_feature_calibration)
        self.enable_minor_boundary_aux = bool(enable_minor_boundary_aux)
        self.use_structural_two_stage_head = bool(use_structural_two_stage_head)
        self.structural_two_stage_use_for_prediction = bool(structural_two_stage_use_for_prediction)
        self.structural_two_stage_prob_mix_weight = float(structural_two_stage_prob_mix_weight)
        self.use_conditional_review_head = bool(use_conditional_review_head)
        self.conditional_review_use_for_prediction = bool(conditional_review_use_for_prediction)
        self.conditional_review_prob_mix_weight = float(conditional_review_prob_mix_weight)
        self.use_severity_aware_scale_router = bool(use_severity_aware_scale_router)
        self.scale_router_use_for_prediction = bool(scale_router_use_for_prediction)
        self.scale_router_prob_mix_weight = float(scale_router_prob_mix_weight)
        self.use_scale_aux_fusion_head = bool(use_scale_aux_fusion_head)
        self.scale_aux_use_for_prediction = bool(scale_aux_use_for_prediction)
        self.scale_aux_gate_warmup_epochs = int(scale_aux_gate_warmup_epochs)
        self.scale_aux_use_prior_during_warmup = bool(scale_aux_use_prior_during_warmup)
        self.enable_pixel_ordinal_line = bool(enable_pixel_ordinal_line)
        self.pixel_line_feature_source = str(pixel_line_feature_source)
        self.pixel_line_head_type = str(pixel_line_head_type)
        self.pixel_line_use_for_final_prediction = bool(pixel_line_use_for_final_prediction)
        self.pixel_line_prob_mix_weight = float(pixel_line_prob_mix_weight)
        self.evidence_fusion_mode = str(evidence_fusion_mode)
        self.head_type = str(head_type or ("two_stage_hierarchical" if self.use_hierarchical_head else "flat_corn"))
        self.evidence_scale_names = set(evidence_scales or ("tight", "context", "neighborhood"))
        if self.evidence_fusion_mode not in {"concat_mlp_original", "gated_residual", "none"}:
            raise ValueError(f"Unsupported evidence_fusion_mode='{self.evidence_fusion_mode}'")
        if self.cross_scale_fusion_mode not in {"tight_query_all_kv", "tight_query_context_kv"}:
            raise ValueError(f"Unsupported cross_scale_fusion_mode='{self.cross_scale_fusion_mode}'")
        if self.head_type not in {"two_stage_hierarchical", "flat_corn"}:
            raise ValueError(f"Unsupported head_type='{self.head_type}'")
        if self.pixel_line_feature_source not in {"tight", "context", "neighborhood", "multi_scale"}:
            raise ValueError(f"Unsupported pixel_line_feature_source='{self.pixel_line_feature_source}'")
        if self.pixel_line_head_type not in {"two_stage_ordinal", "flat_corn"}:
            raise ValueError(f"Unsupported pixel_line_head_type='{self.pixel_line_head_type}'")

        tight_local_layers = int(local_attention_layers if local_attention_layers_tight is None else local_attention_layers_tight)
        context_local_layers = int(local_attention_layers if local_attention_layers_context is None else local_attention_layers_context)
        neighborhood_local_layers = int(
            local_attention_layers if local_attention_layers_neighborhood is None else local_attention_layers_neighborhood
        )

        self.backbone = ConvNeXtV2Backbone(
            backbone_name=backbone_name,
            in_channels=int(self.input_spec["branch_in_channels"]),
            pretrained=pretrained,
        )
        c4_channels = int(self.backbone.feature_channels["c4"])
        c5_channels = int(self.backbone.feature_channels["c5"])

        self.tight_branch = ScaleBranch(
            scale_name="tight",
            c4_channels=c4_channels,
            c5_channels=c5_channels,
            feature_dim=token_dim,
            token_count=tight_token_count,
            window_size=tight_window_size,
            local_attention_heads=local_attention_heads,
            local_attention_layers=tight_local_layers,
            dropout=dropout,
            evidence_dim=evidence_dim,
            evidence_topk_ratio=evidence_topk_ratio,
            evidence_threshold=evidence_threshold,
            use_evidence_head=(use_evidence_head and ("tight" in self.evidence_scale_names)),
            enable_alignment=self.enable_alignment,
            enable_prepost_fusion=self.enable_prepost_fusion,
            fusion_mode=fusion_mode,
            change_block_type=self.change_block_type,
            enable_damage_bdfm_lite=self.enable_damage_bdfm_lite,
            enable_damage_aware_block=self.enable_damage_aware_block,
            enable_change_gate=self.enable_change_gate,
            enable_edqa_lite=self.enable_edqa_lite,
            edqa_lite_scales=self.edqa_lite_scales,
            edqa_lite_window_size=self.edqa_lite_window_size,
            edqa_lite_heads=self.edqa_lite_heads,
            edqa_lite_init_alpha=self.edqa_lite_init_alpha,
        )
        self.context_branch = (
            ScaleBranch(
                scale_name="context",
                c4_channels=c4_channels,
                c5_channels=c5_channels,
                feature_dim=token_dim,
                token_count=context_token_count,
                window_size=context_window_size,
                local_attention_heads=local_attention_heads,
                local_attention_layers=context_local_layers,
                dropout=dropout,
                evidence_dim=evidence_dim,
                evidence_topk_ratio=evidence_topk_ratio,
                evidence_threshold=evidence_threshold,
                use_evidence_head=(use_evidence_head and ("context" in self.evidence_scale_names)),
                enable_alignment=self.enable_alignment,
                enable_prepost_fusion=self.enable_prepost_fusion,
                fusion_mode=fusion_mode,
                change_block_type=self.change_block_type,
                enable_damage_bdfm_lite=self.enable_damage_bdfm_lite,
                enable_damage_aware_block=self.enable_damage_aware_block,
                enable_change_gate=self.enable_change_gate,
                enable_edqa_lite=self.enable_edqa_lite,
                edqa_lite_scales=self.edqa_lite_scales,
                edqa_lite_window_size=self.edqa_lite_window_size,
                edqa_lite_heads=self.edqa_lite_heads,
                edqa_lite_init_alpha=self.edqa_lite_init_alpha,
            )
            if self.use_context_branch
            else None
        )
        self.neighborhood_branch = (
            ScaleBranch(
                scale_name="neighborhood",
                c4_channels=c4_channels,
                c5_channels=c5_channels,
                feature_dim=token_dim,
                token_count=neighborhood_token_count,
                window_size=neighborhood_window_size,
                local_attention_heads=local_attention_heads,
                local_attention_layers=neighborhood_local_layers,
                dropout=dropout,
                evidence_dim=evidence_dim,
                evidence_topk_ratio=evidence_topk_ratio,
                evidence_threshold=evidence_threshold,
                use_evidence_head=(use_evidence_head and ("neighborhood" in self.evidence_scale_names)),
                enable_alignment=self.enable_alignment,
                enable_prepost_fusion=self.enable_prepost_fusion,
                fusion_mode=fusion_mode,
                change_block_type=self.change_block_type,
                enable_damage_bdfm_lite=self.enable_damage_bdfm_lite,
                enable_damage_aware_block=self.enable_damage_aware_block,
                enable_change_gate=self.enable_change_gate,
                enable_edqa_lite=self.enable_edqa_lite,
                edqa_lite_scales=self.edqa_lite_scales,
                edqa_lite_window_size=self.edqa_lite_window_size,
                edqa_lite_heads=self.edqa_lite_heads,
                edqa_lite_init_alpha=self.edqa_lite_init_alpha,
            )
            if self.use_neighborhood_branch
            else None
        )

        self.cross_scale = CrossScaleTargetConditionedAttention(
            token_dim,
            num_heads=cross_scale_heads,
            num_layers=cross_scale_layers,
            dropout=cross_scale_dropout,
            context_dropout_prob=context_dropout_prob,
            neighborhood_dropout_prob=neighborhood_dropout_prob,
        )
        self.instance_project = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, feature_dim),
            nn.GELU(),
        )
        self.evidence_fuse = (
            nn.Sequential(
                nn.LayerNorm(evidence_dim * 3),
                nn.Linear(evidence_dim * 3, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if self.evidence_fusion_mode != "none"
            else None
        )
        self.fusion_mlp = (
            nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if self.evidence_fusion_mode == "concat_mlp_original"
            else None
        )
        self.evidence_to_delta = (
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if self.evidence_fusion_mode == "gated_residual"
            else None
        )
        self.evidence_gate_logit = (
            nn.Parameter(torch.tensor(float(evidence_gate_init_logit), dtype=torch.float32))
            if self.evidence_fusion_mode == "gated_residual"
            else None
        )
        self.calibration = (
            ResidualFeatureCalibration(feature_dim, hidden_features=feature_dim * 2, dropout=dropout, init_alpha=0.1)
            if self.enable_feature_calibration
            else None
        )
        self.hierarchical_head = (
            TwoStageHierarchicalOrdinalHead(
                feature_dim,
                hidden_features=feature_dim * 2,
                dropout=dropout,
                damage_decision_threshold=damage_decision_threshold,
            )
            if self.head_type == "two_stage_hierarchical"
            else None
        )
        self.flat_corn_head = (
            FlatCORNOrdinalHead(feature_dim, hidden_features=feature_dim * 2, num_classes=4, dropout=dropout)
            if self.head_type == "flat_corn"
            else None
        )
        self.share_main_corn_as_global_aux = bool(self.use_global_corn_aux and self.head_type == "flat_corn")
        self.global_corn_aux = (
            OrdinalCORNHead(feature_dim, hidden_features=feature_dim * 2, num_classes=4, dropout=dropout)
            if (self.use_global_corn_aux and self.head_type == "two_stage_hierarchical")
            else None
        )
        self.minor_no_aux_head = nn.Linear(feature_dim, 1) if self.enable_minor_boundary_aux else None
        self.minor_major_aux_head = nn.Linear(feature_dim, 1) if self.enable_minor_boundary_aux else None
        self.structural_two_stage_head = (
            StructuralTwoStageHead(
                feature_dim,
                hidden_dim=int(structural_two_stage_hidden_dim),
                dropout=float(structural_two_stage_dropout),
            )
            if self.use_structural_two_stage_head
            else None
        )
        self.conditional_review_head = (
            ConditionalReviewHead(
                token_dim=token_dim,
                feature_dim=feature_dim,
                num_classes=4,
                hidden_dim=int(conditional_review_hidden_dim),
                dropout=float(conditional_review_dropout),
                alpha_init=float(conditional_review_alpha_init),
                alpha_max=float(conditional_review_alpha_max),
                detach_base_probs=bool(conditional_review_detach_base_probs),
            )
            if self.use_conditional_review_head
            else None
        )
        self.severity_aware_scale_router = (
            SeverityAwareScaleRouter(
                token_dim,
                hidden_dim=int(scale_router_hidden_dim),
                dropout=float(scale_router_dropout),
                init_gates=scale_router_init_gates,
            )
            if self.use_severity_aware_scale_router
            else None
        )
        self.scale_aux_fusion_head = (
            ScaleWiseAuxFusionHead(
                token_dim=token_dim,
                feature_dim=feature_dim,
                num_classes=4,
                hidden_dim=int(scale_aux_hidden_dim),
                dropout=float(scale_aux_dropout),
                fusion_mode=str(scale_aux_fusion_mode),
                initial_weights=tuple(scale_aux_initial_weights),
                temperature=float(scale_aux_temperature),
                detach_base_probs_for_gate=bool(scale_aux_detach_base_probs_for_gate),
                gate_warmup_epochs=int(scale_aux_gate_warmup_epochs),
                use_prior_during_warmup=bool(scale_aux_use_prior_during_warmup),
            )
            if self.use_scale_aux_fusion_head
            else None
        )
        pixel_multiscale_count = 1 + int(self.use_context_branch) + int(self.use_neighborhood_branch)
        self.pixel_multiscale_project = (
            nn.Sequential(
                nn.Conv2d(token_dim * pixel_multiscale_count, token_dim, kernel_size=1, bias=False),
                nn.GroupNorm(max(min(token_dim // 32, 8), 1), token_dim),
                nn.GELU(),
            )
            if self.enable_pixel_ordinal_line and self.pixel_line_feature_source == "multi_scale"
            else None
        )
        self.pixel_ordinal_head = (
            build_pixel_ordinal_head(
                self.pixel_line_head_type,
                in_channels=token_dim,
                hidden_dim=int(pixel_line_hidden_dim),
                dropout=float(pixel_line_dropout),
                topk_ratio=float(pixel_line_topk_ratio),
                aggregation_mode=str(pixel_line_aggregation),
                mean_weight=float(pixel_line_mean_weight),
                topk_weight=float(pixel_line_topk_weight),
                damage_decision_threshold=float(damage_decision_threshold),
            )
            if self.enable_pixel_ordinal_line
            else None
        )

    def _effective_mix_weight(self, enabled: bool, use_for_prediction: bool, configured_weight: float) -> float:
        if not enabled:
            return 0.0
        if use_for_prediction:
            return 1.0
        return float(max(0.0, min(1.0, configured_weight)))

    def _effective_review_mix_weight(self, review_probabilities: torch.Tensor | None) -> float:
        if review_probabilities is None:
            return 0.0
        if not self.use_conditional_review_head:
            return 0.0
        if not self.conditional_review_use_for_prediction:
            return 0.0
        return float(max(0.0, min(1.0, self.conditional_review_prob_mix_weight)))

    def _effective_pixel_mix_weight(self, pixel_probabilities: torch.Tensor | None) -> float:
        if pixel_probabilities is None:
            return 0.0
        if not self.enable_pixel_ordinal_line or not self.pixel_line_use_for_final_prediction:
            return 0.0
        return float(max(0.0, min(1.0, self.pixel_line_prob_mix_weight)))

    def _scale_aux_default_outputs(
        self,
        reference_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        zero_weights = reference_feature.new_zeros(reference_feature.size(0))
        return {
            "scale_aux_tight_logits": None,
            "scale_aux_context_logits": None,
            "scale_aux_neighborhood_logits": None,
            "scale_aux_tight_class_probabilities": None,
            "scale_aux_context_class_probabilities": None,
            "scale_aux_neighborhood_class_probabilities": None,
            "scale_aux_tight_pred_labels": None,
            "scale_aux_context_pred_labels": None,
            "scale_aux_neighborhood_pred_labels": None,
            "scale_aux_fusion_logits": None,
            "scale_aux_fusion_weights": None,
            "scale_aux_fused_class_probabilities": None,
            "scale_aux_fused_pred_labels": None,
            "scale_aux_weight_base": zero_weights,
            "scale_aux_weight_tight": zero_weights,
            "scale_aux_weight_context": zero_weights,
            "scale_aux_weight_neighborhood": zero_weights,
        }

    def _pixel_line_default_outputs(self, reference_feature: torch.Tensor) -> dict[str, Any]:
        return {
            "pixel_line_enabled": False,
            "pixel_feature_source": self.pixel_line_feature_source,
            "pixel_damage_binary_logit": None,
            "pixel_severity_corn_logits": None,
            "pixel_corn_logits": None,
            "pixel_class_probabilities": None,
            "pixel_pred_labels": None,
            "pixel_instance_probabilities_mean": None,
            "pixel_instance_probabilities_topk": None,
            "pixel_instance_probabilities": None,
            "pixel_instance_pred_labels": None,
            "pixel_valid_mask": None,
        }

    def _combine_prediction_probabilities(
        self,
        corn_probabilities: torch.Tensor,
        review_probabilities: torch.Tensor | None,
        structural_probabilities: torch.Tensor | None,
        scale_router_probabilities: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_probabilities = corn_probabilities
        review_weight = self._effective_review_mix_weight(review_probabilities)
        if review_weight > 0.0 and review_probabilities is not None:
            final_probabilities = ((1.0 - review_weight) * final_probabilities) + (review_weight * review_probabilities)

        structural_weight = self._effective_mix_weight(
            structural_probabilities is not None,
            self.structural_two_stage_use_for_prediction,
            self.structural_two_stage_prob_mix_weight,
        )
        if structural_weight > 0.0 and structural_probabilities is not None:
            final_probabilities = ((1.0 - structural_weight) * final_probabilities) + (
                structural_weight * structural_probabilities
            )

        router_weight = self._effective_mix_weight(
            scale_router_probabilities is not None,
            self.scale_router_use_for_prediction,
            self.scale_router_prob_mix_weight,
        )
        if router_weight > 0.0 and scale_router_probabilities is not None:
            final_probabilities = ((1.0 - router_weight) * final_probabilities) + (
                router_weight * scale_router_probabilities
            )

        final_probabilities = final_probabilities / final_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return final_probabilities, final_probabilities.argmax(dim=1)

    def _prepare_input(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.append_mask_channel:
            return torch.cat([image, mask], dim=1)
        return image

    def _resolve_pixel_line_inputs(
        self,
        *,
        tight_out: dict[str, Any],
        context_out: dict[str, Any],
        neighborhood_out: dict[str, Any],
        batch: dict[str, torch.Tensor | list[dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_source = self.pixel_line_feature_source
        if feature_source == "tight":
            return tight_out["fused_feature"], batch["mask_tight"]
        if feature_source == "context":
            return context_out["fused_feature"], batch["mask_context"]
        if feature_source == "neighborhood":
            return neighborhood_out["fused_feature"], batch["mask_neighborhood"]
        features = [tight_out["fused_feature"]]
        if self.use_context_branch:
            features.append(F.interpolate(context_out["fused_feature"], size=tight_out["fused_feature"].shape[-2:], mode="bilinear", align_corners=False))
        if self.use_neighborhood_branch:
            features.append(
                F.interpolate(
                    neighborhood_out["fused_feature"],
                    size=tight_out["fused_feature"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fused_multiscale = torch.cat(features, dim=1)
        if self.pixel_multiscale_project is None:
            raise RuntimeError("pixel_multiscale_project is not registered, but pixel multi-scale source was requested.")
        return self.pixel_multiscale_project(fused_multiscale), batch["mask_tight"]

    def _branch_or_zero(self, branch_name: str, reference: dict[str, Any]) -> dict[str, Any]:
        batch_size = reference["tokens"].size(0)
        token_dim = reference["tokens"].size(-1)
        evidence_dim = reference["evidence_stats"].size(-1)
        zero_tokens = torch.zeros_like(reference["tokens"])
        zero_feature = torch.zeros(batch_size, 1, 1, 1, device=reference["tokens"].device, dtype=reference["tokens"].dtype)
        return {
            "fused_feature": zero_feature,
            "tokens": zero_tokens,
            "evidence_logits": None,
            "severity_map": None,
            "evidence_stats": torch.zeros(batch_size, evidence_dim, device=reference["tokens"].device, dtype=reference["tokens"].dtype),
            "evidence_raw_stats": None,
            "damage_aux_score": None,
            "severity_score": torch.zeros(batch_size, device=reference["tokens"].device, dtype=reference["tokens"].dtype),
            "change_gate": None,
            "evidence_enabled": False,
            "diagnostics": {f"{branch_name}_disabled": torch.ones(batch_size, device=reference["tokens"].device, dtype=reference["tokens"].dtype)},
        }

    def _fuse_instance_and_evidence(self, instance_feature: torch.Tensor, evidence_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.evidence_fusion_mode == "none":
            return instance_feature, instance_feature.new_zeros(instance_feature.size(0))
        if self.evidence_fusion_mode == "concat_mlp_original":
            combined_feature = self.fusion_mlp(torch.cat([instance_feature, evidence_feature], dim=1))
            return combined_feature, instance_feature.new_full((instance_feature.size(0),), 1.0)

        evidence_delta = self.evidence_to_delta(evidence_feature)
        alpha = torch.sigmoid(self.evidence_gate_logit).to(device=instance_feature.device, dtype=instance_feature.dtype)
        combined_feature = instance_feature + (alpha * evidence_delta)
        return combined_feature, instance_feature.new_full((instance_feature.size(0),), float(alpha.detach().item()))

    def forward(self, batch: dict[str, torch.Tensor | list[dict[str, Any]]]) -> dict[str, Any]:
        current_epoch = batch.get("_epoch_index")
        current_epoch_int = None if current_epoch is None else int(current_epoch)
        tight_out = self.tight_branch(
            backbone=self.backbone,
            pre_input=self._prepare_input(batch["pre_tight"], batch["mask_tight"]),
            post_input=self._prepare_input(batch["post_tight"], batch["mask_tight"]),
            mask=batch["mask_tight"],
        )
        context_out = self.context_branch(
            backbone=self.backbone,
            pre_input=self._prepare_input(batch["pre_context"], batch["mask_context"]),
            post_input=self._prepare_input(batch["post_context"], batch["mask_context"]),
            mask=batch["mask_context"],
        ) if self.context_branch is not None else self._branch_or_zero("context", tight_out)
        neighborhood_out = self.neighborhood_branch(
            backbone=self.backbone,
            pre_input=self._prepare_input(batch["pre_neighborhood"], batch["mask_neighborhood"]),
            post_input=self._prepare_input(batch["post_neighborhood"], batch["mask_neighborhood"]),
            mask=batch["mask_neighborhood"],
        ) if self.neighborhood_branch is not None else self._branch_or_zero("neighborhood", tight_out)

        context_tokens = context_out["tokens"]
        neighborhood_tokens = neighborhood_out["tokens"]
        if self.cross_scale_fusion_mode == "tight_query_context_kv":
            neighborhood_tokens = neighborhood_tokens[:, :0, :]

        if self.use_cross_scale_attention and (context_tokens.size(1) + neighborhood_tokens.size(1) > 0):
            cross_outputs = self.cross_scale(
                tight_tokens=tight_out["tokens"],
                context_tokens=context_tokens,
                neighborhood_tokens=neighborhood_tokens,
            )
            instance_token = cross_outputs["instance_feature"]
        else:
            instance_token = tight_out["tokens"].mean(dim=1)
            batch_size = instance_token.size(0)
            zero_stats = instance_token.new_zeros(batch_size)
            cross_outputs = {
                "instance_feature": instance_token,
                "cross_scale_attention_entropy": zero_stats,
                "cross_attn_to_context_mean": zero_stats,
                "cross_attn_to_neighborhood_mean": zero_stats,
            }
        instance_feature = self.instance_project(instance_token)
        evidence_scale_features = torch.cat(
            [
                tight_out["evidence_stats"],
                context_out["evidence_stats"],
                neighborhood_out["evidence_stats"],
            ],
            dim=1,
        )
        has_any_evidence = any(
            bool(branch_out["evidence_enabled"])
            for branch_out in (tight_out, context_out, neighborhood_out)
        )
        evidence_feature = (
            self.evidence_fuse(evidence_scale_features)
            if (self.evidence_fuse is not None and has_any_evidence)
            else instance_feature.new_zeros(instance_feature.shape)
        )
        if has_any_evidence:
            combined_feature, evidence_gate_alpha = self._fuse_instance_and_evidence(instance_feature, evidence_feature)
        else:
            combined_feature = instance_feature
            evidence_gate_alpha = instance_feature.new_zeros(instance_feature.size(0))
        calibrated_feature = self.calibration(combined_feature) if self.calibration is not None else combined_feature

        if self.hierarchical_head is not None:
            head_outputs = self.hierarchical_head(calibrated_feature)
            damage_binary_logit = head_outputs["damage_binary_logit"]
            severity_corn_logits = head_outputs["severity_corn_logits"]
            threshold_probabilities = head_outputs["threshold_probabilities"]
            severity_class_probabilities = head_outputs["severity_class_probabilities"]
            class_probabilities = head_outputs["class_probabilities"]
            pred_labels = head_outputs["pred_labels"]
            corn_logits = None
        else:
            flat_outputs = self.flat_corn_head(calibrated_feature)
            severity_corn_logits = None
            damage_binary_logit = None
            corn_logits = flat_outputs["corn_logits"]
            threshold_probabilities = flat_outputs["threshold_probabilities"]
            class_probabilities = flat_outputs["class_probabilities"]
            pred_labels = flat_outputs["pred_labels"]
            severity_denominator = class_probabilities[:, 1:].sum(dim=1, keepdim=True).clamp_min(1e-8)
            severity_class_probabilities = class_probabilities[:, 1:] / severity_denominator

        global_corn_aux_logits = None
        if self.global_corn_aux is not None:
            global_corn_aux_logits = self.global_corn_aux(calibrated_feature)
        elif self.share_main_corn_as_global_aux:
            global_corn_aux_logits = corn_logits
        minor_no_aux_logit = self.minor_no_aux_head(calibrated_feature).squeeze(1) if self.minor_no_aux_head is not None else None
        minor_major_aux_logit = self.minor_major_aux_head(calibrated_feature).squeeze(1) if self.minor_major_aux_head is not None else None
        structural_outputs = (
            self.structural_two_stage_head(calibrated_feature)
            if self.structural_two_stage_head is not None
            else {
                "structural_binary_logit": None,
                "low_stage_logit": None,
                "high_stage_logit": None,
                "structural_class_probabilities": None,
                "structural_pred_labels": None,
            }
        )
        conditional_review_outputs = (
            self.conditional_review_head(
                base_class_probabilities=class_probabilities,
                calibrated_feature=calibrated_feature,
                tight_tokens=tight_out["tokens"],
                context_tokens=context_out["tokens"],
                neighborhood_tokens=neighborhood_out["tokens"],
            )
            if self.conditional_review_head is not None
            else {
                "conditional_review_logits": None,
                "conditional_review_class_probabilities": None,
                "conditional_review_pred_labels": None,
                "conditional_review_low_logits": None,
                "conditional_review_high_logits": None,
                "conditional_review_low_gate": None,
                "conditional_review_high_gate": None,
                "conditional_review_alpha_low": None,
                "conditional_review_alpha_high": None,
            }
        )
        scale_router_outputs = (
            self.severity_aware_scale_router(tight_out["tokens"], context_out["tokens"])
            if self.severity_aware_scale_router is not None
            else {
                "scale_router_logits": None,
                "scale_router_probs": None,
                "scale_router_pred_labels": None,
                "scale_router_gates": None,
                "tight_logits": None,
                "context_logits": None,
            }
        )
        scale_aux_outputs = (
            self.scale_aux_fusion_head(
                base_class_probabilities=class_probabilities,
                calibrated_feature=calibrated_feature,
                tight_tokens=tight_out["tokens"],
                context_tokens=context_out["tokens"],
                neighborhood_tokens=neighborhood_out["tokens"],
                current_epoch=current_epoch_int,
            )
            if self.scale_aux_fusion_head is not None
            else self._scale_aux_default_outputs(calibrated_feature)
        )
        pixel_outputs = self._pixel_line_default_outputs(calibrated_feature)
        if self.pixel_ordinal_head is not None:
            pixel_feature_map, pixel_target_mask = self._resolve_pixel_line_inputs(
                tight_out=tight_out,
                context_out=context_out,
                neighborhood_out=neighborhood_out,
                batch=batch,
            )
            pixel_outputs = {
                "pixel_line_enabled": True,
                "pixel_feature_source": self.pixel_line_feature_source,
                **self.pixel_ordinal_head(pixel_feature_map, pixel_target_mask),
            }
        class_probabilities_for_final = class_probabilities
        if self.scale_aux_fusion_head is not None and self.scale_aux_use_for_prediction:
            scale_aux_probs = scale_aux_outputs["scale_aux_fused_class_probabilities"]
            if scale_aux_probs is not None:
                class_probabilities_for_final = scale_aux_probs
        final_class_probabilities, final_pred_labels = self._combine_prediction_probabilities(
            class_probabilities_for_final,
            conditional_review_outputs["conditional_review_class_probabilities"],
            structural_outputs["structural_class_probabilities"],
            scale_router_outputs["scale_router_probs"],
        )
        pixel_mix_weight = self._effective_pixel_mix_weight(pixel_outputs.get("pixel_instance_probabilities"))
        if pixel_mix_weight > 0.0 and pixel_outputs.get("pixel_instance_probabilities") is not None:
            final_class_probabilities = (
                (1.0 - pixel_mix_weight) * final_class_probabilities
            ) + (pixel_mix_weight * pixel_outputs["pixel_instance_probabilities"])
            final_class_probabilities = final_class_probabilities / final_class_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
            final_pred_labels = final_class_probabilities.argmax(dim=1)
        diagnostics = {}
        diagnostics.update(tight_out["diagnostics"])
        diagnostics.update(context_out["diagnostics"])
        diagnostics.update(neighborhood_out["diagnostics"])
        diagnostics.update(
            {
                "cross_scale_attention_entropy": cross_outputs["cross_scale_attention_entropy"],
                "cross_attn_to_context_mean": cross_outputs["cross_attn_to_context_mean"],
                "cross_attn_to_neighborhood_mean": cross_outputs["cross_attn_to_neighborhood_mean"],
            }
        )
        damage_aux_scores = {
            "tight": tight_out["damage_aux_score"],
            "context": context_out["damage_aux_score"],
            "neighborhood": neighborhood_out["damage_aux_score"],
        }
        change_gates = {
            "tight": tight_out["change_gate"],
            "context": context_out["change_gate"],
            "neighborhood": neighborhood_out["change_gate"],
        }
        severity_scores = {
            "tight": tight_out["severity_score"],
            "context": context_out["severity_score"],
            "neighborhood": neighborhood_out["severity_score"],
        }
        feature_stats = {
            "instance_feature_norm": instance_feature.float().norm(dim=1),
            "evidence_feature_norm": evidence_feature.float().norm(dim=1),
            "calibrated_feature_norm": calibrated_feature.float().norm(dim=1),
            "evidence_gate_alpha": evidence_gate_alpha,
            "conditional_review_low_gate_mean": (
                conditional_review_outputs["conditional_review_low_gate"]
                if conditional_review_outputs["conditional_review_low_gate"] is not None
                else instance_feature.new_zeros(instance_feature.size(0))
            ),
            "conditional_review_high_gate_mean": (
                conditional_review_outputs["conditional_review_high_gate"]
                if conditional_review_outputs["conditional_review_high_gate"] is not None
                else instance_feature.new_zeros(instance_feature.size(0))
            ),
            "conditional_review_alpha_low": (
                conditional_review_outputs["conditional_review_alpha_low"].expand(instance_feature.size(0))
                if conditional_review_outputs["conditional_review_alpha_low"] is not None
                else instance_feature.new_zeros(instance_feature.size(0))
            ),
            "conditional_review_alpha_high": (
                conditional_review_outputs["conditional_review_alpha_high"].expand(instance_feature.size(0))
                if conditional_review_outputs["conditional_review_alpha_high"] is not None
                else instance_feature.new_zeros(instance_feature.size(0))
            ),
            "scale_aux_weight_base_mean": scale_aux_outputs["scale_aux_weight_base"],
            "scale_aux_weight_tight_mean": scale_aux_outputs["scale_aux_weight_tight"],
            "scale_aux_weight_context_mean": scale_aux_outputs["scale_aux_weight_context"],
            "scale_aux_weight_neighborhood_mean": scale_aux_outputs["scale_aux_weight_neighborhood"],
            "scale_aux_weight_entropy": (
                -(
                    scale_aux_outputs["scale_aux_fusion_weights"]
                    * torch.log(scale_aux_outputs["scale_aux_fusion_weights"].clamp_min(1e-8))
                ).sum(dim=1)
                if scale_aux_outputs["scale_aux_fusion_weights"] is not None
                else instance_feature.new_zeros(instance_feature.size(0))
            ),
            "scale_aux_weight_prior_kl": (
                (
                    scale_aux_outputs["scale_aux_fusion_weights"]
                    * (
                        torch.log(scale_aux_outputs["scale_aux_fusion_weights"].clamp_min(1e-8))
                        - torch.log(
                            self.scale_aux_fusion_head.initial_weight_prior.to(
                                device=instance_feature.device,
                                dtype=scale_aux_outputs["scale_aux_fusion_weights"].dtype,
                            ).clamp_min(1e-8)
                        ).unsqueeze(0)
                    )
                ).sum(dim=1)
                if self.scale_aux_fusion_head is not None and scale_aux_outputs["scale_aux_fusion_weights"] is not None
                else instance_feature.new_zeros(instance_feature.size(0))
            ),
            "pixel_valid_area_ratio": (
                pixel_outputs["pixel_valid_mask"].float().mean(dim=(1, 2, 3))
                if pixel_outputs.get("pixel_valid_mask") is not None
                else instance_feature.new_zeros(instance_feature.size(0))
            ),
        }
        evidence_stats = {
            "tight": tight_out["evidence_raw_stats"],
            "context": context_out["evidence_raw_stats"],
            "neighborhood": neighborhood_out["evidence_raw_stats"],
        }
        evidence_enabled = {
            "tight": bool(tight_out["evidence_enabled"]),
            "context": bool(context_out["evidence_enabled"]),
            "neighborhood": bool(neighborhood_out["evidence_enabled"]),
        }
        return {
            "damage_binary_logit": damage_binary_logit,
            "severity_corn_logits": severity_corn_logits,
            "corn_logits": corn_logits,
            "global_corn_aux_logits": global_corn_aux_logits,
            "minor_no_aux_logit": minor_no_aux_logit,
            "minor_major_aux_logit": minor_major_aux_logit,
            "threshold_probabilities": threshold_probabilities,
            "severity_class_probabilities": severity_class_probabilities,
            "class_probabilities": class_probabilities,
            "pred_labels": pred_labels,
            "pred_label": pred_labels,
            "structural_binary_logit": structural_outputs["structural_binary_logit"],
            "low_stage_logit": structural_outputs["low_stage_logit"],
            "high_stage_logit": structural_outputs["high_stage_logit"],
            "structural_class_probabilities": structural_outputs["structural_class_probabilities"],
            "structural_pred_labels": structural_outputs["structural_pred_labels"],
            "conditional_review_logits": conditional_review_outputs["conditional_review_logits"],
            "conditional_review_class_probabilities": conditional_review_outputs["conditional_review_class_probabilities"],
            "conditional_review_pred_labels": conditional_review_outputs["conditional_review_pred_labels"],
            "conditional_review_low_logits": conditional_review_outputs["conditional_review_low_logits"],
            "conditional_review_high_logits": conditional_review_outputs["conditional_review_high_logits"],
            "conditional_review_low_gate": conditional_review_outputs["conditional_review_low_gate"],
            "conditional_review_high_gate": conditional_review_outputs["conditional_review_high_gate"],
            "conditional_review_alpha_low": conditional_review_outputs["conditional_review_alpha_low"],
            "conditional_review_alpha_high": conditional_review_outputs["conditional_review_alpha_high"],
            "scale_router_logits": scale_router_outputs["scale_router_logits"],
            "scale_router_probs": scale_router_outputs["scale_router_probs"],
            "scale_router_pred_labels": scale_router_outputs["scale_router_pred_labels"],
            "scale_router_gates": scale_router_outputs["scale_router_gates"],
            "scale_router_tight_logits": scale_router_outputs["tight_logits"],
            "scale_router_context_logits": scale_router_outputs["context_logits"],
            "scale_aux_tight_logits": scale_aux_outputs["scale_aux_tight_logits"],
            "scale_aux_context_logits": scale_aux_outputs["scale_aux_context_logits"],
            "scale_aux_neighborhood_logits": scale_aux_outputs["scale_aux_neighborhood_logits"],
            "scale_aux_tight_class_probabilities": scale_aux_outputs["scale_aux_tight_class_probabilities"],
            "scale_aux_context_class_probabilities": scale_aux_outputs["scale_aux_context_class_probabilities"],
            "scale_aux_neighborhood_class_probabilities": scale_aux_outputs["scale_aux_neighborhood_class_probabilities"],
            "scale_aux_tight_pred_labels": scale_aux_outputs["scale_aux_tight_pred_labels"],
            "scale_aux_context_pred_labels": scale_aux_outputs["scale_aux_context_pred_labels"],
            "scale_aux_neighborhood_pred_labels": scale_aux_outputs["scale_aux_neighborhood_pred_labels"],
            "scale_aux_fusion_logits": scale_aux_outputs["scale_aux_fusion_logits"],
            "scale_aux_fusion_weights": scale_aux_outputs["scale_aux_fusion_weights"],
            "scale_aux_fused_class_probabilities": scale_aux_outputs["scale_aux_fused_class_probabilities"],
            "scale_aux_fused_pred_labels": scale_aux_outputs["scale_aux_fused_pred_labels"],
            "scale_aux_weight_base": scale_aux_outputs["scale_aux_weight_base"],
            "scale_aux_weight_tight": scale_aux_outputs["scale_aux_weight_tight"],
            "scale_aux_weight_context": scale_aux_outputs["scale_aux_weight_context"],
            "scale_aux_weight_neighborhood": scale_aux_outputs["scale_aux_weight_neighborhood"],
            "pixel_line_enabled": pixel_outputs["pixel_line_enabled"],
            "pixel_feature_source": pixel_outputs["pixel_feature_source"],
            "pixel_damage_binary_logit": pixel_outputs["pixel_damage_binary_logit"],
            "pixel_severity_corn_logits": pixel_outputs["pixel_severity_corn_logits"],
            "pixel_corn_logits": pixel_outputs["pixel_corn_logits"],
            "pixel_class_probabilities": pixel_outputs["pixel_class_probabilities"],
            "pixel_pred_labels": pixel_outputs["pixel_pred_labels"],
            "pixel_instance_probabilities_mean": pixel_outputs["pixel_instance_probabilities_mean"],
            "pixel_instance_probabilities_topk": pixel_outputs["pixel_instance_probabilities_topk"],
            "pixel_instance_probabilities": pixel_outputs["pixel_instance_probabilities"],
            "pixel_instance_pred_labels": pixel_outputs["pixel_instance_pred_labels"],
            "pixel_valid_mask": pixel_outputs["pixel_valid_mask"],
            "final_class_probabilities": final_class_probabilities,
            "final_pred_labels": final_pred_labels,
            "final_pred_label": final_pred_labels,
            "instance_feature": instance_feature,
            "evidence_feature": evidence_feature,
            "calibrated_feature": calibrated_feature,
            "evidence_logits_tight": tight_out["evidence_logits"],
            "evidence_logits_context": context_out["evidence_logits"],
            "evidence_logits_neighborhood": neighborhood_out["evidence_logits"],
            "severity_map_tight": tight_out["severity_map"],
            "severity_map_context": context_out["severity_map"],
            "severity_map_neighborhood": neighborhood_out["severity_map"],
            "evidence_stats": evidence_stats,
            "evidence_enabled": evidence_enabled,
            "damage_aux_scores": damage_aux_scores,
            "change_gates": change_gates,
            "severity_score": severity_scores,
            "diagnostics": diagnostics,
            "feature_stats": feature_stats,
        }
