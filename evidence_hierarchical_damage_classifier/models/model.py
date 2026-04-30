from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbone import ConvNeXtV2Backbone, DEFAULT_CONVNEXTV2_MODEL, resolve_input_mode
from models.classifier import FlatCORNOrdinalHead, OrdinalCORNHead, ResidualFeatureCalibration, normalize_corn_outputs
from models.cross_scale_attention import CrossScaleTargetConditionedAttention
from models.hierarchical_head import TwoStageHierarchicalOrdinalHead
from models.scale_branch import ScaleBranch


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
        enable_damage_aware_block: bool = True,
        enable_change_gate: bool = True,
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
        self.enable_damage_aware_block = bool(enable_damage_aware_block)
        self.enable_change_gate = bool(enable_change_gate)
        self.use_cross_scale_attention = bool(use_cross_scale_attention)
        self.cross_scale_fusion_mode = str(cross_scale_fusion_mode)
        self.use_hierarchical_head = bool(use_hierarchical_head)
        self.use_global_corn_aux = bool(use_global_corn_aux)
        self.enable_feature_calibration = bool(enable_feature_calibration)
        self.enable_minor_boundary_aux = bool(enable_minor_boundary_aux)
        self.evidence_fusion_mode = str(evidence_fusion_mode)
        self.head_type = str(head_type or ("two_stage_hierarchical" if self.use_hierarchical_head else "flat_corn"))
        self.evidence_scale_names = set(evidence_scales or ("tight", "context", "neighborhood"))
        if self.evidence_fusion_mode not in {"concat_mlp_original", "gated_residual", "none"}:
            raise ValueError(f"Unsupported evidence_fusion_mode='{self.evidence_fusion_mode}'")
        if self.cross_scale_fusion_mode not in {"tight_query_all_kv", "tight_query_context_kv"}:
            raise ValueError(f"Unsupported cross_scale_fusion_mode='{self.cross_scale_fusion_mode}'")
        if self.head_type not in {"two_stage_hierarchical", "flat_corn"}:
            raise ValueError(f"Unsupported head_type='{self.head_type}'")

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
            enable_damage_aware_block=self.enable_damage_aware_block,
            enable_change_gate=self.enable_change_gate,
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
                enable_damage_aware_block=self.enable_damage_aware_block,
                enable_change_gate=self.enable_change_gate,
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
                enable_damage_aware_block=self.enable_damage_aware_block,
                enable_change_gate=self.enable_change_gate,
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

    def _prepare_input(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.append_mask_channel:
            return torch.cat([image, mask], dim=1)
        return image

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
