from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbone import ConvNeXtV2Backbone, DEFAULT_CONVNEXTV2_MODEL, resolve_input_mode
from models.classifier import (
    OrdinalCORNHead,
    ResidualFeatureCalibration,
    corn_logits_to_threshold_probabilities,
    decode_corn_probabilities,
)
from models.cross_scale_attention import CrossScaleTargetConditionedAttention
from models.neighborhood_graph import NeighborhoodGraphModule
from models.normalization import LayerScale
from models.scale_branch import ScaleBranch


class MultiContextDamageModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = DEFAULT_CONVNEXTV2_MODEL,
        pretrained: bool = True,
        input_mode: str = "rgbm",
        feature_dim: int = 256,
        dropout: float = 0.2,
        use_change_suppression: bool = True,
        change_block_channels: int | None = None,
        enable_pseudo_suppression: bool = True,
        fuse_change_to_tokens: bool = True,
        change_residual_scale: float = 0.2,
        change_gate_init_gamma: float = 0.1,
        gate_temperature: float = 2.0,
        gate_bias_init: float = -2.0,
        enable_damage_aux: bool = True,
        enable_severity_aux: bool = True,
        tight_token_count: int = 8,
        context_token_count: int = 8,
        neighborhood_token_count: int = 12,
        local_attention_heads: int = 4,
        local_attention_layers: int = 2,
        tight_window_size: int = 7,
        context_window_size: int = 7,
        neighborhood_window_size: int = 8,
        use_cross_scale_attention: bool = True,
        cross_scale_heads: int = 4,
        cross_scale_layers: int = 2,
        cross_scale_dropout: float = 0.1,
        context_dropout_prob: float = 0.25,
        neighborhood_dropout_prob: float = 0.25,
        enable_neighborhood_graph: bool = False,
        graph_k_neighbors: int = 6,
        graph_layers: int = 1,
        graph_hidden_dim: int = 256,
        graph_attention_heads: int = 4,
        graph_use_distance_bias: bool = True,
        use_tight_branch: bool = True,
        use_context_branch: bool = True,
        use_neighborhood_scale: bool = True,
        use_local_attention: bool = True,
        safe_multicontext_mode: bool = True,
        cross_scale_layerscale_init: float = 1.0e-3,
        cross_scale_residual_max: float = 0.3,
        neighborhood_branch_gate_init: float = -4.0,
        neighborhood_residual_scale_init: float = 0.02,
        neighborhood_residual_scale_max: float = 0.20,
        neighborhood_aux_enabled: bool = False,
        neighborhood_gate_bias_init: float = -4.0,
        neighborhood_gate_temperature: float = 3.0,
        freeze_neighborhood_strength_after_epoch: int = 7,
        neighborhood_branch_gate_max: float = 0.04,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        if not use_tight_branch:
            raise ValueError("MultiContextDamageModel requires use_tight_branch=True.")

        self.num_classes = int(num_classes)
        self.input_spec = resolve_input_mode(input_mode)
        self.append_mask_channel = bool(self.input_spec["append_mask_channel"])
        self.use_cross_scale_attention = bool(use_cross_scale_attention)
        self.enable_neighborhood_graph = bool(enable_neighborhood_graph and (not safe_multicontext_mode))
        self.graph_k_neighbors = int(graph_k_neighbors)
        self.use_context_branch = bool(use_context_branch)
        self.use_neighborhood_scale = bool(use_neighborhood_scale)
        self.safe_multicontext_mode = bool(safe_multicontext_mode)
        self.neighborhood_aux_enabled = bool(neighborhood_aux_enabled)
        self.freeze_neighborhood_strength_after_epoch = int(freeze_neighborhood_strength_after_epoch)
        self.neighborhood_branch_gate_max = float(neighborhood_branch_gate_max)
        self.runtime_epoch = 0
        self.runtime_is_train = False
        self.runtime_collect_diagnostics = True
        self.runtime_collect_feature_stats = True

        self.backbone = ConvNeXtV2Backbone(
            backbone_name=backbone_name,
            in_channels=int(self.input_spec["branch_in_channels"]),
            pretrained=pretrained,
        )
        c4_channels = int(self.backbone.feature_channels["c4"])
        c5_channels = int(self.backbone.feature_channels["c5"])

        shared_branch_kwargs = {
            "c4_channels": c4_channels,
            "c5_channels": c5_channels,
            "feature_dim": feature_dim,
            "local_attention_heads": local_attention_heads,
            "local_attention_layers": local_attention_layers,
            "dropout": dropout,
            "use_change_suppression": use_change_suppression,
            "change_block_channels": int(change_block_channels or feature_dim),
            "enable_pseudo_suppression": enable_pseudo_suppression,
            "fuse_change_to_tokens": fuse_change_to_tokens,
            "change_residual_scale": change_residual_scale,
            "change_gate_init_gamma": change_gate_init_gamma,
            "gate_temperature": gate_temperature,
            "gate_bias_init": gate_bias_init,
            "enable_damage_aux": enable_damage_aux,
            "enable_severity_aux": enable_severity_aux,
            "use_local_attention": use_local_attention,
        }
        self.scale_branches = nn.ModuleDict(
            {
                "tight": ScaleBranch(
                    scale_name="tight",
                    token_count=tight_token_count,
                    window_size=tight_window_size,
                    **shared_branch_kwargs,
                ),
                "context": ScaleBranch(
                    scale_name="context",
                    token_count=context_token_count,
                    window_size=context_window_size,
                    **shared_branch_kwargs,
                ),
                "neighborhood": ScaleBranch(
                    scale_name="neighborhood",
                    token_count=neighborhood_token_count,
                    window_size=neighborhood_window_size,
                    gate_temperature=neighborhood_gate_temperature,
                    gate_bias_init=neighborhood_gate_bias_init,
                    **{key: value for key, value in shared_branch_kwargs.items() if key not in {"gate_temperature", "gate_bias_init"}},
                ),
            }
        )
        self.cross_scale = CrossScaleTargetConditionedAttention(
            feature_dim,
            num_heads=cross_scale_heads,
            num_layers=cross_scale_layers,
            dropout=cross_scale_dropout,
            context_dropout_prob=context_dropout_prob,
            neighborhood_dropout_prob=neighborhood_dropout_prob,
            layerscale_init=cross_scale_layerscale_init,
            residual_max=cross_scale_residual_max,
        )
        self.context_disabled_token = nn.Parameter(torch.zeros(1, context_token_count, feature_dim))
        self.neighborhood_disabled_token = nn.Parameter(torch.zeros(1, neighborhood_token_count, feature_dim))
        self.neighborhood_branch_gate_logit = nn.Parameter(torch.tensor(float(neighborhood_branch_gate_init), dtype=torch.float32))
        self.neighborhood_residual_scale = LayerScale(
            feature_dim,
            init_value=neighborhood_residual_scale_init,
            max_scale=neighborhood_residual_scale_max,
            ndim=3,
        )
        self.pre_classifier_norm = nn.LayerNorm(feature_dim)
        self.graph_module = None
        if self.enable_neighborhood_graph:
            self.graph_module = NeighborhoodGraphModule(
                feature_dim,
                hidden_dim=graph_hidden_dim,
                num_heads=graph_attention_heads,
                num_layers=graph_layers,
                use_distance_bias=graph_use_distance_bias,
            )
        self.feature_calibration = ResidualFeatureCalibration(
            feature_dim,
            hidden_features=max(feature_dim * 2, 256),
            dropout=dropout,
            init_alpha=0.1 if use_change_suppression else 0.0,
        )
        self.classifier = OrdinalCORNHead(
            feature_dim,
            hidden_features=max(feature_dim * 2, 256),
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def set_runtime_context(
        self,
        *,
        epoch_index: int,
        is_train: bool,
        collect_diagnostics: bool = True,
        collect_feature_stats: bool = True,
    ) -> None:
        self.runtime_epoch = int(epoch_index)
        self.runtime_is_train = bool(is_train)
        self.runtime_collect_diagnostics = bool(collect_diagnostics)
        self.runtime_collect_feature_stats = bool(collect_feature_stats)

    def _prepare_branch_input(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.append_mask_channel:
            return torch.cat([image, mask], dim=1)
        return image

    def _build_graph_inputs(
        self,
        neighborhood_feature_map: torch.Tensor,
        meta: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = neighborhood_feature_map.size(0)
        device = neighborhood_feature_map.device
        dtype = neighborhood_feature_map.dtype
        neighbor_mask = torch.zeros(batch_size, self.graph_k_neighbors, device=device, dtype=dtype)
        relative_positions = torch.zeros(batch_size, self.graph_k_neighbors, 2, device=device, dtype=dtype)
        distances = torch.zeros(batch_size, self.graph_k_neighbors, device=device, dtype=dtype)
        grid = torch.zeros(batch_size, self.graph_k_neighbors, 1, 2, device=device, dtype=dtype)

        for batch_index, item in enumerate(meta):
            x1, y1, x2, y2 = [float(value) for value in item["neighborhood_bbox_xyxy"]]
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            center_x, center_y = [float(value) for value in item["centroid_xy"]]
            norm_x = ((center_x - x1) / max(width, 1.0)) * 2.0 - 1.0
            norm_y = ((center_y - y1) / max(height, 1.0)) * 2.0 - 1.0
            grid[batch_index, 0, 0, 0] = norm_x
            grid[batch_index, 0, 0, 1] = norm_y
            neighbor_mask[batch_index, 0] = 1.0

        sampled = torch.nn.functional.grid_sample(
            neighborhood_feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        neighbor_features = sampled.squeeze(-1).transpose(1, 2)
        neighbor_features = neighbor_features * neighbor_mask.unsqueeze(-1)
        return neighbor_features, neighbor_mask, relative_positions, distances

    def _resolve_override_flags(self, batch: dict[str, Any]) -> dict[str, bool]:
        overrides = batch.get("__forward_overrides__", {})
        return {
            "use_context_branch": bool(overrides.get("use_context_branch", True)),
            "use_neighborhood_scale": bool(overrides.get("use_neighborhood_scale", True)),
        }

    def _resolve_neighborhood_gate(self, reference: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.neighborhood_branch_gate_logit).to(device=reference.device, dtype=reference.dtype)
        gate = torch.clamp(gate, max=self.neighborhood_branch_gate_max)
        if self.runtime_is_train and self.runtime_epoch >= self.freeze_neighborhood_strength_after_epoch:
            gate = gate.detach()
        return gate

    def _make_disabled_scale_output(
        self,
        *,
        scale_name: str,
        reference_tokens: torch.Tensor,
        token_count: int,
    ) -> dict[str, Any]:
        batch_size, _, feature_dim = reference_tokens.shape
        device = reference_tokens.device
        dtype = reference_tokens.dtype
        zero_feature = torch.zeros(batch_size, feature_dim, 1, 1, device=device, dtype=dtype)
        zero_tokens = torch.zeros(batch_size, token_count, feature_dim, device=device, dtype=dtype)
        return {
            "pre_feature": zero_feature,
            "post_feature": zero_feature,
            "aligned_pre": zero_feature,
            "feat_pre_refined": None,
            "feat_post_refined": None,
            "fused_feature": zero_feature,
            "tokens": zero_tokens,
            "change_feature": None,
            "change_gate": None,
            "damage_map_logits": None,
            "severity_logit_map": None,
            "severity_score": None,
            "damage_aux_score": None,
            "mask_resized": None,
            "diagnostics": {} if not self.runtime_collect_diagnostics else {
                f"{scale_name}_local_attention_entropy": torch.zeros(batch_size, device=device, dtype=dtype),
                f"{scale_name}_token_norm": torch.zeros(batch_size, device=device, dtype=dtype),
            },
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        override_flags = self._resolve_override_flags(batch)
        use_context_branch = self.use_context_branch and override_flags["use_context_branch"]
        use_neighborhood_scale = self.use_neighborhood_scale and override_flags["use_neighborhood_scale"]

        scale_outputs: dict[str, dict[str, Any]] = {}
        scale_outputs["tight"] = self.scale_branches["tight"](
            backbone=self.backbone,
            pre_input=self._prepare_branch_input(batch["pre_tight"], batch["mask_tight"]),
            post_input=self._prepare_branch_input(batch["post_tight"], batch["mask_tight"]),
            mask=batch["mask_tight"],
            collect_diagnostics=self.runtime_collect_diagnostics,
        )
        tight_tokens = scale_outputs["tight"]["tokens"]

        if use_context_branch:
            scale_outputs["context"] = self.scale_branches["context"](
                backbone=self.backbone,
                pre_input=self._prepare_branch_input(batch["pre_context"], batch["mask_context"]),
                post_input=self._prepare_branch_input(batch["post_context"], batch["mask_context"]),
                mask=batch["mask_context"],
                collect_diagnostics=self.runtime_collect_diagnostics,
            )
        else:
            scale_outputs["context"] = self._make_disabled_scale_output(
                scale_name="context",
                reference_tokens=tight_tokens,
                token_count=self.context_disabled_token.size(1),
            )

        if use_neighborhood_scale:
            scale_outputs["neighborhood"] = self.scale_branches["neighborhood"](
                backbone=self.backbone,
                pre_input=self._prepare_branch_input(batch["pre_neighborhood"], batch["mask_neighborhood"]),
                post_input=self._prepare_branch_input(batch["post_neighborhood"], batch["mask_neighborhood"]),
                mask=batch["mask_neighborhood"],
                collect_diagnostics=self.runtime_collect_diagnostics,
            )
        else:
            scale_outputs["neighborhood"] = self._make_disabled_scale_output(
                scale_name="neighborhood",
                reference_tokens=tight_tokens,
                token_count=self.neighborhood_disabled_token.size(1),
            )

        context_tokens = scale_outputs["context"]["tokens"]
        raw_neighborhood_tokens = scale_outputs["neighborhood"]["tokens"]
        neighborhood_tokens = raw_neighborhood_tokens

        if not use_context_branch:
            context_tokens = self.context_disabled_token.expand(tight_tokens.size(0), -1, -1).to(device=tight_tokens.device, dtype=tight_tokens.dtype)

        neighborhood_branch_gate = self._resolve_neighborhood_gate(tight_tokens)
        if use_neighborhood_scale:
            neighborhood_tokens = neighborhood_tokens * neighborhood_branch_gate
        else:
            neighborhood_tokens = self.neighborhood_disabled_token.expand(tight_tokens.size(0), -1, -1).to(
                device=tight_tokens.device,
                dtype=tight_tokens.dtype,
            )

        if self.use_cross_scale_attention:
            cross_outputs = self.cross_scale(
                tight_tokens,
                context_tokens,
                neighborhood_tokens,
            )
            instance_feature = cross_outputs["instance_feature"]
        else:
            tight_summary = tight_tokens.mean(dim=1)
            cross_outputs = {
                "instance_feature": tight_summary,
                "cross_scale_tokens": tight_tokens,
                "tight_tokens": tight_tokens,
                "cross_attention_weights": torch.zeros(
                    tight_tokens.size(0),
                    tight_tokens.size(1),
                    context_tokens.size(1) + neighborhood_tokens.size(1),
                    device=tight_tokens.device,
                    dtype=tight_tokens.dtype,
                ),
                "cross_attn_to_context_mean": torch.zeros(tight_tokens.size(0), device=tight_tokens.device, dtype=tight_tokens.dtype),
                "cross_attn_to_neighborhood_mean": torch.zeros(tight_tokens.size(0), device=tight_tokens.device, dtype=tight_tokens.dtype),
                "cross_scale_attention_entropy": torch.zeros(tight_tokens.size(0), device=tight_tokens.device, dtype=tight_tokens.dtype),
                "tight_token_norm": tight_tokens.float().norm(dim=-1).mean(dim=1),
                "context_token_norm": context_tokens.float().norm(dim=-1).mean(dim=1),
                "neighborhood_token_norm": neighborhood_tokens.float().norm(dim=-1).mean(dim=1),
                "context_dropout_rate_actual": torch.zeros(tight_tokens.size(0), device=tight_tokens.device, dtype=tight_tokens.dtype),
                "neighborhood_dropout_rate_actual": torch.zeros(tight_tokens.size(0), device=tight_tokens.device, dtype=tight_tokens.dtype),
            }
            instance_feature = tight_summary

        final_instance_feature = self.pre_classifier_norm(instance_feature)
        calibrated_feature = self.feature_calibration(final_instance_feature)
        corn_logits = self.classifier(calibrated_feature)
        threshold_probabilities = corn_logits_to_threshold_probabilities(corn_logits)
        class_probabilities = decode_corn_probabilities(threshold_probabilities)
        pred_labels = class_probabilities.argmax(dim=1)

        severity_scores = [scale_outputs[name]["severity_score"] for name in ("tight", "context", "neighborhood") if scale_outputs[name]["severity_score"] is not None]
        severity_score = None if not severity_scores else torch.stack(severity_scores, dim=0).mean(dim=0)

        neighborhood_graph_feature = None
        graph_outputs: dict[str, torch.Tensor | None] = {
            "graph_gate": None,
            "graph_attention_weights": None,
            "graph_attention_entropy": None,
            "graph_valid_neighbor_count": None,
        }
        if self.enable_neighborhood_graph and self.graph_module is not None:
            neighbor_features, graph_neighbor_mask, relative_positions, distances = self._build_graph_inputs(
                scale_outputs["neighborhood"]["fused_feature"],
                batch["meta"],
            )
            graph_outputs = self.graph_module(
                final_instance_feature,
                neighbor_features,
                graph_neighbor_mask,
                relative_positions,
                distances,
            )
            neighborhood_graph_feature = graph_outputs["graph_feature"]
            final_instance_feature = final_instance_feature + graph_outputs["graph_feature"]
            calibrated_feature = self.feature_calibration(final_instance_feature)
            corn_logits = self.classifier(calibrated_feature)
            threshold_probabilities = corn_logits_to_threshold_probabilities(corn_logits)
            class_probabilities = decode_corn_probabilities(threshold_probabilities)
            pred_labels = class_probabilities.argmax(dim=1)

        feature_stats: dict[str, torch.Tensor] = {}
        if self.runtime_collect_feature_stats:
            feature_stats = {
                "tight_feature_norm": scale_outputs["tight"]["fused_feature"].float().flatten(1).norm(dim=1),
                "context_feature_norm": scale_outputs["context"]["fused_feature"].float().flatten(1).norm(dim=1),
                "neighborhood_feature_norm": scale_outputs["neighborhood"]["fused_feature"].float().flatten(1).norm(dim=1),
                "instance_feature_norm": instance_feature.float().norm(dim=1),
                "final_instance_feature_norm": final_instance_feature.float().norm(dim=1),
                "calibrated_feature_norm": calibrated_feature.float().norm(dim=1),
            }

        diagnostics: dict[str, torch.Tensor | None] = {}
        if self.runtime_collect_diagnostics:
            for scale_name in ("tight", "context", "neighborhood"):
                diagnostics.update(scale_outputs[scale_name]["diagnostics"])
            diagnostics.update(
                {
                    "cross_attn_to_context_mean": cross_outputs["cross_attn_to_context_mean"],
                    "cross_attn_to_neighborhood_mean": cross_outputs["cross_attn_to_neighborhood_mean"],
                    "cross_scale_attention_entropy": cross_outputs["cross_scale_attention_entropy"],
                    "tight_token_norm": cross_outputs["tight_token_norm"],
                    "context_token_norm": cross_outputs["context_token_norm"],
                    "neighborhood_token_norm": cross_outputs["neighborhood_token_norm"],
                    "context_dropout_rate_actual": cross_outputs["context_dropout_rate_actual"],
                    "neighborhood_dropout_rate_actual": cross_outputs["neighborhood_dropout_rate_actual"],
                    "graph_gate_mean": None if graph_outputs["graph_gate"] is None else graph_outputs["graph_gate"],
                    "graph_attention_entropy": graph_outputs["graph_attention_entropy"],
                    "graph_valid_neighbor_count": graph_outputs["graph_valid_neighbor_count"],
                    "neighborhood_branch_gate": torch.full(
                        (tight_tokens.size(0),),
                        float(neighborhood_branch_gate.item()),
                        device=tight_tokens.device,
                        dtype=tight_tokens.dtype,
                    ),
                    "neighborhood_residual_scale": self.neighborhood_residual_scale.current_scale().mean().to(
                        device=tight_tokens.device,
                        dtype=tight_tokens.dtype,
                    ).expand(tight_tokens.size(0)),
                }
            )

        damage_aux_scores = {name: scale_outputs[name]["damage_aux_score"] for name in ("tight", "context", "neighborhood")}

        return {
            "logits": corn_logits,
            "corn_logits": corn_logits,
            "probs": class_probabilities,
            "class_probabilities": class_probabilities,
            "threshold_probabilities": threshold_probabilities,
            "pred_labels": pred_labels,
            "instance_feature": instance_feature,
            "final_instance_feature": final_instance_feature,
            "calibrated_feature": calibrated_feature,
            "scale_features": {name: scale_outputs[name]["fused_feature"] for name in ("tight", "context", "neighborhood")},
            "scale_tokens": {"tight": tight_tokens, "context": context_tokens, "neighborhood": neighborhood_tokens},
            "cross_scale_tokens": cross_outputs["cross_scale_tokens"],
            "scale_outputs": scale_outputs,
            "damage_aux_logits": scale_outputs["tight"]["damage_map_logits"],
            "damage_aux_scores": damage_aux_scores,
            "severity_logit_map": scale_outputs["tight"]["severity_logit_map"],
            "severity_score": severity_score,
            "change_gate": scale_outputs["tight"]["change_gate"],
            "neighborhood_graph_feature": neighborhood_graph_feature,
            "graph_attention_weights": graph_outputs["graph_attention_weights"],
            "feature_stats": feature_stats,
            "diagnostics": diagnostics,
        }
