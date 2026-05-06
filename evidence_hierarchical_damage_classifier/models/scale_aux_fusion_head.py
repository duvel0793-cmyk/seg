from __future__ import annotations

import torch
import torch.nn as nn

from models.classifier import FlatCORNOrdinalHead


class ScaleWiseAuxFusionHead(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        feature_dim: int = 768,
        num_classes: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        fusion_mode: str = "dynamic",
        initial_weights: tuple[float, float, float, float] = (0.85, 0.06, 0.06, 0.03),
        temperature: float = 1.0,
        detach_base_probs_for_gate: bool = True,
        gate_warmup_epochs: int = 0,
        use_prior_during_warmup: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.fusion_mode = str(fusion_mode)
        self.temperature = float(temperature)
        self.detach_base_probs_for_gate = bool(detach_base_probs_for_gate)
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.use_prior_during_warmup = bool(use_prior_during_warmup)

        if self.fusion_mode != "dynamic":
            raise ValueError(f"Unsupported scale aux fusion_mode='{fusion_mode}'")
        if self.num_classes != 4:
            raise ValueError("ScaleWiseAuxFusionHead currently expects num_classes=4")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if len(initial_weights) != 4:
            raise ValueError("initial_weights must contain 4 values: [base, tight, context, neighborhood]")
        if self.gate_warmup_epochs < 0:
            raise ValueError("gate_warmup_epochs must be >= 0")

        initial_weight_tensor = torch.as_tensor(initial_weights, dtype=torch.float32).clamp_min(1e-8)
        initial_weight_tensor = initial_weight_tensor / initial_weight_tensor.sum().clamp_min(1e-8)
        self.register_buffer("initial_weight_prior", initial_weight_tensor, persistent=False)

        self.tight_projector = self._build_projector()
        self.context_projector = self._build_projector()
        self.neighborhood_projector = self._build_projector()

        self.tight_scale_head = FlatCORNOrdinalHead(
            self.feature_dim,
            hidden_features=self.feature_dim * 2,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )
        self.context_scale_head = FlatCORNOrdinalHead(
            self.feature_dim,
            hidden_features=self.feature_dim * 2,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )
        self.neighborhood_scale_head = FlatCORNOrdinalHead(
            self.feature_dim,
            hidden_features=self.feature_dim * 2,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )

        gate_input_dim = self.feature_dim + (self.token_dim * 3) + self.num_classes
        self.gate_norm = nn.LayerNorm(gate_input_dim)
        self.gate_fc1 = nn.Linear(gate_input_dim, self.hidden_dim)
        self.gate_act = nn.GELU()
        self.gate_dropout = nn.Dropout(self.dropout)
        self.gate_fc2 = nn.Linear(self.hidden_dim, 4)

        with torch.no_grad():
            # Zero weights + log-prior bias makes the initial softmax match the configured prior,
            # regardless of the gate input content.
            nn.init.zeros_(self.gate_fc2.weight)
            self.gate_fc2.bias.copy_(torch.log(self.initial_weight_prior.clamp_min(1e-8)))

    def _prior_weights(self, reference: torch.Tensor) -> torch.Tensor:
        return self.initial_weight_prior.to(device=reference.device, dtype=reference.dtype).unsqueeze(0).expand(reference.size(0), -1)

    def _should_use_prior_during_warmup(self, current_epoch: int | None) -> bool:
        if not self.use_prior_during_warmup:
            return False
        if self.gate_warmup_epochs <= 0:
            return False
        if current_epoch is None:
            return False
        return int(current_epoch) < self.gate_warmup_epochs

    def _build_projector(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

    @staticmethod
    def _mean_pool_tokens(tokens: torch.Tensor | None, reference: torch.Tensor, out_dim: int) -> torch.Tensor:
        if tokens is None or tokens.ndim != 3 or tokens.size(1) == 0:
            return reference.new_zeros((reference.size(0), out_dim))
        return tokens.mean(dim=1)

    def _gate_forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        hidden = self.gate_fc1(self.gate_norm(gate_input))
        hidden = self.gate_act(hidden)
        hidden = self.gate_dropout(hidden)
        return self.gate_fc2(hidden)

    def forward(
        self,
        *,
        base_class_probabilities: torch.Tensor,
        calibrated_feature: torch.Tensor,
        tight_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        neighborhood_tokens: torch.Tensor | None,
        current_epoch: int | None = None,
    ) -> dict[str, torch.Tensor]:
        tight_summary = self._mean_pool_tokens(tight_tokens, calibrated_feature, self.token_dim)
        context_summary = self._mean_pool_tokens(context_tokens, calibrated_feature, self.token_dim)
        neighborhood_summary = self._mean_pool_tokens(neighborhood_tokens, calibrated_feature, self.token_dim)

        tight_feature = self.tight_projector(tight_summary)
        context_feature = self.context_projector(context_summary)
        neighborhood_feature = self.neighborhood_projector(neighborhood_summary)

        tight_outputs = self.tight_scale_head(tight_feature)
        context_outputs = self.context_scale_head(context_feature)
        neighborhood_outputs = self.neighborhood_scale_head(neighborhood_feature)

        base_probs_for_gate = (
            base_class_probabilities.detach()
            if self.detach_base_probs_for_gate
            else base_class_probabilities
        )
        gate_input = torch.cat(
            [
                calibrated_feature,
                tight_summary,
                context_summary,
                neighborhood_summary,
                base_probs_for_gate,
            ],
            dim=1,
        )
        if self._should_use_prior_during_warmup(current_epoch):
            scale_fusion_weights = self._prior_weights(base_class_probabilities)
            gate_logits = torch.log(scale_fusion_weights.clamp_min(1e-8))
        else:
            gate_logits = self._gate_forward(gate_input)
            scale_fusion_weights = torch.softmax(gate_logits / self.temperature, dim=1)

        fused_probabilities = (
            scale_fusion_weights[:, 0:1] * base_class_probabilities
            + scale_fusion_weights[:, 1:2] * tight_outputs["class_probabilities"]
            + scale_fusion_weights[:, 2:3] * context_outputs["class_probabilities"]
            + scale_fusion_weights[:, 3:4] * neighborhood_outputs["class_probabilities"]
        )
        fused_probabilities = fused_probabilities.clamp_min(1e-8)
        fused_probabilities = fused_probabilities / fused_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused_pred_labels = fused_probabilities.argmax(dim=1)

        return {
            "scale_aux_tight_logits": tight_outputs["corn_logits"],
            "scale_aux_context_logits": context_outputs["corn_logits"],
            "scale_aux_neighborhood_logits": neighborhood_outputs["corn_logits"],
            "scale_aux_tight_class_probabilities": tight_outputs["class_probabilities"],
            "scale_aux_context_class_probabilities": context_outputs["class_probabilities"],
            "scale_aux_neighborhood_class_probabilities": neighborhood_outputs["class_probabilities"],
            "scale_aux_tight_pred_labels": tight_outputs["pred_labels"],
            "scale_aux_context_pred_labels": context_outputs["pred_labels"],
            "scale_aux_neighborhood_pred_labels": neighborhood_outputs["pred_labels"],
            "scale_aux_fusion_logits": gate_logits,
            "scale_aux_fusion_weights": scale_fusion_weights,
            "scale_aux_fused_class_probabilities": fused_probabilities,
            "scale_aux_fused_pred_labels": fused_pred_labels,
            "scale_aux_weight_base": scale_fusion_weights[:, 0],
            "scale_aux_weight_tight": scale_fusion_weights[:, 1],
            "scale_aux_weight_context": scale_fusion_weights[:, 2],
            "scale_aux_weight_neighborhood": scale_fusion_weights[:, 3],
        }


MultiScaleAuxFusionHead = ScaleWiseAuxFusionHead
