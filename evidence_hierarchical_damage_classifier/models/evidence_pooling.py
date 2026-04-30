from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_mask(mask: torch.Tensor, spatial_size: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return F.interpolate(mask.float(), size=spatial_size, mode="nearest").to(dtype=dtype)


class EvidencePooling(nn.Module):
    def __init__(self, *, topk_ratio: float, threshold: float, out_dim: int) -> None:
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.threshold = float(threshold)
        self.stats_dim = 18
        self.norm = nn.LayerNorm(self.stats_dim)
        self.project = nn.Sequential(
            nn.Linear(self.stats_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def _topk_mean(self, values: torch.Tensor, k: int) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_tensor(0.0)
        k = max(1, min(k, values.numel()))
        return torch.topk(values, k=k, dim=0).values.mean()

    def forward(self, *, evidence_logits: torch.Tensor, severity_map: torch.Tensor, target_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = _resize_mask(target_mask, evidence_logits.shape[-2:], evidence_logits.dtype)
        batch_size = evidence_logits.size(0)
        evidence_prob = evidence_logits.softmax(dim=1)
        severity_value = torch.sigmoid(severity_map)
        stats = []
        damaged_area_ratio = []
        high_damage_ratio = []
        target_area_ratio = []
        for batch_index in range(batch_size):
            valid = mask[batch_index, 0] > 0.5
            total_pixels = int(valid.sum().item())
            flat_prob = evidence_prob[batch_index].permute(1, 2, 0)[valid]
            flat_severity = severity_value[batch_index, 0][valid]
            if total_pixels <= 0:
                class_mean = evidence_prob.new_zeros(4)
                class_max = evidence_prob.new_zeros(4)
                class_topk = evidence_prob.new_zeros(4)
                sev_mean = evidence_prob.new_tensor([0.0])
                sev_max = evidence_prob.new_tensor([0.0])
                sev_topk = evidence_prob.new_tensor([0.0])
                damaged_area_ratio.append(evidence_prob.new_tensor(0.0))
                high_damage_ratio.append(evidence_prob.new_tensor(0.0))
                target_area_ratio.append(mask.new_tensor(0.0))
            else:
                k = max(1, int(round(total_pixels * self.topk_ratio)))
                class_mean = flat_prob.mean(dim=0)
                class_max = flat_prob.max(dim=0).values
                class_topk = torch.stack([self._topk_mean(flat_prob[:, idx], k) for idx in range(4)], dim=0)
                sev_mean = flat_severity.mean().unsqueeze(0)
                sev_max = flat_severity.max().unsqueeze(0)
                sev_topk = self._topk_mean(flat_severity, k).unsqueeze(0)
                damaged_mask = ((flat_prob[:, 1] + flat_prob[:, 2] + flat_prob[:, 3]) > self.threshold).float()
                high_mask = ((flat_prob[:, 2] + flat_prob[:, 3]) > self.threshold).float()
                damaged_area_ratio.append(damaged_mask.mean())
                high_damage_ratio.append(high_mask.mean())
                target_area_ratio.append(valid.float().mean())
            vector = torch.cat(
                [
                    class_mean,
                    class_max,
                    class_topk,
                    sev_mean,
                    sev_max,
                    sev_topk,
                    damaged_area_ratio[-1].unsqueeze(0),
                    high_damage_ratio[-1].unsqueeze(0),
                    target_area_ratio[-1].unsqueeze(0),
                ],
                dim=0,
            )
            stats.append(vector)
        stats_tensor = torch.stack(stats, dim=0)
        projected = self.project(self.norm(stats_tensor))
        return {
            "raw": stats_tensor,
            "projected": projected,
            "damaged_area_ratio": torch.stack(damaged_area_ratio, dim=0),
            "high_damage_ratio": torch.stack(high_damage_ratio, dim=0),
            "target_area_ratio": torch.stack(target_area_ratio, dim=0),
        }
