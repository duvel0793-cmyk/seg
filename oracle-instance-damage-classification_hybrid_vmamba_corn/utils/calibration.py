from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from utils.losses import CORNLoss


def apply_corn_calibration(
    logits: torch.Tensor,
    calibration_payload: dict[str, Any] | None,
) -> torch.Tensor:
    if calibration_payload is None:
        return logits
    temperature = float(calibration_payload.get("temperature", 1.0))
    bias = calibration_payload.get("bias", [0.0 for _ in range(logits.size(1))])
    bias_tensor = torch.tensor(bias, device=logits.device, dtype=logits.dtype)
    return (logits / max(temperature, 1e-6)) + bias_tensor.unsqueeze(0)


def fit_corn_threshold_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    calibration_type: str = "temperature_plus_bias",
    max_iter: int = 200,
    lr: float = 0.01,
) -> dict[str, Any]:
    calibration_type = str(calibration_type)
    if logits.ndim != 2:
        raise ValueError(f"Expected CORN logits with shape [N, T], got {tuple(logits.shape)}.")

    work_logits = logits.detach().float()
    work_labels = labels.detach().long().to(device=work_logits.device)
    log_temperature = torch.nn.Parameter(work_logits.new_tensor(0.0))
    bias = torch.nn.Parameter(torch.zeros(work_logits.size(1), device=work_logits.device, dtype=work_logits.dtype))
    parameters = [log_temperature]
    if calibration_type in {"temperature_plus_bias", "bias_only"}:
        parameters.append(bias)

    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    for _ in range(int(max_iter)):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature)
        if calibration_type == "temperature_only":
            calibrated_logits = work_logits / temperature
        elif calibration_type == "bias_only":
            calibrated_logits = work_logits + bias.unsqueeze(0)
        else:
            calibrated_logits = (work_logits / temperature) + bias.unsqueeze(0)
        class_probabilities = CORNLoss.logits_to_class_probabilities(calibrated_logits).clamp_min(1e-8)
        loss = F.nll_loss(class_probabilities.log(), work_labels)
        loss.backward()
        optimizer.step()

    final_temperature = float(torch.exp(log_temperature.detach()).cpu().item())
    final_bias = bias.detach().cpu().tolist()
    payload = {
        "type": calibration_type,
        "temperature": final_temperature,
        "bias": [float(value) for value in final_bias],
        "max_iter": int(max_iter),
        "lr": float(lr),
    }
    return payload
