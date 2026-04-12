"""Helpers for VMamba-only run identity and backend resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


SUPPORTED_BACKENDS = {"vmamba", "fallback"}
RETIRED_BACKENDS = {"flowmamba", "auto"}


def resolve_requested_backend(model_cfg: Dict[str, Any]) -> str:
    """Resolve the supported backend for this project.

    The project now has a single formal backbone family: VMamba.
    The only alternate path is the internal fallback encoder for smoke/debug runs.
    """

    backend = str(model_cfg.get("backend", "vmamba") or "vmamba").lower()
    use_fallback_backbone = bool(model_cfg.get("use_fallback_backbone", False))

    if backend in RETIRED_BACKENDS:
        raise ValueError(
            f"model.backend={backend!r} is retired. "
            "This project only supports backend='vmamba' for formal runs "
            "or backend='fallback' for explicit debug/smoke runs."
        )

    if use_fallback_backbone:
        if backend not in {"vmamba", "fallback"}:
            raise ValueError(
                f"use_fallback_backbone=true is incompatible with model.backend={backend!r}. "
                "Use backend='fallback' for debug fallback runs."
            )
        return "fallback"

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported model.backend={backend!r}. "
            "Supported values are 'vmamba' and 'fallback'."
        )
    return backend


def build_run_identity(
    cfg: Dict[str, Any],
    model: Any,
    resolved_config_path: str | Path | None = None,
) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {})
    backbone_info = model.backbone.get_metadata() if hasattr(model, "backbone") else {}
    requested_backend = resolve_requested_backend(model_cfg) if model_cfg else "unknown"
    resolved_backend = str(backbone_info.get("backend_name", "unknown"))
    fallback_used = bool(backbone_info.get("fallback_used", resolved_backend == "fallback"))

    return {
        "project_mode": "vmamba_pixel_tau_corn_upper",
        "experiment_name": str(cfg.get("experiment", {}).get("name", "")),
        "backbone_name": str(backbone_info.get("backbone_name") or model_cfg.get("backbone_name", "")),
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "fallback_used": fallback_used,
        "pretrained_path": str(backbone_info.get("pretrained_path") or model_cfg.get("pretrained_path", "")),
        "pretrained_loaded": bool(backbone_info.get("pretrained_loaded", False)),
        "pretrained_required": bool(model_cfg.get("pretrained_required", False)),
        "resolved_config_path": str(resolved_config_path or cfg.get("config_path", "")),
        "input_config_path": str(cfg.get("config_path", "")),
        "tau_mode": str(model_cfg.get("tau_mode", "")),
        "instance_pool_source": str(model_cfg.get("instance_pool_source", "")),
    }
