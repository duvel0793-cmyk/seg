"""Helpers for backend metadata, config hashing, and checkpoint compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .run_identity import resolve_requested_backend


def stable_config_hash(config: Dict[str, Any]) -> str:
    """Create a stable hash for a resolved config dictionary."""

    payload = json.dumps(_normalize_json(config), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_checkpoint_metadata(cfg: Dict[str, Any], model: Any, resolved_config_path: str | Path | None = None) -> Dict[str, Any]:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    backbone_info = model.backbone.get_metadata() if hasattr(model, "backbone") else {}
    requested_backend = resolve_requested_backend(model_cfg)
    fallback_used = bool(backbone_info.get("fallback_used", backbone_info.get("backend_name") == "fallback"))
    resolved_config = str(resolved_config_path or cfg.get("config_path", ""))

    return {
        "experiment_name": cfg.get("experiment", {}).get("name", ""),
        "config_path": resolved_config,
        "resolved_config_path": resolved_config,
        "config_hash": stable_config_hash(cfg),
        "backbone_name": str(backbone_info.get("backbone_name") or model_cfg.get("backbone_name", "")),
        "requested_backend": requested_backend,
        "backend_name": backbone_info.get("backend_name"),
        "backend_reason": backbone_info.get("backend_reason"),
        "fallback_used": fallback_used,
        "pretrained_path": backbone_info.get("pretrained_path") or model_cfg.get("pretrained_path"),
        "pretrained_loaded": backbone_info.get("pretrained_loaded", False),
        "pretrained_strict": backbone_info.get("pretrained_strict", False),
        "feature_channels": backbone_info.get("feature_channels", []),
        "feature_strides": backbone_info.get("feature_strides", []),
        "num_classes_loc": int(data_cfg["num_classes_loc"]),
        "num_classes_damage": int(data_cfg["num_classes_damage"]),
        "ordinal_dims": int(data_cfg["num_classes_damage"]) - 1,
        "tau_mode": str(model_cfg.get("tau_mode", "disabled")),
        "instance_pool_source": str(model_cfg.get("instance_pool_source", "ordinal_logits")),
    }


def compare_checkpoint_metadata(
    checkpoint_meta: Dict[str, Any] | None,
    expected_meta: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return errors and warnings describing compatibility issues."""

    if not checkpoint_meta:
        return [], ["Checkpoint metadata missing; compatibility could not be fully verified."]

    errors: List[str] = []
    warnings: List[str] = []

    def _mismatch(key: str) -> bool:
        return key in checkpoint_meta and key in expected_meta and checkpoint_meta.get(key) != expected_meta.get(key)

    hard_keys = ["backbone_name", "num_classes_loc", "num_classes_damage", "ordinal_dims"]
    warn_keys = ["pretrained_path", "pretrained_loaded", "requested_backend", "tau_mode", "instance_pool_source"]

    if _mismatch("backend_name") or _mismatch("fallback_used"):
        errors.append(
            "runtime identity mismatch: "
            f"checkpoint {checkpoint_meta.get('backend_name')!r} "
            f"(fallback_used={checkpoint_meta.get('fallback_used')!r}) vs current "
            f"{expected_meta.get('backend_name')!r} "
            f"(fallback_used={expected_meta.get('fallback_used')!r})"
        )

    for key in hard_keys:
        if _mismatch(key):
            errors.append(
                f"metadata[{key}] mismatch: checkpoint={checkpoint_meta.get(key)!r} current={expected_meta.get(key)!r}"
            )

    for key in warn_keys:
        if _mismatch(key):
            warnings.append(
                f"metadata[{key}] mismatch: checkpoint={checkpoint_meta.get(key)!r} current={expected_meta.get(key)!r}"
            )

    checkpoint_cfg_hash = checkpoint_meta.get("config_hash")
    current_cfg_hash = expected_meta.get("config_hash")
    if checkpoint_cfg_hash and current_cfg_hash and checkpoint_cfg_hash != current_cfg_hash:
        warnings.append(
            f"config_hash mismatch: checkpoint={checkpoint_cfg_hash} current={current_cfg_hash}"
        )
    checkpoint_resolved_config = checkpoint_meta.get("resolved_config_path") or checkpoint_meta.get("config_path")
    current_resolved_config = expected_meta.get("resolved_config_path") or expected_meta.get("config_path")
    if checkpoint_resolved_config and current_resolved_config and checkpoint_resolved_config != current_resolved_config:
        warnings.append(
            f"resolved_config_path mismatch: checkpoint={checkpoint_resolved_config} current={current_resolved_config}"
        )

    return errors, warnings


def summarize_backend_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "backbone_name": metadata.get("backbone_name"),
        "backend_name": metadata.get("backend_name"),
        "requested_backend": metadata.get("requested_backend"),
        "backend_reason": metadata.get("backend_reason"),
        "fallback_used": metadata.get("fallback_used"),
        "repo_path": metadata.get("repo_path"),
        "pretrained_loaded": metadata.get("pretrained_loaded"),
        "pretrained_path": metadata.get("pretrained_path"),
        "pretrained_strict": metadata.get("pretrained_strict"),
        "pretrained_reason": (metadata.get("pretrained_report") or {}).get("reason"),
        "matched_pretrained_keys": metadata.get("matched_pretrained_keys"),
        "feature_channels": metadata.get("feature_channels"),
        "feature_strides": metadata.get("feature_strides"),
    }


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return value
