from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch


_PREFIXES = ("module.", "model.", "backbone.", "encoder.")


def ensure_pretrained_checkpoint(
    pretrained_path: str | Path,
    pretrained_url: str,
    auto_download: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(pretrained_path).expanduser().resolve()
    report = {
        "path": str(checkpoint_path),
        "url": str(pretrained_url),
        "exists": checkpoint_path.is_file(),
        "download_attempted": False,
        "download_succeeded": False,
        "message": "",
    }
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        report["exists"] = True
        report["message"] = "Checkpoint already exists."
        return report

    if not auto_download or not pretrained_url:
        report["message"] = "Auto download disabled or URL missing."
        return report

    report["download_attempted"] = True
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(pretrained_url, timeout=60) as response, tmp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        tmp_path.replace(checkpoint_path)
        report["exists"] = checkpoint_path.is_file()
        report["download_succeeded"] = True
        report["message"] = "Checkpoint downloaded successfully."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        report["message"] = f"Download failed: {exc}"
    return report


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "backbone"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def _clean_key(key: str) -> str:
    cleaned = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                changed = True
    return cleaned


def load_convnext_pretrained(
    model: torch.nn.Module,
    pretrained_path: str | Path,
    *,
    use_4ch_stem: bool = False,
    stem_extra_init: str = "mean",
) -> dict[str, Any]:
    checkpoint_path = Path(pretrained_path).expanduser().resolve()
    report = {
        "path": str(checkpoint_path),
        "load_attempted": True,
        "load_succeeded": False,
        "loaded_key_count": 0,
        "skipped_key_count": 0,
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "skipped_keys_preview": [],
        "message": "",
    }
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        report["message"] = "Checkpoint missing; using random initialization."
        return report

    state_dict = _extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    current_state = model.state_dict()
    loadable_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for raw_key, value in state_dict.items():
        key = _clean_key(raw_key)
        if key.startswith("head.") or ".head." in key:
            skipped_keys.append(f"{raw_key} -> skipped_head")
            continue
        if key not in current_state:
            skipped_keys.append(f"{raw_key} -> missing_in_model")
            continue

        current_tensor = current_state[key]
        if key == "downsample_layers.0.0.weight" and use_4ch_stem:
            if tuple(value.shape) == (current_tensor.shape[0], 3, current_tensor.shape[2], current_tensor.shape[3]):
                if stem_extra_init == "zeros":
                    extra = torch.zeros_like(value[:, :1, :, :])
                else:
                    extra = value.mean(dim=1, keepdim=True)
                value = torch.cat([value, extra], dim=1)

        if tuple(value.shape) != tuple(current_tensor.shape):
            skipped_keys.append(f"{raw_key} -> shape_mismatch{tuple(value.shape)}!={tuple(current_tensor.shape)}")
            continue
        loadable_state[key] = value

    incompatible = model.load_state_dict(loadable_state, strict=False)
    report["load_succeeded"] = len(loadable_state) > 0
    report["loaded_key_count"] = len(loadable_state)
    report["skipped_key_count"] = len(skipped_keys)
    report["missing_key_count"] = len(incompatible.missing_keys)
    report["unexpected_key_count"] = len(incompatible.unexpected_keys)
    report["skipped_keys_preview"] = skipped_keys[:20]
    report["message"] = (
        f"Loaded {len(loadable_state)} keys from ConvNeXt checkpoint."
        if loadable_state
        else "No compatible pretrained keys loaded."
    )
    return report

