from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONVNEXTV2_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "convnextv2"
DEFAULT_CONVNEXTV2_MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"

INPUT_MODE_SPECS = {
    "rgb": {
        "append_mask_channel": False,
        "branch_in_channels": 3,
    },
    "rgbm": {
        "append_mask_channel": True,
        "branch_in_channels": 4,
    },
}


def resolve_input_mode(input_mode: str) -> dict[str, int | bool]:
    normalized = str(input_mode).lower()
    if normalized not in INPUT_MODE_SPECS:
        raise ValueError(f"Unsupported input_mode='{input_mode}'.")
    return dict(INPUT_MODE_SPECS[normalized])


class ConvNeXtV2Backbone(nn.Module):
    """Shared lightweight backbone that exposes the last two high-level maps."""

    def __init__(
        self,
        backbone_name: str = DEFAULT_CONVNEXTV2_MODEL,
        *,
        in_channels: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.in_channels = int(in_channels)
        self.pretrained = bool(pretrained)
        self.net = self._build_backbone()
        feature_info = self.net.feature_info.get_dicts()
        self.feature_channels = {
            "c4": int(feature_info[-2]["num_chs"]),
            "c5": int(feature_info[-1]["num_chs"]),
        }

    def _resolve_local_weight_path(self) -> Path:
        model_name = self.backbone_name
        if model_name.startswith("hf_hub:timm/"):
            model_name = model_name.removeprefix("hf_hub:timm/")
        return LOCAL_CONVNEXTV2_WEIGHTS_DIR / f"{model_name}.safetensors"

    @staticmethod
    def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prefixes = ("model.", "module.", "backbone.", "encoder.", "net.")
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key[len(prefix) :]
                        changed = True
            normalized[normalized_key] = value
        return normalized

    @staticmethod
    def _remap_convnextv2_key(key: str) -> str:
        remapped = key
        if remapped.startswith("stem."):
            remapped = remapped.replace("stem.", "stem_", 1)
        for stage_index in range(4):
            remapped = remapped.replace(f"stages.{stage_index}.", f"stages_{stage_index}.")
        return remapped

    @staticmethod
    def _adapt_input_conv_weight(weight: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        target_in_channels = int(target_shape[1])
        source_in_channels = int(weight.shape[1])
        if source_in_channels == target_in_channels:
            return weight
        if source_in_channels == 3 and target_in_channels == 4:
            extra_channel = weight.mean(dim=1, keepdim=True)
            return torch.cat([weight, extra_channel], dim=1)
        if source_in_channels == 4 and target_in_channels == 3:
            return weight[:, :3, :, :]
        raise ValueError(
            f"Unable to adapt input conv weight from in_channels={source_in_channels} "
            f"to in_channels={target_in_channels}."
        )

    def _load_local_pretrained_weights(self, model: nn.Module) -> None:
        weight_path = self._resolve_local_weight_path()
        if not weight_path.exists():
            raise FileNotFoundError(f"Expected local ConvNeXtV2 weights at '{weight_path}'.")

        state_dict = load_safetensors(str(weight_path))
        state_dict = self._strip_known_prefixes(state_dict)
        state_dict = {
            self._remap_convnextv2_key(key): value
            for key, value in state_dict.items()
        }
        model_state = model.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key not in model_state:
                continue
            target_tensor = model_state[key]
            candidate = value
            if candidate.ndim == 4 and target_tensor.ndim == 4 and candidate.shape != target_tensor.shape:
                if candidate.shape[0] == target_tensor.shape[0] and candidate.shape[2:] == target_tensor.shape[2:]:
                    candidate = self._adapt_input_conv_weight(candidate, target_tensor.shape)
            if candidate.shape == target_tensor.shape:
                filtered_state[key] = candidate
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        missing_keys = [key for key in missing_keys if not key.startswith(("head.", "classifier."))]
        unexpected_keys = [key for key in unexpected_keys if not key.startswith(("head.", "classifier."))]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Loaded local weights from '{weight_path}' but state_dict was incompatible. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )

    def _build_backbone(self) -> nn.Module:
        if not timm.models.is_model(self.backbone_name):
            raise RuntimeError(
                f"Backbone '{self.backbone_name}' is not available in timm {timm.__version__}. "
                "This project requires a newer timm release with ConvNeXtV2 support. "
                "Upgrade timm, for example: `pip install -U timm`."
            )
        model = timm.create_model(
            self.backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(2, 3),
            in_chans=self.in_channels,
        )
        if self.pretrained:
            self._load_local_pretrained_weights(model)
        return model

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c4, c5 = self.net(x)
        return {"c4": c4, "c5": c5}
