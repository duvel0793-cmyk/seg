from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.oracle_mc_damage_model import OracleMCDamageClassifier


def main() -> None:
    model = OracleMCDamageClassifier(
        backbone="convnext_tiny",
        pretrained=False,
        head_type="corn",
    )
    model.eval()

    pre_image = torch.randn(2, 3, 224, 224)
    post_image = torch.randn(2, 3, 224, 224)
    instance_mask = torch.randint(0, 2, (2, 1, 224, 224)).float()

    with torch.no_grad():
        outputs = model(pre_image=pre_image, post_image=post_image, instance_mask=instance_mask)

    logits = outputs["logits"]
    pooled_feature = outputs["pooled_feature"]

    expected_logits_shape = (2, 3)
    expected_pooled_shape = (2, 2880)
    if tuple(logits.shape) != expected_logits_shape:
        raise RuntimeError(f"Unexpected logits shape: got {tuple(logits.shape)}, expected {expected_logits_shape}")
    if tuple(pooled_feature.shape) != expected_pooled_shape:
        raise RuntimeError(
            f"Unexpected pooled feature shape: got {tuple(pooled_feature.shape)}, expected {expected_pooled_shape}"
        )

    print(f"logits shape: {list(logits.shape)}")
    print(f"pooled_feature shape: {list(pooled_feature.shape)}")
    for name, channels in model.encoder.feature_channels.items():
        print(f"{name} channels: {channels}")


if __name__ == "__main__":
    main()
