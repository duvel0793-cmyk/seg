from __future__ import annotations

import torch.nn as nn


def build_deep_stage(
    backend: str,
    dim: int,
    depth: int,
    drop_path_rates: list[float],
    dropout: float = 0.0,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> nn.Module:
    backend = str(backend).lower()

    if backend == "legacy":
        from models.vmamba_blocks import VMambaStage

        return VMambaStage(
            dim=dim,
            depth=depth,
            drop_path_rates=drop_path_rates,
            dropout=dropout,
        )

    if backend == "official_ss2d":
        from models.vss_official import OfficialVSSStage

        return OfficialVSSStage(
            dim=dim,
            depth=depth,
            drop_path_rates=drop_path_rates,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    raise ValueError(
        f"Unsupported deep scan backend: {backend}. "
        "Expected one of ['legacy', 'official_ss2d']."
    )