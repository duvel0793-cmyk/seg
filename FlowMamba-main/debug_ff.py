import importlib.util

import torch

from changedetection.models.STMambaBDA import FlowMamba


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FlowMamba debug requires CUDA, but torch.cuda.is_available() is False in the "
            "current flowmamba environment."
        )
    return torch.device("cuda")


def _require_extension(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"Missing required CUDA extension '{module_name}' in the current flowmamba environment."
        )


def build_model(device: torch.device) -> FlowMamba:
    # The flowmamba conda env in this workspace ships with the oflex selective-scan extension.
    # Use the matching non-v0 forward path and keep the backbone in channel-last mode.
    _require_extension("selective_scan_cuda_oflex")
    model = FlowMamba(
        output_building=2,
        output_damage=5,
        pretrained=None,
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v3noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        use_checkpoint=False,
        use_ossm=False,
        norm_layer="ln",
    )
    return model.to(device).eval()


def main() -> None:
    device = _require_cuda()
    torch.manual_seed(0)

    model = build_model(device)

    x1 = torch.randn(1, 3, 256, 256, device=device)
    x2 = torch.randn(1, 3, 256, 256, device=device)
    y = torch.randint(0, 2, (1, 256, 256), device=device)

    with torch.no_grad():
        pre_feats = model.encoder(x1)
        post_feats = model.encoder(x2)

        print("device:", device)
        print("num pre feats:", len(pre_feats))
        print("num post feats:", len(post_feats))
        print("pre feat shapes:", [f.shape for f in pre_feats])
        print("post feat shapes:", [f.shape for f in post_feats])

        aligned_feats, flows = model.align(pre_feats, post_feats)
        print("num aligned feats:", len(aligned_feats))
        print("aligned feat shapes:", [f.shape for f in aligned_feats])
        print("num flows:", len(flows))
        print("flow shapes:", [f.shape for f in flows])

        out_b, out_d, align_loss = model(x1, x2, y)
        print("building out:", out_b.shape)
        print("damage out:", out_d.shape)
        print("align loss:", float(align_loss))


if __name__ == "__main__":
    main()
