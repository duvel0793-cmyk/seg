import torch
import torch.nn as nn

from classification.models.vmamba import Permute, VSSBlock
from changedetection.models.blocks import (
    ResBlock,
    make_cross_scan_tensor,
    make_sequential_scan_tensor,
    upsample_add,
)


def _make_vss_stage(
    in_channels: int,
    hidden_dim: int,
    channel_first: bool,
    norm_layer,
    ssm_act_layer,
    mlp_act_layer,
    **kwargs,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
        Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
        VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=0.1,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=kwargs["ssm_d_state"],
            ssm_ratio=kwargs["ssm_ratio"],
            ssm_dt_rank=kwargs["ssm_dt_rank"],
            ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs["ssm_conv"],
            ssm_conv_bias=kwargs["ssm_conv_bias"],
            ssm_drop_rate=kwargs["ssm_drop_rate"],
            ssm_init=kwargs["ssm_init"],
            forward_type=kwargs["forward_type"],
            mlp_ratio=kwargs["mlp_ratio"],
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=kwargs["mlp_drop_rate"],
            gmlp=kwargs["gmlp"],
            use_checkpoint=kwargs["use_checkpoint"],
            use_ossm=kwargs["use_ossm"],
        ),
        Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
    )


class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        hidden_dim = 128
        dims = list(reversed(encoder_dims))

        self.parallel_blocks = nn.ModuleList()
        self.cross_blocks = nn.ModuleList()
        self.sequential_blocks = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList(ResBlock(hidden_dim) for _ in range(len(dims) - 1))

        for in_channels in dims:
            self.parallel_blocks.append(
                _make_vss_stage(
                    in_channels * 2,
                    hidden_dim,
                    channel_first,
                    norm_layer,
                    ssm_act_layer,
                    mlp_act_layer,
                    **kwargs,
                )
            )
            self.cross_blocks.append(
                _make_vss_stage(
                    in_channels,
                    hidden_dim,
                    channel_first,
                    norm_layer,
                    ssm_act_layer,
                    mlp_act_layer,
                    **kwargs,
                )
            )
            self.sequential_blocks.append(
                _make_vss_stage(
                    in_channels,
                    hidden_dim,
                    channel_first,
                    norm_layer,
                    ssm_act_layer,
                    mlp_act_layer,
                    **kwargs,
                )
            )
            self.fuse_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim * 5, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, pre_features, post_features):
        decoded = None
        for level, (pre_feat, post_feat) in enumerate(zip(reversed(pre_features), reversed(post_features))):
            width = pre_feat.shape[-1]

            parallel = self.parallel_blocks[level](torch.cat([pre_feat, post_feat], dim=1))
            cross = self.cross_blocks[level](make_cross_scan_tensor(pre_feat, post_feat))
            sequential = self.sequential_blocks[level](make_sequential_scan_tensor(pre_feat, post_feat))

            fused = self.fuse_layers[level](
                torch.cat(
                    [
                        parallel,
                        cross[:, :, :, ::2],
                        cross[:, :, :, 1::2],
                        sequential[:, :, :, :width],
                        sequential[:, :, :, width:],
                    ],
                    dim=1,
                )
            )

            if decoded is not None:
                fused = upsample_add(decoded, fused)
                fused = self.smooth_layers[level - 1](fused)

            decoded = fused

        return decoded
