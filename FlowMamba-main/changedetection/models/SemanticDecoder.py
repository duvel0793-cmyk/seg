import torch.nn as nn

from classification.models.vmamba import Permute, VSSBlock
from changedetection.models.blocks import ResBlock, upsample_add


def _make_semantic_block(hidden_dim, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
    return nn.Sequential(
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


class SemanticDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        hidden_dim = 128

        self.top_block = nn.Sequential(
            nn.Conv2d(encoder_dims[-1], hidden_dim, kernel_size=1),
            _make_semantic_block(
                hidden_dim,
                channel_first,
                norm_layer,
                ssm_act_layer,
                mlp_act_layer,
                **kwargs,
            ),
        )

        self.transition_layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for in_channels in reversed(encoder_dims[:-1])
        )

        self.smooth_layers = nn.ModuleList(ResBlock(hidden_dim) for _ in range(4))
        self.semantic_blocks = nn.ModuleList(
            _make_semantic_block(
                hidden_dim,
                channel_first,
                norm_layer,
                ssm_act_layer,
                mlp_act_layer,
                **kwargs,
            )
            for _ in range(3)
        )

    def forward(self, features):
        feat1, feat2, feat3, feat4 = features
        pyramid = [feat3, feat2, feat1]

        decoded = self.top_block(feat4)
        for level, feat in enumerate(pyramid):
            decoded = upsample_add(decoded, self.transition_layers[level](feat))
            decoded = self.smooth_layers[level](decoded)
            decoded = self.semantic_blocks[level](decoded)

        decoded = self.smooth_layers[-1](decoded)
        return decoded
