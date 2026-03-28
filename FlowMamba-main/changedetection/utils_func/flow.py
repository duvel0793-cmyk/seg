import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from classification.models.vmamba import Permute, VSSBlock


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.vss_block1 = nn.Sequential(
            VSSBlock(
                hidden_dim=hidden_features,
                drop_path=0.1,
                norm_layer=kwargs["norm_layer"],
                channel_first=kwargs["channel_first"],
                ssm_d_state=kwargs["ssm_d_state"],
                ssm_ratio=kwargs["ssm_ratio"],
                ssm_dt_rank=kwargs["ssm_dt_rank"],
                ssm_act_layer=kwargs["ssm_act_layer"],
                ssm_conv=kwargs["ssm_conv"],
                ssm_conv_bias=kwargs["ssm_conv_bias"],
                ssm_drop_rate=kwargs["ssm_drop_rate"],
                ssm_init=kwargs["ssm_init"],
                forward_type=kwargs["forward_type"],
                mlp_ratio=kwargs["mlp_ratio"],
                mlp_act_layer=kwargs["mlp_act_layer"],
                mlp_drop_rate=kwargs["mlp_drop_rate"],
                gmlp=kwargs["gmlp"],
                use_checkpoint=kwargs["use_checkpoint"],
                use_ossm=kwargs["use_ossm"],
            )
        )
        self.vss_block2 = nn.Sequential(
            VSSBlock(
                hidden_dim=hidden_features,
                drop_path=0.1,
                norm_layer=kwargs["norm_layer"],
                channel_first=kwargs["channel_first"],
                ssm_d_state=kwargs["ssm_d_state"],
                ssm_ratio=kwargs["ssm_ratio"],
                ssm_dt_rank=kwargs["ssm_dt_rank"],
                ssm_act_layer=kwargs["ssm_act_layer"],
                ssm_conv=kwargs["ssm_conv"],
                ssm_conv_bias=kwargs["ssm_conv_bias"],
                ssm_drop_rate=kwargs["ssm_drop_rate"],
                ssm_init=kwargs["ssm_init"],
                forward_type=kwargs["forward_type"],
                mlp_ratio=kwargs["mlp_ratio"],
                mlp_act_layer=kwargs["mlp_act_layer"],
                mlp_drop_rate=kwargs["mlp_drop_rate"],
                gmlp=kwargs["gmlp"],
                use_checkpoint=kwargs["use_checkpoint"],
                use_ossm=kwargs["use_ossm"],
            )
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.vss_block1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.vss_block2(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + self.dropout(attn_output))


class FlowMLP(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        channel_first = kwargs["channel_first"]
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, groups=channels)
        self.vss_block = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=channels,
                drop_path=0.1,
                norm_layer=kwargs["norm_layer"],
                channel_first=channel_first,
                ssm_d_state=kwargs["ssm_d_state"],
                ssm_ratio=kwargs["ssm_ratio"],
                ssm_dt_rank=kwargs["ssm_dt_rank"],
                ssm_act_layer=kwargs["ssm_act_layer"],
                ssm_conv=kwargs["ssm_conv"],
                ssm_conv_bias=kwargs["ssm_conv_bias"],
                ssm_drop_rate=kwargs["ssm_drop_rate"],
                ssm_init=kwargs["ssm_init"],
                forward_type=kwargs["forward_type"],
                mlp_ratio=kwargs["mlp_ratio"],
                mlp_act_layer=kwargs["mlp_act_layer"],
                mlp_drop_rate=kwargs["mlp_drop_rate"],
                gmlp=kwargs["gmlp"],
                use_checkpoint=kwargs["use_checkpoint"],
                use_ossm=kwargs["use_ossm"],
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.vss_block(self.conv(x)))


class FlowN(nn.Module):
    def __init__(self, channels, _height, _width, **kwargs):
        super().__init__()
        self.flowmlp1 = FlowMLP(channels, **kwargs)
        self.flowmlp2 = FlowMLP(channels, **kwargs)
        self.flow_make = nn.Conv2d(channels * 2, 2, kernel_size=3, padding=1, bias=False)
        self.channel_align = CrossAttention(channels)
        self.mlp = Mlp(channels, channels, channels, act_layer=nn.GELU, drop=0.0, **kwargs)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(0.1)

    def align(self, x, cond):
        batch, channels, height, width = x.shape
        x = x.reshape(batch, -1, channels)
        cond = cond.reshape(batch, -1, channels)
        x = x + self.drop(self.channel_align(self.norm(x), self.norm(cond), self.norm(cond)))
        x = x.reshape(batch, channels, height, width)
        x = x + self.drop(self.mlp(self.norm(x.permute(0, 2, 3, 1)), height, width)).permute(0, 3, 1, 2)
        return x

    def forward(self, x1, x2):
        x1 = self.flowmlp1(x1)
        x2 = self.flowmlp2(x2)
        x2 = self.align(x2, x1)
        flow = self.flow_make(torch.cat([x2, x1], dim=1))
        return self.flow_warp(x2, flow), flow

    def flow_warp(self, inputs, flow):
        batch, _, out_h, out_w = inputs.shape
        norm = inputs.new_tensor([[[[out_w, out_h]]]])
        h_grid = torch.linspace(-1.0, 1.0, out_h, device=inputs.device, dtype=inputs.dtype).view(-1, 1).repeat(1, out_w)
        w_grid = torch.linspace(-1.0, 1.0, out_w, device=inputs.device, dtype=inputs.dtype).repeat(out_h, 1)
        grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        return F.grid_sample(inputs, grid, align_corners=False)
