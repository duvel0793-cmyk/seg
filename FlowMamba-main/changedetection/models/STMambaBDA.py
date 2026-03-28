from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from classification.models.vmamba import LayerNorm2d
from changedetection.models.ChangeDecoder import ChangeDecoder
from changedetection.models.Mamba_backbone import Backbone_VSSM
from changedetection.models.SemanticDecoder import SemanticDecoder
from changedetection.utils_func.flow import FlowN


NORM_LAYERS = {
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "bn": nn.BatchNorm2d,
}

ACT_LAYERS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


class STMambaBDA(nn.Module):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super().__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        self.channel_first = self.encoder.channel_first

        norm_layer_name = kwargs["norm_layer"].lower()
        ssm_act_layer_name = kwargs["ssm_act_layer"].lower()
        mlp_act_layer_name = kwargs["mlp_act_layer"].lower()

        norm_layer = NORM_LAYERS[norm_layer_name]
        ssm_act_layer = ACT_LAYERS[ssm_act_layer_name]
        mlp_act_layer = ACT_LAYERS[mlp_act_layer_name]

        decoder_kwargs = dict(kwargs)
        decoder_kwargs.update(
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            channel_first=self.channel_first,
        )

        clean_decoder_kwargs = {
            key: value
            for key, value in decoder_kwargs.items()
            if key not in {"channel_first", "norm_layer", "ssm_act_layer", "mlp_act_layer"}
        }

        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_decoder_kwargs,
        )
        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_decoder_kwargs,
        )

        self.main_clf = nn.Conv2d(128, output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(128, output_building, kernel_size=1)
        self.flow_kwargs = dict(
            clean_decoder_kwargs,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            channel_first=self.channel_first,
        )

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        output_building = self.aux_clf(self.decoder_building(pre_features))
        output_damage = self.main_clf(self.decoder_damage(pre_features, post_features))

        output_building = F.interpolate(output_building, size=pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_damage = F.interpolate(output_damage, size=post_data.shape[-2:], mode="bilinear", align_corners=False)
        return output_building, output_damage


class FlowMamba(STMambaBDA):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super().__init__(output_building, output_damage, pretrained, **kwargs)
        self.criterion = nn.MSELoss(reduction="none")
        self.flow = nn.ModuleList(
            [
                FlowN(self.encoder.dims[0], 64, 64, **self.flow_kwargs),
                FlowN(self.encoder.dims[1], 32, 32, **self.flow_kwargs),
                FlowN(self.encoder.dims[2], 16, 16, **self.flow_kwargs),
                FlowN(self.encoder.dims[3], 8, 8, **self.flow_kwargs),
            ]
        )
    def masked_alignment_loss(self, ref_feat, aligned_feat, label):
        """
        ref_feat:      [B, C, H, W]
        aligned_feat:  [B, C, H, W]
        label:         [B, 1, H0, W0], building mask / ignore mask
        """
        mask = F.interpolate(label.float(), size=ref_feat.shape[-2:], mode="nearest")

        # 有效建筑区域：值为 1 的地方参与对齐
        valid_mask = (mask == 1).float()  # [B, 1, H, W]

        # MSE 不先做 reduce，后面手动按有效区域归一化
        sq_error = self.criterion(aligned_feat, ref_feat)  # [B, C, H, W]

        # 广播到通道维
        sq_error = sq_error * valid_mask

        denom = valid_mask.sum() * ref_feat.shape[1]
        denom = denom.clamp_min(1.0)

        return sq_error.sum() / denom
    
    def align(self, pre_features, post_features, label=None):
        assert len(pre_features) == 4, f"expected 4 pre features, got {len(pre_features)}"
        assert len(post_features) == 4, f"expected 4 post features, got {len(post_features)}"
        assert len(self.flow) == 4, f"expected 4 flow modules, got {len(self.flow)}"

        aligned_features = list(post_features)
        total_loss = pre_features[0].new_tensor(0.0)
        flows = []

        for index, flow_module in enumerate(self.flow):
            aligned_features[index], flow = flow_module(pre_features[index], aligned_features[index])
            flows.append(flow)

            if label is not None:
                total_loss = total_loss + self.masked_alignment_loss(
                    ref_feat=pre_features[index],
                    aligned_feat=aligned_features[index],
                    label=label,
                )

        if label is None:
            return aligned_features, flows
        return aligned_features, total_loss / len(self.flow)


    def forward(self, pre_data, post_data, label=None):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        align_output = self.align(pre_features, post_features, None if label is None else label.unsqueeze(1).detach())
        if label is None:
            post_features, _ = align_output
        else:
            post_features, align_loss = align_output

        output_building = self.aux_clf(self.decoder_building(pre_features))
        output_damage = self.main_clf(self.decoder_damage(pre_features, post_features))

        output_building = F.interpolate(output_building, size=pre_data.shape[-2:], mode="bilinear", align_corners=False)
        output_damage = F.interpolate(output_damage, size=post_data.shape[-2:], mode="bilinear", align_corners=False)

        if label is None:
            return output_building, output_damage
        return output_building, output_damage, align_loss
