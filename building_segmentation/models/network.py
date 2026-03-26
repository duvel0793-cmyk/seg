import torch.nn as nn
import torch.nn.functional as F

from building_segmentation.models.vmamba import LayerNorm2d

from building_segmentation.models.backbone import BackboneVSSM
from building_segmentation.models.decoder import SemanticDecoder


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


class FlowMambaBuilding(nn.Module):
    def __init__(self, pretrained, output_classes=2, **kwargs):
        super().__init__()
        self.encoder = BackboneVSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
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

        self.decoder = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_decoder_kwargs,
        )
        self.classifier = nn.Conv2d(128, output_classes, kernel_size=1)

    def forward(self, image):
        features = self.encoder(image)
        logits = self.classifier(self.decoder(features))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
