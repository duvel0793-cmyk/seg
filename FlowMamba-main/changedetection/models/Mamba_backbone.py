import torch
import torch.nn as nn

from classification.models.vmamba import LayerNorm2d, VSSM


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, pretrain_key="model", norm_layer="ln2d", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = norm_layer.lower() in {"bn", "ln2d"}
        self.out_indices = out_indices
        self.pretrain_key = pretrain_key

        norm_layers = {
            "ln": nn.LayerNorm,
            "ln2d": LayerNorm2d,
            "bn": nn.BatchNorm2d,
        }
        output_norm = norm_layers[norm_layer.lower()]
        for index in out_indices:
            self.add_module(f"outnorm{index}", output_norm(self.dims[index]))

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt_path=None):
        if ckpt_path is None:
            return

        key = self.pretrain_key or "model"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if checkpoint.get(key) is None:
            raise KeyError(f"Checkpoint {ckpt_path} does not contain key '{key}'")

        incompatible = self.load_state_dict(checkpoint[key], strict=False)
        print(f"Successfully load ckpt {ckpt_path}")
        print(f"totol number of incompatibleKeys: {len(incompatible)}")

    def forward(self, x):
        outputs = []
        x = self.patch_embed(x)
        for index, layer in enumerate(self.layers):
            features = layer.blocks(x)
            x = layer.downsample(features)
            if index in self.out_indices:
                norm = getattr(self, f"outnorm{index}")
                features = norm(features)
                if not self.channel_first:
                    features = features.permute(0, 3, 1, 2).contiguous()
                outputs.append(features)
        return outputs if self.out_indices else x
