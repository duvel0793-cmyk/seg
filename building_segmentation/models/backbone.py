import torch
import torch.nn as nn

from building_segmentation.models.vmamba import LayerNorm2d, VSSM


class BackboneVSSM(VSSM):
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

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and checkpoint.get(self.pretrain_key) is not None:
            state_dict = checkpoint[self.pretrain_key]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

        incompatible = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded encoder checkpoint from {ckpt_path}")
        print(f"Missing keys: {len(incompatible.missing_keys)}, unexpected keys: {len(incompatible.unexpected_keys)}")

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
