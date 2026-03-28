import copy

import torch
import torch.nn as nn
from thop import clever_format, profile
from thop.vision.basic_hooks import count_convNd


CUSTOM_OPS = {"Conv2d": count_convNd}


def _purge_thop_attrs(model: nn.Module):
    for module in model.modules():
        for attr in ("total_ops", "total_params"):
            if hasattr(module, attr):
                delattr(module, attr)


def param_calculate(model, batch_size, size, logger=None, channels=(3, 3)):
    device = next(model.parameters()).device
    input1 = torch.randn(batch_size, channels[0], size, size, device=device)
    input2 = torch.randn(batch_size, channels[1], size, size, device=device)

    proxy = copy.deepcopy(model).eval()
    _purge_thop_attrs(proxy)
    flops, params = profile(proxy, inputs=(input1, input2), custom_ops=CUSTOM_OPS)
    flops, params = clever_format([flops, params], "%.3f")
    total_params = sum(param.numel() for param in model.parameters()) / 1e6

    message = f"flops: {flops}, params: {params}, Total params: {total_params:.3f} M"
    print(message)
    if logger is not None:
        logger.info(message)
