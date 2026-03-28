import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False) + y


def make_cross_scan_tensor(pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = pre_feat.shape
    fused = pre_feat.new_empty(batch, channels, height, 2 * width)
    fused[:, :, :, ::2] = pre_feat
    fused[:, :, :, 1::2] = post_feat
    return fused


def make_sequential_scan_tensor(pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = pre_feat.shape
    fused = pre_feat.new_empty(batch, channels, height, 2 * width)
    fused[:, :, :, :width] = pre_feat
    fused[:, :, :, width:] = post_feat
    return fused


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)
