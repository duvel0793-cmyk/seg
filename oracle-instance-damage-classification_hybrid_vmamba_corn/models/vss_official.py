from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ChannelLastLayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> BHWC -> LN(C) -> BCHW
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvChannelMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def _to_seq_hw(x: torch.Tensor) -> torch.Tensor:
    # BCHW -> B(HW)C, 按行优先展开
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)


def _from_seq_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # B(HW)C -> BCHW
    b, l, c = x.shape
    if l != h * w:
        raise ValueError(f"Sequence length mismatch for HW route: got {l}, expected {h * w}.")
    return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def _to_seq_wh(x: torch.Tensor) -> torch.Tensor:
    # BCHW -> B(WH)C, 按列优先展开
    b, c, h, w = x.shape
    return x.permute(0, 3, 2, 1).contiguous().view(b, w * h, c)


def _from_seq_wh(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # B(WH)C -> BCHW
    b, l, c = x.shape
    if l != h * w:
        raise ValueError(f"Sequence length mismatch for WH route: got {l}, expected {h * w}.")
    return x.view(b, w, h, c).permute(0, 3, 2, 1).contiguous()


class OfficialSS2DOperator(nn.Module):
    """
    工程落地版 SS2D-style operator:
    - 底层序列建模核：官方 mamba_ssm.Mamba
    - 外层：2D 四方向路由
    - 输入输出：BCHW
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. Install it first with:\n"
                "pip install --no-build-isolation mamba-ssm"
            )

        self.dim = int(dim)

        # 本地空间建模支路
        self.local_dwconv = nn.Conv2d(
            self.dim,
            self.dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.dim,
            bias=False,
        )
        self.local_pwconv = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)
        self.local_act = nn.GELU()

        # 两套官方 Mamba 核：
        # 1) 行优先展开
        # 2) 列优先展开
        # 反向扫描通过 flip 后复用同一个核
        self.row_scan = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.col_scan = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )

        self.out_norm = nn.LayerNorm(self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BCHW
        b, c, h, w = x.shape

        local = self.local_pwconv(self.local_act(self.local_dwconv(x)))

        # 1) 左->右 / 右->左
        seq_hw = _to_seq_hw(x)
        lr = self.row_scan(seq_hw)
        rl = torch.flip(self.row_scan(torch.flip(seq_hw, dims=[1])), dims=[1])

        # 2) 上->下 / 下->上
        seq_wh = _to_seq_wh(x)
        tb = self.col_scan(seq_wh)
        bt = torch.flip(self.col_scan(torch.flip(seq_wh, dims=[1])), dims=[1])

        mixed = 0.25 * (
            _from_seq_hw(lr, h=h, w=w)
            + _from_seq_hw(rl, h=h, w=w)
            + _from_seq_wh(tb, h=h, w=w)
            + _from_seq_wh(bt, h=h, w=w)
        )
        mixed = mixed + local

        mixed = mixed.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
        mixed = self.out_norm(mixed)
        mixed = self.out_proj(mixed)
        mixed = self.out_drop(mixed)
        return mixed.permute(0, 3, 1, 2).contiguous()


class OfficialVSSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(dim * mlp_ratio), dim)

        self.norm1 = ChannelLastLayerNorm2d(dim)
        self.op = OfficialSS2DOperator(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = ChannelLastLayerNorm2d(dim)
        self.mlp = ConvChannelMlp(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.op(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class OfficialVSSStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: list[float],
        dropout: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if len(drop_path_rates) != int(depth):
            raise ValueError(
                f"drop_path_rates length ({len(drop_path_rates)}) must equal depth ({depth})."
            )

        self.blocks = nn.Sequential(
            *[
                OfficialVSSBlock(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    drop_path=float(drop_path_rates[idx]),
                )
                for idx in range(int(depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)