"""可替换的时序编码器实现（时间建模层）。

默认 ``gated_conv`` 复刻 legacy 的时间门控卷积；另提供 ``gru`` / ``lstm`` /
``transformer`` 作为真实可用的替代，便于做消融实验。全部满足 (N,T,C)->(N,T,C)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.current.models.base import TemporalEncoder
from src.current.registry import TEMPORAL_ENCODERS


@TEMPORAL_ENCODERS.register("gated_conv")
class GatedConvEncoder(TemporalEncoder):
    """门控时间卷积：Conv1d 后按通道对半拆成 gate/filter，sigmoid(gate)*tanh(filter)。"""

    def __init__(self, dim: int, dropout: float = 0.3, kernel_size: int = 3) -> None:
        super().__init__(dim, dropout)
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)              # (N, C, T)
        h = self.conv(h).transpose(1, 2)   # (N, T, 2C)
        gate, filt = torch.chunk(h, 2, dim=-1)
        out = torch.sigmoid(gate) * torch.tanh(filt)
        return F.dropout(out, p=self.dropout, training=self.training)


@TEMPORAL_ENCODERS.register("gru")
class GRUEncoder(TemporalEncoder):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__(dim, dropout)
        self.rnn = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return F.dropout(out, p=self.dropout, training=self.training)


@TEMPORAL_ENCODERS.register("lstm")
class LSTMEncoder(TemporalEncoder):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__(dim, dropout)
        self.rnn = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return F.dropout(out, p=self.dropout, training=self.training)


@TEMPORAL_ENCODERS.register("transformer")
class TransformerEncoder(TemporalEncoder):
    def __init__(self, dim: int, dropout: float = 0.3, nhead: int = 4) -> None:
        super().__init__(dim, dropout)
        # nhead 需整除 dim；dim 不被整除时回退到 1 头
        if dim % nhead != 0:
            nhead = 1
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
