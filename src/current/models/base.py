"""时序模型插入点：TemporalEncoder 抽象基类。

约定：输入 (N, T, C)，输出 (N, T, C)（时间维长度不变，通道数不变），
使得它可以在 TGC 中作为“时间建模层”被任意替换（门控卷积/GRU/LSTM/Transformer）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module, ABC):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.dim = dim
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, T, C) -> (N, T, C)
        raise NotImplementedError
