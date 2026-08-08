"""TGC 网络：时间建模（可插拔时序编码器）+ 空间图卷积（GCNConv）。

结构对齐 legacy TGCN + SpatioTemporalBlock：
  input_proj -> 时序编码器(+残差+LN) -> 每个时间步 GCNConv(+残差+LN)
  -> 时间维平均池化 -> MLP -> sigmoid
batch 的第 0 维既是样本数也是图节点数（edge_index 指向 batch 行号）。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.current.config import CONFIG, ModelConfig
from src.current.registry import TEMPORAL_ENCODERS


class _SimpleSpatial(nn.Module):
    """无图/边太少时的退回层：等价于 Linear+ReLU，不使用邻接信息。"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_weight=None):
        x = F.relu(self.linear(x))
        return F.dropout(x, p=self.dropout, training=self.training)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim: int, cfg: ModelConfig, use_gcn: bool) -> None:
        super().__init__()
        self.temporal = TEMPORAL_ENCODERS.create(cfg.temporal_encoder, dim=dim, dropout=cfg.dropout)
        self.use_gcn = use_gcn
        self.spatial = GCNConv(dim, dim) if use_gcn else _SimpleSpatial(dim, dim, cfg.dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = cfg.dropout

    def forward(self, x, edge_index=None, edge_weight=None):
        residual = x
        x = self.temporal(x)
        x = self.norm1(x + residual)

        _, seq_len, _ = x.shape
        outs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            if self.use_gcn:
                x_t = F.relu(self.spatial(x_t, edge_index, edge_weight))
                x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            else:
                x_t = self.spatial(x_t)
            outs.append(x_t.unsqueeze(1))
        x = torch.cat(outs, dim=1)
        return self.norm2(x + residual)


class TGCModel(nn.Module):
    def __init__(self, input_dim: int, cfg: Optional[ModelConfig] = None,
                 use_gcn: bool = True, final_activation: str = "sigmoid") -> None:
        super().__init__()
        cfg = cfg or CONFIG.model
        h = cfg.hidden_dim
        self.input_proj = nn.Linear(input_dim, h)
        self.block = SpatioTemporalBlock(h, cfg, use_gcn)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(h, 32), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(32, 1),
        )
        # "sigmoid"：输出概率(0,1)；"identity"：输出实数，用于 logit 空间回归
        self.final_activation = final_activation

    def forward(self, x, edge_index=None, edge_weight=None):
        x = self.input_proj(x)
        x = self.block(x, edge_index, edge_weight)
        x = x.transpose(1, 2)             # (N, C, T)
        x = self.pool(x).squeeze(-1)       # (N, C)
        out = self.head(x)
        if self.final_activation == "sigmoid":
            return torch.sigmoid(out)
        return out
