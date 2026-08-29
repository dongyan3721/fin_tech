"""三张核心表的规范定义与轻量校验（面向图神经网络 + 服务可视化查询）。

- nodes：一行 = 某公司 × 某年 的财务节点特征。
- edges：一行 = 一条有向供应链关系（source -> target），带年份与权重。
- labels：一行 = 某公司 × 某年 的风险标签（可含多列，来自不同 labeler）。
"""
from __future__ import annotations

from typing import List

import pandas as pd

from src.current.config import CONFIG

# 节点表：主键 + 财务特征
NODE_KEYS: List[str] = ["symbol", "year"]
NODE_FEATURE_COLS: List[str] = list(CONFIG.feature_columns)
NODE_COLUMNS: List[str] = NODE_KEYS + NODE_FEATURE_COLS

# 边表
EDGE_COLUMNS: List[str] = ["source", "target", "weight", "relationship", "proportion", "year"]

# 标签表：主键 + 至少一个标签列（P0 为 default_probability）
LABEL_KEYS: List[str] = ["symbol", "year"]
LABEL_CORE_COL: str = CONFIG.label_column  # default_probability


def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}；实际列: {list(df.columns)}")


def validate_nodes(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, NODE_KEYS + NODE_FEATURE_COLS, "nodes")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["year"] = out["year"].astype(int)
    return out


def validate_edges(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, EDGE_COLUMNS, "edges")
    out = df.copy()
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    out["year"] = out["year"].astype(int)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0)
    return out


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, LABEL_KEYS + [LABEL_CORE_COL], "labels")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["year"] = out["year"].astype(int)
    out[LABEL_CORE_COL] = pd.to_numeric(out[LABEL_CORE_COL], errors="coerce")
    return out
