"""风险标签插入点：底层 RiskLabeler + 可配置的标签方案 LabelScheme。

两个层次的抽象（与「时序模型」完全同构）：
1. ``RiskLabeler`` —— 底层标签器（kmv / st / market_garch），产出单个或少数标签列。
   用 ``@LABELERS.register("<name>")`` 注册，供标签方案组合复用。
2. ``LabelScheme`` —— 标签方案（可配置对象）：组合底层 labeler 并做后处理，产出
   最终标签表。用 ``@LABEL_SCHEMES.register("<name>")`` 注册，在
   ``config.LabelConfig.label_scheme`` 切换（默认 "kmv" = 基线简化 KMV，
   "hybrid" = 方案D 混合标签）。

generate_labels 会按 config 选中的方案生成标签，统一写 interim/labels.parquet。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.current.config import CONFIG
from src.current.registry import LABEL_SCHEMES


@dataclass
class LabelContext:
    """标签生成所需的输入数据（按需取用）。"""
    financial: pd.DataFrame            # interim/financial
    market: Optional[pd.DataFrame] = None  # interim/market
    events: Optional[pd.DataFrame] = None  # interim/events（方案D ST/退市事件）


class RiskLabeler(ABC):
    #: 输出的主标签列名（用于日志/校验）
    output_column: str = "label"

    @abstractmethod
    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        """返回含 [symbol, year, <标签列...>] 的 DataFrame。"""
        raise NotImplementedError


class LabelScheme(ABC):
    """标签方案插入点：组合底层 labeler + 后处理，产出最终标签表。

    约定：返回含 [symbol, year, <标签列...>] 的 DataFrame，其中必须包含
    ``CONFIG.label_column``（default_probability）。类似时序模型
    ``TemporalEncoder``，新增方案继承本类并 ``@LABEL_SCHEMES.register("<name>")``。
    """

    @abstractmethod
    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        raise NotImplementedError


def _load_context() -> LabelContext:
    financial = pd.read_parquet(CONFIG.financial_interim)
    market = None
    if CONFIG.market_interim.exists():
        market = pd.read_parquet(CONFIG.market_interim)
    events = None
    if CONFIG.events_interim.exists():
        events = pd.read_parquet(CONFIG.events_interim)
    return LabelContext(financial=financial, market=market, events=events)


def generate_labels(scheme: Optional[str] = None) -> pd.DataFrame:
    """按 config.LabelConfig.label_scheme 生成最终标签并落盘 interim/labels.parquet。

    Args:
        scheme: 标签方案注册名（None 则用 config 默认）。
    """
    name = scheme or CONFIG.labels.label_scheme
    ctx = _load_context()

    try:
        label_scheme: LabelScheme = LABEL_SCHEMES.create(name)
    except KeyError:
        raise KeyError(
            f"[labels] 未注册的标签方案: {name!r}。已注册: {sorted(LABEL_SCHEMES.keys())}"
        ) from None

    merged = label_scheme.generate(ctx)
    if merged is None or merged.empty:
        raise RuntimeError(f"[labels] 方案 {name!r} 无输出，无法继续。")

    merged = merged.copy()
    merged["symbol"] = merged["symbol"].astype(str)
    merged["year"] = merged["year"].astype(int)

    merged.to_parquet(CONFIG.labels_interim, index=False)
    print(f"[labels] 方案 {name!r}：{len(merged)} 行 -> {CONFIG.labels_interim}")
    return merged