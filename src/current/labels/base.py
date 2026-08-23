"""风险标签插入点：RiskLabeler 抽象基类 + 多标签合并入口。

一个 labeler 消费 LabelContext（财务/行情等 interim 数据），产出以
(symbol, year) 为主键、携带一个或多个标签列的 DataFrame。generate_labels 会把
config 中启用的所有 labeler 结果按主键外连接合并，写入 interim/labels.parquet。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional

import numpy as np
import pandas as pd

from src.current.config import CONFIG
from src.current.registry import LABELERS


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


def _load_context() -> LabelContext:
    financial = pd.read_parquet(CONFIG.financial_interim)
    market = None
    if CONFIG.market_interim.exists():
        market = pd.read_parquet(CONFIG.market_interim)
    events = None
    if CONFIG.events_interim.exists():
        events = pd.read_parquet(CONFIG.events_interim)
    return LabelContext(financial=financial, market=market, events=events)


def generate_labels(active: Optional[List[str]] = None) -> pd.DataFrame:
    active = active or CONFIG.labels.active_labelers
    ctx = _load_context()

    frames: List[pd.DataFrame] = []
    for name in active:
        labeler: RiskLabeler = LABELERS.create(name)
        df = labeler.generate(ctx)
        if df is None or df.empty:
            print(f"[labels] {name}: 无输出（可能为未实现的插入点），跳过")
            continue
        df["symbol"] = df["symbol"].astype(str)
        df["year"] = df["year"].astype(int)
        frames.append(df)
        print(f"[labels] {name}: {len(df)} 行，列={[c for c in df.columns if c not in ('symbol','year')]}")

    if not frames:
        raise RuntimeError("没有任何 labeler 产出标签，无法继续。请检查 active_labelers。")

    merged = reduce(lambda a, b: pd.merge(a, b, on=["symbol", "year"], how="outer"), frames)

    # ---- 方案D：混合标签 ----
    # 若同时存在 KMV 标签(default_probability)与事件概率(event_probability)，
    # 则 default_probability = max(KMV, 事件概率)：ST/*ST/失败退市只上调风险、不下调。
    if "event_probability" in merged.columns and CONFIG.label_column in merged.columns:
        base = pd.to_numeric(merged[CONFIG.label_column], errors="coerce")
        ev = pd.to_numeric(merged["event_probability"], errors="coerce")
        mask = base.notna() & ev.notna() & (ev > 0)
        if mask.any():
            from src.current.labels.kmv import _risk_rating
            merged.loc[mask, CONFIG.label_column] = np.maximum(base[mask], ev[mask])
            if "risk_rating" in merged.columns:
                prb = pd.to_numeric(merged[CONFIG.label_column], errors="coerce")
                merged["risk_rating"] = [ _risk_rating(float(x)) if pd.notna(x) else None
                                          for x in prb ]
            merged["label_source"] = "kmv"
            merged.loc[mask, "label_source"] = "mixed"
            n_mixed = int(mask.sum())
            n_up = int((base[mask] < ev[mask]).sum())
        else:
            merged["label_source"] = "kmv"
            n_mixed, n_up = 0, 0
        merged.loc[merged[CONFIG.label_column].notna() & merged["label_source"].isna(),
                   "label_source"] = "kmv"
        # 事件概率已并入 default_probability，删除中间列避免混淆
        merged = merged.drop(columns=["event_probability"])
        print(f"[labels] 混合标签：{n_mixed} 行含事件，其中 {n_up} 行走高概率（max(KMV,事件)）")

    merged.to_parquet(CONFIG.labels_interim, index=False)
    print(f"[labels] 合并后 {len(merged)} 行 -> {CONFIG.labels_interim}")
    return merged
