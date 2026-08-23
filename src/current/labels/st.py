"""方案D：ST/退市事件标签器。

消费 interim/events.parquet（EventCollector 产出），输出以 (symbol, year) 为主键的
事件列：st_level（0/1/2）、delisted（0/1）、event_probability。

generate_labels 会把它与 KMV 标签外连接合并，并把 default_probability 做
max(KMV, event_probability) 的软修正（事件只上调风险、不下调），形成“混合标签”。
"""
from __future__ import annotations

import pandas as pd

from src.current.config import CONFIG
from src.current.labels.base import LabelContext, RiskLabeler
from src.current.registry import LABELERS


@LABELERS.register("st")
class STLabeler(RiskLabeler):
    output_column = "event_probability"

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        events = ctx.events
        if events is None or events.empty:
            print("[st] 无事件数据（interim/events.parquet），跳过。请先运行 events 采集。")
            return pd.DataFrame(columns=["symbol", "year", self.output_column,
                                         "st_level", "delisted"])

        df = events.copy()
        df["symbol"] = df["symbol"].map(str)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)

        # 只保留真实事件行（year>0，即非占位）
        df = df[df["year"] > 0]

        cols = ["symbol", "year", "st_level", "delisted", "event_probability"]
        for c in cols:
            if c not in df.columns:
                df[c] = 0 if c in ("st_level", "delisted", "event_probability") else None
        df = df[cols].copy()
        df = df.drop_duplicates(subset=["symbol", "year"], keep="last")
        df["st_level"] = pd.to_numeric(df["st_level"], errors="coerce").fillna(0).astype(int)
        df["delisted"] = pd.to_numeric(df["delisted"], errors="coerce").fillna(0).astype(int)
        df["event_probability"] = pd.to_numeric(df["event_probability"], errors="coerce").fillna(0.0)

        n_event = (df["event_probability"] > 0).sum()
        print(f"[st] {len(df)} 行事件标签，其中 {n_event} 行有上调概率 -> 混入 KMV")
        return df