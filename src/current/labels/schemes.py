"""可配置的标签方案（LabelScheme 注册处，与时序模型 TEMPORAL_ENCODERS 同构）。

默认 ``kmv`` 复刻基线：简化版 KMV 违约概率。另提供 ``hybrid``（方案D）：把
ST/*ST/失败退市事件软修正到 KMV 标签上（default_probability = max(KMV, 事件概率)，
事件只上调风险、不下调）。新增方案：继承 LabelScheme 并 @LABEL_SCHEMES.register。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.current.config import CONFIG
from src.current.labels.base import LabelContext, LabelScheme
from src.current.labels.kmv import KMVLabeler, _risk_rating
from src.current.labels.st import STLabeler
from src.current.registry import LABEL_SCHEMES


@LABEL_SCHEMES.register("kmv")
class KmvScheme(LabelScheme):
    """基线：简化版 KMV 标签（不含任何事件混合）。"""

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        return KMVLabeler().generate(ctx)


@LABEL_SCHEMES.register("hybrid")
class HybridScheme(LabelScheme):
    """方案D：KMV 与 ST/退市事件混合标签。

    default_probability = max(KMV, 事件概率)；新增列 st_level / delisted / label_source。
    """

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        kmv = KMVLabeler().generate(ctx)
        st = STLabeler().generate(ctx)

        if st is None or st.empty or "event_probability" not in st.columns:
            print("[hybrid] 无事件数据（interim/events.parquet），回退纯 KMV。")
            kmv["label_source"] = "kmv"
            return kmv

        kmv = kmv.copy()
        st = st.copy()
        kmv["symbol"] = kmv["symbol"].astype(str)
        st["symbol"] = st["symbol"].astype(str)
        kmv["year"] = kmv["year"].astype(int)
        st["year"] = st["year"].astype(int)

        merged = pd.merge(kmv, st, on=["symbol", "year"], how="outer")

        base = pd.to_numeric(merged[CONFIG.label_column], errors="coerce")
        ev = pd.to_numeric(merged["event_probability"], errors="coerce")
        mask = base.notna() & ev.notna() & (ev > 0)

        if mask.any():
            merged.loc[mask, CONFIG.label_column] = np.maximum(base[mask], ev[mask])
            if "risk_rating" in merged.columns:
                prb = pd.to_numeric(merged[CONFIG.label_column], errors="coerce")
                merged["risk_rating"] = [
                    _risk_rating(float(x)) if pd.notna(x) else None for x in prb
                ]
            n_mixed = int(mask.sum())
            n_up = int((base[mask] < ev[mask]).sum())
        else:
            n_mixed, n_up = 0, 0

        merged["label_source"] = "kmv"
        merged.loc[mask, "label_source"] = "mixed"

        # 事件概率已并入 default_probability，删除中间列避免混淆
        merged = merged.drop(columns=["event_probability"])
        print(f"[hybrid] 混合标签：{n_mixed} 行含事件，其中 {n_up} 行走高概率（max(KMV,事件)）")
        return merged