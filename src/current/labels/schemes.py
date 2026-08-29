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
from src.current.labels.market_garch import MarketGarchLabeler
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


@LABEL_SCHEMES.register("mix")
class MixScheme(LabelScheme):
    """综合风险标签方案：KMV 违约概率 × 市场风险标签 秩归一化融合。

    composite_risk_label = w·rank(KMV) + (1-w)·rank(市场风险)   （年内百分位秩）
    仅一方可用时权重自动重归一化；训练经 --target-column composite_risk_label 启用。
    """

    output_column = "composite_risk_label"

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        kmv = KMVLabeler().generate(ctx).copy()
        market = MarketGarchLabeler().generate(ctx).copy()
        for df in (kmv, market):
            df["symbol"] = df["symbol"].astype(str)
            df["year"] = df["year"].astype(int)
        merged = pd.merge(kmv, market, on=["symbol", "year"], how="outer")

        w = CONFIG.labels.mix_kmv_weight
        per_year = merged.groupby("year")
        merged["kmv_rank"] = per_year["default_probability"].rank(pct=True)
        merged["market_rank"] = per_year["market_risk_label"].rank(pct=True)

        col = self.output_column
        merged[col] = np.nan
        both = merged["kmv_rank"].notna() & merged["market_rank"].notna()
        only_kmv = merged["kmv_rank"].notna() & merged["market_rank"].isna()
        only_mkt = merged["kmv_rank"].isna() & merged["market_rank"].notna()
        merged.loc[both, col] = w * merged.loc[both, "kmv_rank"] + (1 - w) * merged.loc[both, "market_rank"]
        merged.loc[only_kmv, col] = merged.loc[only_kmv, "kmv_rank"]
        merged.loc[only_mkt, col] = merged.loc[only_mkt, "market_rank"]

        merged["label_source"] = "none"
        merged.loc[only_kmv, "label_source"] = "kmv"
        merged.loc[only_mkt, "label_source"] = "market"
        merged.loc[both, "label_source"] = "mix"

        n_both, n_k, n_m = int(both.sum()), int(only_kmv.sum()), int(only_mkt.sum())
        lab = merged[col].dropna()
        print(f"[mix] 综合标签 {len(lab)} 行（双源 {n_both} / 仅KMV {n_k} / 仅市场 {n_m}，w={w}），"
              f"范围 [{lab.min():.4f}, {lab.max():.4f}]")
        return merged


@LABEL_SCHEMES.register("market")
class MarketScheme(LabelScheme):
    """市场风险标签方案（手册：GARCH 商品风险 → 行业加权 → 企业份额调整）。

    KMV 与市场风险并列外连接合并（两套语义独立的标签共存于 labels 表）；
    训练时通过 config.LabelConfig.target_column / train.py --target-column 选择监督目标列。
    """

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        kmv = KMVLabeler().generate(ctx)
        market = MarketGarchLabeler().generate(ctx)

        kmv = kmv.copy()
        market = market.copy()
        kmv["symbol"] = kmv["symbol"].astype(str)
        market["symbol"] = market["symbol"].astype(str)
        kmv["year"] = kmv["year"].astype(int)
        market["year"] = market["year"].astype(int)

        merged = pd.merge(kmv, market, on=["symbol", "year"], how="outer")
        merged["label_source"] = "kmv"
        merged.loc[merged["market_risk_label"].notna(), "label_source"] = "market"
        print(f"[market] 合并后 {len(merged)} 行，其中含市场风险标签 "
              f"{int(merged['market_risk_label'].notna().sum())} 行")
        return merged