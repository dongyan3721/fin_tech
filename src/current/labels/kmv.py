"""KMV / Merton 思想的信用风险伪标签（P0）。

违约距离 DD = (资产价值 - 违约点) / (资产价值 × 资产波动率)
违约概率 EDF = Φ(-DD)

资产价值 ≈ 股权市值(market_cap) + 总负债(total_liab)（简化，未严格迭代 Merton）。
market_cap / asset_volatility 来自行情 interim；total_liab 来自财务 interim。
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from src.current.config import CONFIG
from src.current.labels.base import LabelContext, RiskLabeler
from src.current.registry import LABELERS
from src.current.transform.symbols import normalize_symbol


def _risk_rating(edf: float) -> str:
    thresholds = [(0.01, "AAA"), (0.05, "AA"), (0.1, "A"), (0.2, "BBB"),
                  (0.3, "BB"), (0.4, "B"), (0.6, "CCC")]
    for t, r in thresholds:
        if edf < t:
            return r
    return "D"


@LABELERS.register("kmv")
class KMVLabeler(RiskLabeler):
    output_column = "default_probability"

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        fin = ctx.financial.copy()
        fin["symbol"] = fin["symbol"].map(normalize_symbol)
        fin["year"] = pd.to_numeric(fin["year"], errors="coerce")
        fin = fin.dropna(subset=["year"])
        fin["year"] = fin["year"].astype(int)
        fin = fin[["symbol", "year", "total_liab"]].dropna(subset=["total_liab"])

        if ctx.market is None or ctx.market.empty:
            raise RuntimeError("KMV 需要行情数据(market interim)，但未找到。请先运行 market 采集。")
        mkt = ctx.market.copy()
        mkt["symbol"] = mkt["symbol"].map(normalize_symbol)
        mkt["year"] = pd.to_numeric(mkt["year"], errors="coerce")
        mkt = mkt.dropna(subset=["year"])
        mkt["year"] = mkt["year"].astype(int)
        mkt = mkt[["symbol", "year", "market_cap", "asset_volatility"]]

        df = pd.merge(fin, mkt, on=["symbol", "year"], how="inner")
        df = df.dropna(subset=["market_cap", "total_liab", "asset_volatility"])
        if df.empty:
            return pd.DataFrame(columns=["symbol", "year", self.output_column])

        ratio = CONFIG.labels.default_point_ratio
        vol_floor = CONFIG.labels.min_asset_volatility

        asset_value = df["market_cap"].astype(float) + df["total_liab"].astype(float)
        default_point = df["total_liab"].astype(float) * ratio
        vol = df["asset_volatility"].astype(float).clip(lower=vol_floor)
        dd = (asset_value - default_point) / (asset_value * vol)
        edf = stats.norm.cdf(-dd)

        out = pd.DataFrame({
            "symbol": df["symbol"].values,
            "year": df["year"].values,
            "distance_to_default": dd,
            "default_probability": edf,
            "asset_value": asset_value.values,
            "default_point": default_point.values,
        })
        out["risk_rating"] = [_risk_rating(float(x)) for x in edf]
        return out
