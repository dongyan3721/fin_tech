"""KMV 信用风险标签。

两种模式：
1. 标准 KMV（迭代求解）：通过 Black-Scholes 方程组迭代求解资产价值 A 和资产波动率 σ_A
   - E = A·N(d₁) - DPT·e^(-rT)·N(d₂)
   - σ_E = (A/E)·N(d₁)·σ_A
   - DPT = 短期负债 + 0.5 × 长期负债
   - DD = (A - DPT) / (A · σ_A)
   - EDF = Φ(-DD)

2. 简化版（非迭代）：
   - A ≈ market_cap + total_liab
   - DPT = total_liab × ratio
   - σ_A = asset_volatility（股权波动率）

默认使用标准 KMV（迭代求解）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve

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


def _kmv_equations(vars: np.ndarray, E: float, sigma_E: float,
                   DPT: float, r: float, T: float) -> list[float]:
    """标准 KMV 方程组：求解资产价值 A 和资产波动率 σ_A。

    方程 1: E = A·N(d₁) - DPT·e^(-rT)·N(d₂)  （Black-Scholes 股权定价）
    方程 2: σ_E = (A/E)·N(d₁)·σ_A            （波动率关系）
    """
    A, sigma_A = vars
    if A <= 0 or sigma_A <= 0 or DPT <= 0:
        return [1e10, 1e10]

    d1 = (np.log(A / DPT) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)

    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)

    eq1 = A * N_d1 - DPT * np.exp(-r * T) * N_d2 - E
    eq2 = sigma_E - (A / E) * N_d1 * sigma_A

    return [eq1, eq2]


def _solve_kmv_iterative(E: float, sigma_E: float, DPT: float,
                         r: float, T: float,
                         max_iter: int = 100) -> tuple[float, float]:
    """迭代求解标准 KMV 的资产价值 A 和资产波动率 σ_A。

    Returns:
        (A, sigma_A) 资产价值和资产波动率
    """
    # 初始猜测：简化版公式
    A_init = E + DPT
    sigma_A_init = sigma_E * E / A_init if A_init > 0 else sigma_E

    try:
        solution = fsolve(
            _kmv_equations,
            [A_init, sigma_A_init],
            args=(E, sigma_E, DPT, r, T),
            full_output=False,
            maxfev=max_iter * 10,
        )
        A_sol, sigma_A_sol = solution

        # 验证解的合理性
        if A_sol > 0 and sigma_A_sol > 0 and A_sol > DPT * 0.5:
            return float(A_sol), float(sigma_A_sol)
    except Exception:
        pass

    # 迭代失败，回退到简化版
    return float(A_init), float(sigma_A_init)


def _calculate_dpt(total_cur_liab: float, total_liab: float) -> float:
    """计算违约点 DPT = 短期负债 + 0.5 × 长期负债。

    由于 Tushare 不提供短期/长期负债明细，使用近似：
    - 短期负债 ≈ 流动负债 (total_cur_liab)
    - 长期负债 ≈ 总负债 - 流动负债 (total_liab - total_cur_liab)

    DPT = total_cur_liab + 0.5 × (total_liab - total_cur_liab)
        = 0.5 × total_cur_liab + 0.5 × total_liab
    """
    if pd.isna(total_cur_liab) or pd.isna(total_liab):
        return np.nan
    return 0.5 * total_cur_liab + 0.5 * total_liab


@LABELERS.register("kmv")
class KMVLabeler(RiskLabeler):
    output_column = "default_probability"

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        fin = ctx.financial.copy()
        fin["symbol"] = fin["symbol"].map(normalize_symbol)
        fin["year"] = pd.to_numeric(fin["year"], errors="coerce")
        fin = fin.dropna(subset=["year"])
        fin["year"] = fin["year"].astype(int)

        # 需要 total_liab 和流动负债（用于计算 DPT）
        # 优先使用 total_cur_liab，回退到 current_liab
        required_cols = ["symbol", "year", "total_liab"]
        if "total_cur_liab" in fin.columns:
            required_cols.append("total_cur_liab")
        elif "current_liab" in fin.columns:
            required_cols.append("current_liab")
        fin = fin[required_cols].dropna(subset=["total_liab"])

        if ctx.market is None or ctx.market.empty:
            raise RuntimeError("KMV 需要行情数据 (market interim)，但未找到。请先运行 market 采集。")
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

        # 配置参数
        use_iterative = CONFIG.labels.kmv_use_iterative
        r = CONFIG.labels.risk_free_rate
        T = CONFIG.labels.kmv_time_horizon
        ratio = CONFIG.labels.default_point_ratio
        vol_floor = CONFIG.labels.min_asset_volatility

        E = df["market_cap"].astype(float)
        sigma_E = df["asset_volatility"].astype(float).clip(lower=vol_floor)

        # 获取流动负债
        if "total_cur_liab" in df.columns:
            cur_liab = df["total_cur_liab"].astype(float)
        elif "current_liab" in df.columns:
            cur_liab = df["current_liab"].astype(float)
        else:
            cur_liab = None

        # 计算违约点 DPT
        if cur_liab is not None and not cur_liab.isna().all():
            # 标准 DPT = 短期负债 + 0.5 × 长期负债
            # 近似：短期负债 ≈ 流动负债，长期负债 ≈ 总负债 - 流动负债
            dpt_values = 0.5 * cur_liab + 0.5 * df["total_liab"].astype(float)
        else:
            # 回退到简化版 DPT
            dpt_values = df["total_liab"].astype(float) * ratio

        DPT = dpt_values.values

        # 计算资产价值 A 和资产波动率 σ_A
        if use_iterative:
            # 标准 KMV：迭代求解
            asset_values = []
            asset_volatilities = []
            for i in range(len(df)):
                A, sigma_A = _solve_kmv_iterative(E.iloc[i], sigma_E.iloc[i], DPT[i], r, T)
                asset_values.append(A)
                asset_volatilities.append(sigma_A)
            asset_value = np.array(asset_values)
            asset_vol = np.array(asset_volatilities)
        else:
            # 简化版
            asset_value = E.values + df["total_liab"].astype(float).values
            asset_vol = sigma_E.values

        # 计算违约距离 DD 和违约概率 EDF
        dd = (asset_value - DPT) / (asset_value * asset_vol)
        edf = stats.norm.cdf(-dd)

        out = pd.DataFrame({
            "symbol": df["symbol"].values,
            "year": df["year"].values,
            "distance_to_default": dd,
            "default_probability": edf,
            "asset_value": asset_value,
            "default_point": DPT,
        })
        out["risk_rating"] = [_risk_rating(float(x)) for x in edf]

        method = "迭代求解" if use_iterative else "简化版"
        print(f"[kmv] {len(out)} 行，{method}，平均违约概率={edf.mean():.6f}")
        return out
