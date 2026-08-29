"""市场风险标签（手册：商品期货 GARCH → 行业加权基准 → 企业份额调整）。

MarketRiskLabel_{i,t} = [Σ_k Weight_{j,k} × AnnualCommodityRisk_{k,t}] × [1 + α × ShareScore_{i,t}]

实现对应手册 18 步中的 STEP 7-16：
1. 行业成员（申万 L1，index_member_all，按 in_date/out_date 展开到企业-年份）
2. 商品连续价格：Tushare 主力连续合约日线（settle 优先），fut_mapping 识别换月日并
   剔除换月日收益（避免跨合约价差污染收益率）
3. GARCH(1,1) 条件波动率 → VaR(95%) → 年度均值
4. 行业基准风险 = Σ 权重 × 年度商品风险（某年缺某商品时按可用商品重归一化并注明）
5. 企业份额 = revenue / 行业收入；ShareScore = 行业-年度内百分位 rank(pct=True)
6. 标签 = 基准 × (1 + α × ShareScore)，缺失数据保留 NaN + missing_reason（手册 §九）

映射/权重表由 scripts/gen_industry_commodity_mapping.py 生成（手册 §5 规则），
存放于 repository/market/。期货/行业数据经 TushareClient 磁盘缓存，重复执行不耗额度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.current.config import CONFIG
from src.current.data.tushare_client import TushareClient
from src.current.labels.base import LabelContext, RiskLabeler
from src.current.registry import LABELERS
from src.current.transform.symbols import normalize_symbol


def _years() -> list[int]:
    return list(range(CONFIG.labels.market_start_year, CONFIG.labels.market_end_year + 1))


class _MarketRiskData:
    """市场风险标签所需数据的加载/采集（全部带磁盘缓存）。"""

    def __init__(self) -> None:
        self.client = TushareClient()
        self.dir = CONFIG.market_dir
        (self.dir / "futures").mkdir(parents=True, exist_ok=True)

    # -- 行业成员（申万 L1） --------------------------------------------
    def industry_members(self, industry_codes: list[str]) -> pd.DataFrame:
        """返回列: symbol, l1_code, l1_name, in_date, out_date。"""
        cache = self.dir / "industry_members.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        frames = []
        for code in industry_codes:
            df = self.client.query("index_member_all", l1_code=code)
            if df is None or df.empty:
                print(f"[market] 行业成员为空: {code}")
                continue
            df = df.copy()
            df["l1_code"] = code
            frames.append(df)
            print(f"[market] 行业成员 {code}: {len(df)}")
        if not frames:
            return pd.DataFrame(columns=["symbol", "l1_code", "l1_name", "in_date", "out_date"])
        allm = pd.concat(frames, ignore_index=True)
        allm["symbol"] = allm["ts_code"].map(normalize_symbol)
        allm = allm[["symbol", "l1_code", "l1_name", "in_date", "out_date"]].drop_duplicates(
            subset=["symbol", "l1_code"], keep="last")
        allm.to_parquet(cache, index=False)
        print(f"[market] 行业成员缓存: {len(allm)} 条 -> {cache.name}")
        return allm

    # -- 商品连续价格 + 换月映射 ----------------------------------------
    def commodity_price(self, code: str, suffix: str) -> pd.DataFrame:
        """主力连续日线（settle 优先），列: trade_date, price。带缓存。"""
        cache = self.dir / "futures" / f"{code}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        ts_code = f"{code}.{suffix}"
        frames = []
        for y in _years():
            df = self.client.query("fut_daily", ts_code=ts_code,
                                   start_date=f"{y}0101", end_date=f"{y}1231",
                                   fields="ts_code,trade_date,close,settle")
            if df is None or df.empty:
                print(f"[market] {ts_code} {y} 无行情")
                continue
            frames.append(df)
        if not frames:
            print(f"[market] 无期货价格: {ts_code}")
            return pd.DataFrame(columns=["trade_date", "price"])
        raw = pd.concat(frames, ignore_index=True)
        raw["price"] = raw["settle"].where(raw["settle"].notna() & (raw["settle"] > 0), raw["close"])
        raw = raw.dropna(subset=["price"])
        raw = raw[raw["price"] > 0]
        out = (raw[["trade_date", "price"]]
               .drop_duplicates(subset="trade_date", keep="last")
               .sort_values("trade_date").reset_index(drop=True))
        out.to_parquet(cache, index=False)
        print(f"[market] {ts_code} 连续价格 {len(out)} 天 -> 缓存")
        return out

    def roll_days(self, code: str, suffix: str) -> set[str]:
        """主力合约切换的交易日（这些日的对数收益跨合约，需剔除）。"""
        cache = self.dir / "futures" / f"{code}_mapping.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            ts_code = f"{code}.{suffix}"
            frames = []
            for y in _years():
                df = self.client.query("fut_mapping", ts_code=ts_code,
                                       start_date=f"{y}0101", end_date=f"{y}1231")
                if df is None or df.empty:
                    continue
                frames.append(df)
            if not frames:
                print(f"[market] 无换月映射: {ts_code}（不剔除换月日）")
                return set()
            df = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates(subset="trade_date", keep="last")
                  .sort_values("trade_date").reset_index(drop=True))
            df.to_parquet(cache, index=False)
        mapped = df["mapping_ts_code"].astype(str)
        changed = mapped != mapped.shift(1)
        return set(df.loc[changed, "trade_date"].astype(str))


def _annual_commodity_risk(data: _MarketRiskData, weights: pd.DataFrame) -> pd.DataFrame:
    """GARCH(1,1) → VaR(95%) → 年度均值。返回列: commodity_code, year, annual_commodity_risk。"""
    try:
        from arch import arch_model
    except ImportError as e:
        raise RuntimeError("需要 arch 包：pip install arch") from e

    cache = data.dir / "annual_commodity_risk.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    z = CONFIG.labels.market_var_z
    out_rows = []
    for code in sorted(weights["commodity_code"].unique()):
        suffix = weights.loc[weights["commodity_code"] == code, "suffix"].iloc[0]
        px = data.commodity_price(code, suffix)
        if len(px) < 200:
            print(f"[market] {code} 价格不足({len(px)})，该商品年度风险缺失")
            continue
        rolls = data.roll_days(code, suffix)
        px = px.copy()
        px["ret"] = np.log(px["price"] / px["price"].shift(1))
        px = px.dropna(subset=["ret"])
        n_roll = int(px["trade_date"].astype(str).isin(rolls).sum())
        px = px[~px["trade_date"].astype(str).isin(rolls)]  # 剔除换月日收益
        try:
            am = arch_model(px["ret"] * 100, vol="GARCH", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            sigma = res.conditional_volatility / 100.0
        except Exception as e:  # noqa: BLE001
            print(f"[market] {code} GARCH 拟合失败: {e}")
            continue
        daily = px[["trade_date"]].copy()
        daily["var"] = z * sigma.values
        daily["year"] = daily["trade_date"].astype(str).str[:4].astype(int)
        annual = daily.groupby("year")["var"].mean().reset_index()
        annual["commodity_code"] = code
        out_rows.append(annual[["commodity_code", "year", "var"]]
                        .rename(columns={"var": "annual_commodity_risk"}))
        print(f"[market] {code}: 收益样本 {len(px)}（剔除换月日 {n_roll}），GARCH 完成")
    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
        columns=["commodity_code", "year", "annual_commodity_risk"])
    out.to_parquet(cache, index=False)
    print(f"[market] 年度商品风险缓存: {len(out)} 条")
    return out


def _assign_industry(fin: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """给每个 (symbol, year) 指派申万一级行业（in_date/out_date 覆盖判定，取最新 in_date）。"""
    if members.empty:
        return fin.assign(industry_code=None, industry_name=None)
    m = members.copy()
    m["in_int"] = pd.to_numeric(m["in_date"], errors="coerce")
    m["out_int"] = pd.to_numeric(m["out_date"], errors="coerce")
    cross = fin[["symbol", "year"]].drop_duplicates().merge(m, on="symbol", how="left")
    yend = (cross["year"] * 10000 + 1231).astype(float)
    ybeg = (cross["year"] * 10000 + 101).astype(float)
    ok = (cross["in_int"] <= yend) & (cross["out_int"].isna() | (cross["out_int"] >= ybeg))
    cross = cross[ok]
    cross = (cross.sort_values(["symbol", "year", "in_int"])
                  .drop_duplicates(subset=["symbol", "year"], keep="last"))
    return fin.merge(cross[["symbol", "year", "l1_code", "l1_name"]]
                     .rename(columns={"l1_code": "industry_code", "l1_name": "industry_name"}),
                     on=["symbol", "year"], how="left")


@LABELERS.register("market_garch")
class MarketGarchLabeler(RiskLabeler):
    """市场风险标签器（监督目标列 market_risk_label，非违约概率语义）。"""

    output_column = "market_risk_label"

    def generate(self, ctx: LabelContext) -> pd.DataFrame:
        data = _MarketRiskData()
        mdir = CONFIG.market_dir

        wpath = mdir / "industry_commodity_weight.csv"
        if not wpath.exists():
            raise RuntimeError(f"缺少权重表 {wpath}，请先运行 scripts/gen_industry_commodity_mapping.py")
        weights = pd.read_csv(wpath, encoding="utf-8-sig")
        if "suffix" not in weights.columns:
            raise RuntimeError("权重表缺少 suffix 列，请重新运行 scripts/gen_industry_commodity_mapping.py")

        # ---- 企业样本（STEP 1：不删企业，缺失保留并注明原因）----
        fin = ctx.financial.copy()
        fin["symbol"] = fin["symbol"].map(normalize_symbol)
        fin["year"] = pd.to_numeric(fin["year"], errors="coerce")
        fin = fin.dropna(subset=["year"])
        fin["year"] = fin["year"].astype(int)
        fin["revenue"] = pd.to_numeric(fin["revenue"], errors="coerce")
        fin = fin[["symbol", "year", "revenue"]].drop_duplicates(subset=["symbol", "year"], keep="last")

        # ---- 年度商品风险（STEP 7-11，首跑需拉取约 1400 次期货数据，已缓存）----
        annual = _annual_commodity_risk(data, weights)

        # ---- 行业基准风险（STEP 12，缺商品年份按可用商品重归一化）----
        base = (weights.merge(annual, on="commodity_code", how="left")
                .dropna(subset=["annual_commodity_risk"]))
        g = base.groupby(["industry_code", "industry_name", "year"])
        wsum = g["weight"].transform("sum")
        base = base.assign(_w=base["weight"] / wsum)
        base = (base.assign(_c=base["_w"] * base["annual_commodity_risk"])
                .groupby(["industry_code", "year"], as_index=False)
                .agg(base_market_risk=("_c", "sum")))
        print(f"[market] 行业基准风险: {len(base)} 条（行业-年份）")

        # ---- 企业-行业指派（STEP 2）----
        fin = _assign_industry(fin, data.industry_members(
            sorted(weights["industry_code"].unique())))

        # ---- 市场份额与标准化（STEP 3-6）----
        valid_rev = fin["revenue"].notna() & (fin["revenue"] > 0)
        fin["invalid_revenue"] = ~valid_rev
        ind_rev = (fin[valid_rev].groupby(["industry_code", "year"], as_index=False)
                   .agg(industry_revenue=("revenue", "sum")))
        fin = fin.merge(ind_rev, on=["industry_code", "year"], how="left")
        fin["market_share"] = np.where(
            valid_rev & fin["industry_revenue"].notna() & (fin["industry_revenue"] > 0),
            fin["revenue"] / fin["industry_revenue"], np.nan)
        # 行业内年度百分位（rank pct=True）
        fin["share_score"] = fin.groupby(["industry_code", "year"])["market_share"].rank(pct=True)

        # ---- 最终标签（STEP 13-14）----
        fin = fin.merge(base, on=["industry_code", "year"], how="left")
        alpha = CONFIG.labels.market_share_alpha
        fin["adjustment_coefficient"] = 1 + alpha * fin["share_score"].fillna(0)
        fin[self.output_column] = np.where(
            fin["base_market_risk"].notna() & fin["market_share"].notna(),
            fin["base_market_risk"] * fin["adjustment_coefficient"], np.nan)

        # ---- 缺失原因（手册 §九）----
        def _reason(row) -> str:
            if row["invalid_revenue"]:
                return "Invalid revenue"
            if pd.isna(row["industry_code"]):
                return "Missing industry classification"
            if pd.isna(row["base_market_risk"]):
                return "No commodity mapping"
            if pd.isna(row[self.output_column]):
                return "Missing commodity risk"
            return ""
        fin["missing_reason"] = fin.apply(_reason, axis=1)

        out = fin[["symbol", "year", "industry_code", "industry_name", "revenue",
                   "industry_revenue", "market_share", "share_score",
                   "base_market_risk", "adjustment_coefficient",
                   self.output_column, "missing_reason"]].sort_values(["symbol", "year"])

        # ---- 质量检查（§8.6）----
        lab = out[self.output_column].dropna()
        n_neg = int((lab < 0).sum())
        n_inf = int(np.isinf(lab).sum())
        labeled = out[out[self.output_column].notna()]
        n_ind = labeled["industry_code"].nunique()
        print(f"[market] 标签行 {len(lab)} / {len(out)}；覆盖行业 {n_ind} 个；"
              f"负值 {n_neg}、无穷 {n_inf}；"
              f"范围 [{lab.min():.5f}, {lab.max():.5f}]，均值 {lab.mean():.5f}")
        reason_counts = out.loc[out["missing_reason"] != "", "missing_reason"].value_counts()
        for r, c in reason_counts.items():
            print(f"[market] 缺失原因 {r}: {c}")

        out.to_csv(mdir / "company_market_risk_label.csv", index=False, encoding="utf-8-sig")
        print(f"[market] 明细已存 {mdir / 'company_market_risk_label.csv'}")
        return out
