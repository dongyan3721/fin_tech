"""商品期货与行业成员数据采集（市场风险标签的原始数据层，手册 STEP 2/6-7）。

- 期货：按权重表 ``repository/market/industry_commodity_weight.csv`` 里的商品清单，
  逐年拉取主力连续日线（``fut_daily``，settle 优先）与换月映射（``fut_mapping``），
  落 ``repository/market/futures/{CODE}.parquet`` / ``{CODE}_mapping.parquet``；
- 行业：申万一级行业成员（``index_member_all``，按 l1_code）→
  ``repository/market/industry_members.parquet``。

全部经 TushareClient 磁盘缓存，重复执行只补缺失。
``cli collect`` 与市场风险标签器（labels/market_garch.py）共用同一批缓存文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.current.config import CONFIG
from src.current.data.tushare_client import TushareClient
from src.current.transform.symbols import normalize_symbol


def market_years() -> List[int]:
    return list(range(CONFIG.labels.market_start_year, CONFIG.labels.market_end_year + 1))


def _futures_dir() -> Path:
    d = CONFIG.market_dir / "futures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _price_cache(code: str) -> Path:
    return _futures_dir() / f"{code}.parquet"


def _mapping_cache(code: str) -> Path:
    return _futures_dir() / f"{code}_mapping.parquet"


def _members_cache() -> Path:
    return CONFIG.market_dir / "industry_members.parquet"


# -- 单商品取数（标签器与采集器共用） ------------------------------------

def fetch_commodity_price(client: TushareClient, code: str, suffix: str) -> pd.DataFrame:
    """主力连续日线（settle 优先），列: trade_date, price。带缓存。"""
    cache = _price_cache(code)
    if cache.exists():
        return pd.read_parquet(cache)
    ts_code = f"{code}.{suffix}"
    frames = []
    for y in market_years():
        df = client.query("fut_daily", ts_code=ts_code,
                          start_date=f"{y}0101", end_date=f"{y}1231",
                          fields="ts_code,trade_date,close,settle")
        if df is None or df.empty:
            print(f"[futures] {ts_code} {y} 无行情")
            continue
        frames.append(df)
    if not frames:
        print(f"[futures] 无期货价格: {ts_code}")
        return pd.DataFrame(columns=["trade_date", "price"])
    raw = pd.concat(frames, ignore_index=True)
    raw["price"] = raw["settle"].where(raw["settle"].notna() & (raw["settle"] > 0), raw["close"])
    raw = raw.dropna(subset=["price"])
    raw = raw[raw["price"] > 0]
    out = (raw[["trade_date", "price"]]
           .drop_duplicates(subset="trade_date", keep="last")
           .sort_values("trade_date").reset_index(drop=True))
    out.to_parquet(cache, index=False)
    print(f"[futures] {ts_code} 连续价格 {len(out)} 天 -> 缓存")
    return out


def fetch_roll_days(client: TushareClient, code: str, suffix: str) -> set:
    """主力合约切换的交易日（这些日的对数收益跨合约，需剔除）。"""
    cache = _mapping_cache(code)
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        ts_code = f"{code}.{suffix}"
        frames = []
        for y in market_years():
            df = client.query("fut_mapping", ts_code=ts_code,
                              start_date=f"{y}0101", end_date=f"{y}1231")
            if df is None or df.empty:
                continue
            frames.append(df)
        if not frames:
            print(f"[futures] 无换月映射: {ts_code}（不剔除换月日）")
            return set()
        df = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset="trade_date", keep="last")
              .sort_values("trade_date").reset_index(drop=True))
        df.to_parquet(cache, index=False)
    mapped = df["mapping_ts_code"].astype(str)
    changed = mapped != mapped.shift(1)
    return set(df.loc[changed, "trade_date"].astype(str))


def fetch_industry_members(client: TushareClient, industry_codes: List[str]) -> pd.DataFrame:
    """申万 L1 行业成员，列: symbol, l1_code, l1_name, in_date, out_date。带缓存。"""
    cache = _members_cache()
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for code in industry_codes:
        df = client.query("index_member_all", l1_code=code)
        if df is None or df.empty:
            print(f"[futures] 行业成员为空: {code}")
            continue
        df = df.copy()
        df["l1_code"] = code
        frames.append(df)
        print(f"[futures] 行业成员 {code}: {len(df)}")
    if not frames:
        return pd.DataFrame(columns=["symbol", "l1_code", "l1_name", "in_date", "out_date"])
    allm = pd.concat(frames, ignore_index=True)
    allm["symbol"] = allm["ts_code"].map(normalize_symbol)
    allm = allm[["symbol", "l1_code", "l1_name", "in_date", "out_date"]].drop_duplicates(
        subset=["symbol", "l1_code"], keep="last")
    allm.to_parquet(cache, index=False)
    print(f"[futures] 行业成员缓存: {len(allm)} 条 -> {cache.name}")
    return allm


# -- collect 阶段的批量采集器 -------------------------------------------

class FuturesCollector:
    """按权重表批量预采集商品期货与行业成员（cli collect 调用）。"""

    def __init__(self, client: Optional[TushareClient] = None) -> None:
        self.client = client or TushareClient()

    def collect(self) -> pd.DataFrame:
        wpath = CONFIG.market_dir / "industry_commodity_weight.csv"
        if not wpath.exists():
            print("[futures] 缺少权重表 repository/market/industry_commodity_weight.csv，"
                  "请先运行 scripts/gen_industry_commodity_mapping.py，跳过期货采集")
            return pd.DataFrame(columns=["commodity_code", "days", "mapping_days"])

        weights = pd.read_csv(wpath, encoding="utf-8-sig")
        if "suffix" not in weights.columns:
            print("[futures] 权重表缺少 suffix 列，请重新生成映射表，跳过")
            return pd.DataFrame(columns=["commodity_code", "days", "mapping_days"])

        rows = []
        codes = sorted(weights["commodity_code"].unique())
        print(f"[futures] 待采集商品 {len(codes)} 个，年份 {market_years()[0]}–{market_years()[-1]}")
        for i, code in enumerate(codes, 1):
            suffix = weights.loc[weights["commodity_code"] == code, "suffix"].iloc[0]
            px = fetch_commodity_price(self.client, code, suffix)
            mapping = fetch_roll_days(self.client, code, suffix)
            rows.append({"commodity_code": code, "days": int(len(px)),
                         "mapping_days": int(len(mapping))})
            print(f"[futures] 进度 {i}/{len(codes)}: {code} 价格 {len(px)} 天 / 映射 {len(mapping)} 天")

        # 行业成员（与标签器共用缓存）
        members = fetch_industry_members(self.client, sorted(weights["industry_code"].unique()))
        print(f"[futures] 行业成员 {len(members)} 条")

        summary = pd.DataFrame(rows)
        ok = int((summary["days"] > 0).sum()) if not summary.empty else 0
        print(f"[futures] 完成：{ok}/{len(codes)} 个商品有价格数据 -> {CONFIG.market_dir / 'futures'}")
        return summary
