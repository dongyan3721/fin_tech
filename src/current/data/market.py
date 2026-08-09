"""行情采集：为 KMV 提供股权市值与资产波动率。

对每个 (symbol, year)：
- market_cap：年末 total_mv（daily_basic，单位万元 -> 元）；缺失则用年末收盘价 × 总股本兜底。
- asset_volatility：过去约一年日收益率年化标准差（daily）。

产出 interim/market.parquet。也是未来市场风险(GARCH)标签的行情数据来源之一。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.current.config import CONFIG
from src.current.data.tushare_client import TushareClient
from src.current.transform.symbols import to_ts_code

_COLUMNS = [
    "symbol", "year", "ts_code", "data_status",
    "close", "total_mv", "market_cap", "asset_volatility"
]


def _annualized_vol(daily: pd.DataFrame, fallback: float, floor: float) -> float:
    if daily is None or daily.empty or "close" not in daily.columns:
        return fallback
    try:
        d = daily.sort_values("trade_date")
        ret = d["close"].pct_change()
        vol = ret.std() * np.sqrt(252)
        if pd.isna(vol):
            return fallback
        return max(float(vol), floor)
    except Exception:
        return fallback


class MarketCollector:
    def __init__(self, client: Optional[TushareClient] = None) -> None:
        self.client = client or TushareClient()

    def _load_existing(self) -> pd.DataFrame:
        p = CONFIG.market_interim
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame(columns=_COLUMNS)

    @staticmethod
    def _done_keys(df: pd.DataFrame) -> set:
        done = set()
        if df.empty:
            return done
        for _, row in df.iterrows():
            status = str(row.get("data_status", "success"))
            if status.startswith(("failed", "error")):
                continue
            done.add((str(row["symbol"]), int(row["year"])))
        return done

    def _fetch_one(self, symbol: str, year: int) -> dict:
        ts_code = to_ts_code(symbol)
        if not ts_code:
            return {"symbol": symbol, "year": year, "ts_code": None, "data_status": "failed:no_ts_code"}
        start = f"{year - 1}0101"
        end = f"{year}1231"
        daily = self.client.daily(ts_code=ts_code, start_date=start, end_date=end)
        basic = self.client.daily_basic(ts_code=ts_code, start_date=start, end_date=end,
                                        fields="trade_date,close,total_mv,total_share")

        vol = _annualized_vol(daily, CONFIG.labels.fallback_volatility, CONFIG.labels.min_asset_volatility)

        close = np.nan
        total_mv = np.nan
        market_cap = np.nan
        if daily is not None and not daily.empty:
            last = daily.sort_values("trade_date").iloc[-1]
            close = float(last.get("close", np.nan))
        if basic is not None and not basic.empty and "total_mv" in basic.columns:
            b = basic.dropna(subset=["total_mv"]).sort_values("trade_date")
            if not b.empty:
                total_mv = float(b["total_mv"].iloc[-1])  # 万元
                market_cap = total_mv * 1e4  # -> 元

        if pd.isna(market_cap) and pd.notna(close):
            # 兜底：收盘价 × 总股本（total_share 单位万股）
            if basic is not None and not basic.empty and "total_share" in basic.columns:
                bs = basic.dropna(subset=["total_share"]).sort_values("trade_date")
                if not bs.empty:
                    market_cap = close * float(bs["total_share"].iloc[-1]) * 1e4

        status = "success"
        if pd.isna(market_cap) and pd.isna(close):
            status = "empty"
        return {"symbol": symbol, "year": year, "ts_code": ts_code, "data_status": status,
                "close": close, "total_mv": total_mv, "market_cap": market_cap,
                "asset_volatility": vol}

    def collect(self, combos: List[Tuple[str, int]], resume: bool = True,
                save_every: int = 50) -> pd.DataFrame:
        existing = self._load_existing() if resume else pd.DataFrame(columns=_COLUMNS)
        done = self._done_keys(existing) if resume else set()
        todo = [c for c in combos if c not in done]
        print(f"[market] 待采集 {len(todo)} / 全集 {len(combos)}（已完成 {len(done)}）")

        merged = {(str(r["symbol"]), int(r["year"])): r for r in existing.to_dict("records")}
        for i, (symbol, year) in enumerate(todo, 1):
            merged[(symbol, year)] = self._fetch_one(symbol, year)
            if i % save_every == 0:
                self._save(list(merged.values()))
                print(f"[market] 进度 {i}/{len(todo)}，已落盘临时结果")
        self._save(list(merged.values()))
        out = pd.DataFrame(list(merged.values()))
        print(f"[market] 完成：总 {len(out)} 行 -> {CONFIG.market_interim}")
        return out

    @staticmethod
    def _save(records: List[dict]) -> None:
        df = pd.DataFrame(records)
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[_COLUMNS]
        df.to_parquet(CONFIG.market_interim, index=False)
