"""财务节点特征采集：从 Tushare 资产负债表/利润表算出 10 个模型特征。

产出 interim/financial.parquet，列包含主键(symbol,year)、data_status 与全部特征列。
支持断点续跑：success/empty 视为已定性，failed/error 允许重试。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.current.config import CONFIG
from src.current.data.tushare_client import TushareClient
from src.current.transform.symbols import to_ts_code

# 注意 Tushare 字段名与内部特征名不同：
#   current_assets -> total_cur_assets, current_liab -> total_cur_liab, inventory -> inventories
_BALANCE_FIELDS = "total_assets,total_liab,total_cur_assets,total_cur_liab,inventories"
_INCOME_FIELDS = "revenue,operate_profit,fin_exp"

# 采集/落盘涉及的所有列
_COLUMNS = [
    "symbol", "year", "ts_code", "data_status",
    "debt_to_asset_ratio", "current_ratio", "quick_ratio", "interest_coverage_ratio",
    "total_assets", "total_liab", "total_cur_liab",
    "current_assets", "current_liab", "revenue", "operate_profit",
    "inventory", "fin_exp",
]


def _first(df: pd.DataFrame, field: str):
    if df is None or df.empty or field not in df.columns:
        return np.nan
    val = df[field].iloc[0]
    return np.nan if pd.isna(val) else val


def _ratios(bs: pd.DataFrame, inc: pd.DataFrame, symbol: str, year: int, ts_code: str) -> dict:
    total_assets = _first(bs, "total_assets")
    total_liab = _first(bs, "total_liab")
    current_assets = _first(bs, "total_cur_assets")
    total_cur_liab = _first(bs, "total_cur_liab")  # 流动负债（用于 KMV DPT 计算）
    inventory = _first(bs, "inventories")
    revenue = _first(inc, "revenue")
    operate_profit = _first(inc, "operate_profit")
    fin_exp = _first(inc, "fin_exp")

    rec = {
        "symbol": symbol, "year": year, "ts_code": ts_code, "data_status": "success", "total_assets": total_assets,
        "total_liab": total_liab, "total_cur_liab": total_cur_liab,
        "current_assets": current_assets, "current_liab": total_cur_liab, "revenue": revenue,
        "operate_profit": operate_profit, "inventory": inventory, "fin_exp": fin_exp,
        "debt_to_asset_ratio": (total_liab / total_assets) if (pd.notna(total_assets) and total_assets) else np.nan,
        "current_ratio": (current_assets / total_cur_liab) if (pd.notna(total_cur_liab) and total_cur_liab) else np.nan
    }

    if pd.notna(current_liab) and current_liab and pd.notna(current_assets):
        inv = 0.0 if pd.isna(inventory) else inventory
        rec["quick_ratio"] = (current_assets - inv) / current_liab
    else:
        rec["quick_ratio"] = np.nan
    if pd.notna(fin_exp) and fin_exp != 0 and pd.notna(operate_profit):
        fe = abs(fin_exp)
        rec["interest_coverage_ratio"] = (operate_profit + fe) / fe
    else:
        rec["interest_coverage_ratio"] = np.nan
    return rec


class FinancialCollector:
    def __init__(self, client: Optional[TushareClient] = None) -> None:
        self.client = client or TushareClient()

    def _load_existing(self) -> pd.DataFrame:
        p = CONFIG.financial_interim
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
        period = f"{year}1231"
        bs = self.client.balancesheet(ts_code=ts_code, period=period, fields=_BALANCE_FIELDS)
        inc = self.client.income(ts_code=ts_code, period=period, fields=_INCOME_FIELDS)
        if (bs is None or bs.empty) and (inc is None or inc.empty):
            return {"symbol": symbol, "year": year, "ts_code": ts_code, "data_status": "empty"}
        return _ratios(bs, inc, symbol, year, ts_code)

    def collect(self, combos: List[Tuple[str, int]], resume: bool = True,
                save_every: int = 50) -> pd.DataFrame:
        existing = self._load_existing() if resume else pd.DataFrame(columns=_COLUMNS)
        done = self._done_keys(existing) if resume else set()
        todo = [c for c in combos if c not in done]
        print(f"[financial] 待采集 {len(todo)} / 全集 {len(combos)}（已完成 {len(done)}）")

        records: List[dict] = []
        merged = {(str(r["symbol"]), int(r["year"])): r for r in existing.to_dict("records")}
        for i, (symbol, year) in enumerate(todo, 1):
            rec = self._fetch_one(symbol, year)
            records.append(rec)
            merged[(symbol, year)] = rec
            if i % save_every == 0:
                self._save(list(merged.values()))
                print(f"[financial] 进度 {i}/{len(todo)}，已落盘临时结果")
        self._save(list(merged.values()))
        out = pd.DataFrame(list(merged.values()))
        ok = (out["data_status"] == "success").sum() if not out.empty else 0
        print(f"[financial] 完成：总 {len(out)} 行，成功 {ok} 行 -> {CONFIG.financial_interim}")
        return out

    @staticmethod
    def _save(records: List[dict]) -> None:
        df = pd.DataFrame(records)
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[_COLUMNS]
        df.to_parquet(CONFIG.financial_interim, index=False)
