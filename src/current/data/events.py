"""风险事件采集：ST/*ST 状态 + 退市事件（方案D 混合标签的真值来源）。

对供应链全集每家公司的每个年份标记：
- st_level：0=无 / 1=ST（其他风险警示） / 2=*ST（退市风险警示）
- delisted：当年是否退市（仅退市简称带 ST/*ST 前缀的“失败退市”视为风险事件）
- event_probability：该年对应的事件违约概率（见 config.LabelConfig）

数据来源（Tushare）：
- namechange：按 ts_code 查询曾用名，从 change_reason/name 提取 ST 起止区间。
  观测到的 reason 取值：ST / *ST / 从ST变为*ST / 撤销ST / 撤销*ST / 其他。
  ▶ 含 "ST" 且不含 "撤销" 的记录 = 一段 ST 状态区间，取记录自身的 [start_date, end_date]。
  ▶ 含 "*ST" 或 "退市风险警示" → level 2；否则 level 1。
- stock_basic(list_status='D')：退市股票列表（ts_code, name, list_date, delist_date）。
  按 delist_date 所在年标记退市；delist 日名称含 *ST/ST 视为风险退市（失败退市）。

产出 interim/events.parquet，行主键 (symbol, year)；支持断点续跑（按 symbol 记状态）。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.current.config import CONFIG
from src.current.data.tushare_client import TushareClient
from src.current.transform.symbols import normalize_symbol, to_ts_code

_COLUMNS = [
    "symbol", "year", "ts_code", "data_status",
    "st_level", "delisted", "delist_date", "event_probability",
]

_CURRENT_YEAR = 2026  # 未摘帽的持续 ST 区间按此年截止


def _to_year(date_str) -> Optional[int]:
    if date_str is None or pd.isna(date_str) or str(date_str).strip() == "":
        return None
    s = str(date_str).strip()
    return int(s[:4])


def _is_star_text(text: str) -> bool:
    return "*ST" in text or "＊ST" in text or "退市风险警示" in text or "ST＊" in text


def _is_st_text(text: str) -> bool:
    return "ST" in text or "＊ST" in text or "退市风险警示" in text or "特别处理" in text


class EventCollector:
    def __init__(self, client: Optional[TushareClient] = None) -> None:
        self.client = client or TushareClient()

    # -- 状态/缓存 -------------------------------------------------------
    def _load_existing(self) -> pd.DataFrame:
        p = CONFIG.events_interim
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
            done.add(str(row["symbol"]))
        return done

    # -- ST 状态（namechange）-------------------------------------------
    def _st_periods(self, ts_code: str) -> List[dict]:
        """解析一只股票的 ST/*ST 状态区间列表。

        Returns: [{"start": int, "end": Optional[int], "level": 1|2}, ...]
        end=None 表示持续至今。
        """
        df = self.client.query("namechange", ts_code=ts_code,
                               fields="ts_code,name,start_date,end_date,change_reason")
        if df is None or df.empty or "change_reason" not in df.columns:
            return []

        periods: List[dict] = []
        for _, row in df.iterrows():
            reason = str(row.get("change_reason") or "")
            name = str(row.get("name") or "")
            if "撤销" in reason:          # 摘帽记录，区间由进入记录自带的 end 覆盖
                continue
            if not (_is_st_text(name) or _is_st_text(reason)):
                continue
            start = _to_year(row.get("start_date"))
            end = _to_year(row.get("end_date"))
            if start is None:
                continue
            level = 2 if (_is_star_text(name) or _is_star_text(reason)) else 1
            periods.append({"start": start, "end": end, "level": level})
        return periods

    def _st_rows(self, symbol: str, ts_code: str) -> List[dict]:
        """把 ST 区间铺展为逐 (symbol, year) 行。"""
        periods = self._st_periods(ts_code)
        rows: List[dict] = []
        for p in periods:
            e = p["end"] if p["end"] is not None else _CURRENT_YEAR
            for y in range(p["start"], e + 1):
                rows.append({"symbol": symbol, "year": y, "level": int(p["level"])})
        return rows

    # -- 退市（stock_basic）---------------------------------------------
    def _delist_rows(self) -> List[dict]:
        df = self.client.query("stock_basic", list_status="D",
                               fields="ts_code,name,list_date,delist_date")
        if df is None or df.empty or "delist_date" not in df.columns:
            return []
        rows: List[dict] = []
        for _, r in df.iterrows():
            sym = normalize_symbol(r.get("ts_code"))
            if not sym:
                continue
            delist_date = r.get("delist_date")
            yr = _to_year(delist_date)
            if yr is None:
                continue
            name = str(r.get("name") or "")
            risk = _is_st_text(name)      # 失败退市（简称带 ST/*ST）
            rows.append({"symbol": sym, "delist_year": yr,
                         "delist_date": str(delist_date), "risk": risk})
        return rows

    # -- 汇总 -----------------------------------------------------------
    def collect(self, symbols: List[str], resume: bool = True,
                save_every: int = 100) -> pd.DataFrame:
        existing = self._load_existing() if resume else pd.DataFrame(columns=_COLUMNS)
        done = self._done_keys(existing) if resume else set()
        todo = [s for s in symbols if s not in done]
        print(f"[events] 待采集 {len(todo)} / {len(symbols)} 家（已完成 {len(done)}）")

        merged = {(str(r["symbol"]), int(r["year"])): r for r in existing.to_dict("records")}

        if len(todo) > 0:
            for i, symbol in enumerate(todo, 1):
                sym6 = str(symbol).zfill(6)
                ts_code = to_ts_code(sym6)
                if not ts_code:
                    merged[(sym6, -1)] = {"symbol": sym6, "year": -1, "ts_code": None,
                                          "data_status": "failed:no_ts_code", "st_level": 0,
                                          "delisted": 0, "delist_date": None, "event_probability": 0.0}
                    continue
                st_rows = self._st_rows(sym6, ts_code)
                if not st_rows:
                    # 无 ST 记录：仅标记该 symbol 已完成（占位 -1 不妨碍合并）
                    merged[(sym6, -1)] = {"symbol": sym6, "year": -1, "ts_code": ts_code,
                                          "data_status": "empty", "st_level": 0,
                                          "delisted": 0, "delist_date": None, "event_probability": 0.0}
                else:
                    for r in st_rows:
                        merged[(r["symbol"], r["year"])] = {
                            "symbol": r["symbol"], "year": r["year"], "ts_code": ts_code,
                            "data_status": "success", "st_level": r["level"],
                            "delisted": 0, "delist_date": None,
                            "event_probability": 0.0,
                        }
                if i % save_every == 0:
                    self._save(list(merged.values()))
                    print(f"[events] ST 进度 {i}/{len(todo)}，已落盘临时结果")

        # 退市事件（全局一次拉取，覆盖所有 symbol）
        delist_rows = self._delist_rows()
        st_prob = CONFIG.labels.st_probability
        star_prob = CONFIG.labels.star_st_probability
        delist_prob = CONFIG.labels.delist_probability

        n_matched = 0
        for dr in delist_rows:
            key = (dr["symbol"], dr["delist_year"])
            ev = delist_prob if dr["risk"] else 0.0   # 失败退市才上调概率
            if key in merged:
                r = merged[key]
                r["delisted"] = 1
                r["delist_date"] = dr["delist_date"]
                r["event_probability"] = max(r.get("event_probability", 0.0), ev)
                n_matched += 1
            else:
                merged[key] = {"symbol": dr["symbol"], "year": dr["delist_year"],
                               "ts_code": (dr["symbol"] + ".SH"), "data_status": "success",
                               "st_level": 2 if dr["risk"] else 0, "delisted": 1,
                               "delist_date": dr["delist_date"], "event_probability": ev}
        print(f"[events] 退市事件共 {len(delist_rows)} 条，与样本匹配 {n_matched} 条")

        # 事件概率 = max(ST/*ST/退市)
        for rec in merged.values():
            if rec.get("st_level") == 2:
                rec["event_probability"] = max(rec.get("event_probability", 0.0), star_prob)
            elif rec.get("st_level") == 1:
                rec["event_probability"] = max(rec.get("event_probability", 0.0), st_prob)
            if rec.get("st_level") is None:
                rec["st_level"] = 0
            if rec.get("delisted") is None:
                rec["delisted"] = 0

        self._save(list(merged.values()))
        out = pd.DataFrame(list(merged.values()))
        active = out[out["year"] > 0]
        print(f"[events] 完成：总 {len(out)} 行（含占位），事件行 {len(active)} -> {CONFIG.events_interim}")
        return out

    @staticmethod
    def _save(records: List[dict]) -> None:
        df = pd.DataFrame(records)
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df["year"] = pd.to_numeric(df.get("year", np.nan), errors="coerce").fillna(-1).astype(int)
        df = df[_COLUMNS]
        df.to_parquet(CONFIG.events_interim, index=False)