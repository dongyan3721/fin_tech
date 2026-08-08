"""股票代码统一处理（全项目唯一实现）。

- ``normalize_symbol``：统一为 6 位字符串（如 26 -> "000026"），用于对齐财务/边/标签表。
- ``to_ts_code``：转换为 Tushare 需要的带交易所后缀格式（如 "000026.SZ"）。
"""
from __future__ import annotations

from typing import Optional

try:  # pandas 仅用于 NaN 判断，避免强依赖顺序
    import pandas as pd

    def _is_na(v) -> bool:
        try:
            return pd.isna(v)
        except Exception:
            return v is None
except Exception:  # pragma: no cover
    def _is_na(v) -> bool:
        return v is None


def normalize_symbol(value) -> str:
    """把任意来源的股票代码统一为 6 位字符串；非法/空返回空串。"""
    if _is_na(value):
        return ""
    s = str(value).strip().replace(" ", "")
    if s in ("", "nan", "None"):
        return ""
    if s.endswith(".0"):  # Excel 读成浮点的情况
        s = s[:-2]
    # 带交易所后缀（000026.SZ）时取前段
    if "." in s:
        s = s.split(".")[0]
    if s.isdigit():
        if len(s) > 6:
            return s[:6]
        return s.zfill(6)
    return s


def to_ts_code(value) -> Optional[str]:
    """6 位代码 -> Tushare ts_code（推断交易所后缀）。无法识别返回 None。"""
    sym = normalize_symbol(value)
    if not sym or not sym.isdigit() or len(sym) != 6:
        return None
    head = sym[0]
    if head in ("0", "3"):
        return f"{sym}.SZ"
    if head in ("6", "9"):
        return f"{sym}.SH"
    if head in ("4", "8"):
        return f"{sym}.BJ"  # 北交所
    return f"{sym}.SH"
