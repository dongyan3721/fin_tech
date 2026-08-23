"""股票代码统一处理（全项目唯一实现）。

- ``normalize_symbol``：统一为 6 位字符串（如 26 -> "000026"），用于对齐财务/边/标签表。
- ``to_ts_code``：转换为 Tushare 需要的带交易所后缀格式（如 "000026.SZ"）。
"""
from __future__ import annotations

import re
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


def _pick_ashare(codes: list[str]) -> str:
    """从多个 6 位数字代码中优先取 A 股（0/3/6 开头），其次其他。"""
    for pref in "036948":
        for c in codes:
            if c[0] == pref:
                return c
    return codes[0]


def normalize_symbol(value) -> str:
    """把任意来源的股票代码统一为 6 位字符串；非法/空返回空串。

    兼容多种脏数据：
    - 数值型被 Excel 吞掉前导零（672 -> "000672"）；
    - 多代码黏连（A+H / A+B / 境外，如 "00386;600028;SNP"）拆出 6 位 A 股代码；
    - 纯字母 ticker 的境外公司（BIDU/JD 等）原样保留作稳定标识。
    """
    if _is_na(value):
        return ""
    s = str(value).strip().replace(" ", "")
    if s in ("", "nan", "None"):
        return ""

    # 多代码黏连（; ,）拆分
    if ";" in s or "," in s:
        parts = [p for p in re.split(r"[;,]+", s) if p.strip()]
        codes: list[str] = []
        for p in parts:
            p = p.strip()
            if p.endswith(".0"):
                p = p[:-2]
            if "." in p:
                p = p.split(".")[0]
            if p.isdigit() and len(p) == 6:
                codes.append(p)
        if codes:
            return _pick_ashare(codes)
        # 无 6 位 A 股：优先纯字母 ticker（境外公司），否则最长数字串补零
        for p in parts:
            p2 = p.strip().replace(".0", "")
            if p2.isalpha() and 1 <= len(p2) <= 6:
                return p2
        digits = [p for p in parts if p.strip().isdigit()]
        if digits:
            return max(digits, key=len).zfill(6)[-6:]
        return parts[0]

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
