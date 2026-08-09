"""供应链数据采集/清洗：从本地整合 Excel 生成边表，并给出需要采集的 (symbol, year) 全集。

边定义（与 legacy 语义一致）：
- supply：供应商 -> 核心公司（source=Supplier_Symbol, target=Symbol，weight=采购额）
- sale：  核心公司 -> 客户    （source=Symbol, target=Customer_Symbol，weight=销售额）

扩展插入点（EDGE_SOURCES）：未来若接入其他边来源（如工商股权、Tushare 关联方），
新增一个返回同样 EDGE 列结构的函数并在 collect_edges 里合并即可。
"""
from __future__ import annotations

from typing import List, Set, Tuple

import pandas as pd

from src.current.config import CONFIG
from src.current.transform.symbols import normalize_symbol


def _read_raw() -> pd.DataFrame:
    df = pd.read_excel(CONFIG.raw_supply_chain_xlsx, sheet_name=CONFIG.raw_supply_chain_sheet)
    return df


def collect_edges() -> pd.DataFrame:
    """构建边表 DataFrame，列: source,target,weight,relationship,proportion,year。"""
    df = _read_raw()
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    core = df["Symbol"].map(normalize_symbol)
    supplier = df["Supplier_Symbol"].map(normalize_symbol)
    customer = df["Customer_Symbol"].map(normalize_symbol)

    edges: List[pd.DataFrame] = []

    supply_mask = (supplier != "") & (core != "")
    if supply_mask.any():
        edges.append(pd.DataFrame({
            "source": supplier[supply_mask],
            "target": core[supply_mask],
            "weight": pd.to_numeric(df.loc[supply_mask, "Purchase_Amount"], errors="coerce").fillna(1.0),
            "relationship": "supply",
            "proportion": pd.to_numeric(df.loc[supply_mask, "Purchase_Proportion"], errors="coerce"),
            "year": df.loc[supply_mask, "Year"],
        }))

    sale_mask = (customer != "") & (core != "")
    if sale_mask.any():
        edges.append(pd.DataFrame({
            "source": core[sale_mask],
            "target": customer[sale_mask],
            "weight": pd.to_numeric(df.loc[sale_mask, "Sales_Amount"], errors="coerce").fillna(1.0),
            "relationship": "sale",
            "proportion": pd.to_numeric(df.loc[sale_mask, "Sales_Proportion"], errors="coerce"),
            "year": df.loc[sale_mask, "Year"],
        }))

    if not edges:
        return pd.DataFrame(columns=["source", "target", "weight", "relationship", "proportion", "year"])

    out = pd.concat(edges, ignore_index=True)
    out = out[(out["source"] != "") & (out["target"] != "") & (out["source"] != out["target"])]
    # 同一 (source,target,relationship,year) 去重，权重取最大（避免重复披露）
    out = (out.sort_values("weight", ascending=False)
              .drop_duplicates(subset=["source", "target", "relationship", "year"], keep="first")
              .reset_index(drop=True))
    return out


def collect_symbol_year_universe() -> List[Tuple[str, int]]:
    """需要采集财务/行情的 (symbol, year) 全集：核心 + 供应商 + 客户。"""
    df = _read_raw().copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    universe: Set[Tuple[str, int]] = set()
    for col in ("Symbol", "Supplier_Symbol", "Customer_Symbol"):
        if col not in df.columns:
            continue
        syms = df[col].map(normalize_symbol)
        for sym, year in zip(syms, df["Year"]):
            if sym:
                universe.add((sym, int(year)))
    return sorted(universe)
