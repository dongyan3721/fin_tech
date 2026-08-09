"""合并三个供应链 Excel 为代码可读的统一格式。

源文件:
  1. repository/supply/整合的供应链数据.xlsx       (2001-2023, sheet:供应链网络数据)
  2. repository/supply/SC_TopFivePurchaseInfo 2024-2025.xlsx  (前五大供应商)
  3. repository/supply/SC_TopFiveSaleInfo 2024-2025.xlsx      (前五大客户)

输出:
  data/raw/整合的供应链数据.xlsx  (sheet:供应链网络数据, 2001-2025)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPLY_DIR = PROJECT_ROOT / "repository" / "supply"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "整合的供应链数据.xlsx"
OUTPUT_SHEET = "供应链网络数据"

# 目标列（与 supply_chain.py 期望的 schema 一致）
TARGET_COLUMNS = [
    "Year", "Symbol", "EndDate",
    "Supplier_InstitutionID", "Supplier_Name", "Supplier_Symbol",
    "Purchase_Amount", "Purchase_Proportion",
    "Customer_InstitutionID", "Customer_Name", "Customer_Symbol",
    "Sales_Amount", "Sales_Proportion",
]


def _extract_year(enddate) -> int | None:
    """从 EndDate 提取年份，支持 datetime / 'YYYY-MM-DD' 字符串。"""
    if pd.isna(enddate):
        return None
    if isinstance(enddate, (int, float)):
        return int(enddate)
    s = str(enddate).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def load_legacy() -> pd.DataFrame:
    """读取原始整合数据 (2001-2023)。"""
    src = SUPPLY_DIR / "整合的供应链数据.xlsx"
    df = pd.read_excel(src, sheet_name=OUTPUT_SHEET)
    # 确保 Year 为 int
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    print(f"[legacy] {len(df)} 行, 年份 {df['Year'].min()}-{df['Year'].max()}")
    return df


def load_top5_purchase() -> pd.DataFrame:
    """读取 2024-2025 前五大供应商，转换为统一 schema。"""
    src = SUPPLY_DIR / "SC_TopFivePurchaseInfo 2024-2025.xlsx"
    raw = pd.read_excel(src, sheet_name="sheet1")
    # 过滤: 仅保留 Rank 1-5 的有效数据行
    raw = raw[raw["Rank"].isin([1, 2, 3, 4, 5])].copy()
    raw["EndDate"] = raw["EndDate"].astype(str).str.strip()
    raw = raw[raw["EndDate"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    raw["Year"] = raw["EndDate"].apply(_extract_year)
    raw = raw.dropna(subset=["Year"])
    raw["Year"] = raw["Year"].astype(int)
    raw = raw.reset_index(drop=True)

    # 仅上市公司有 BusinessSymbol
    mask_listed = raw["IsListed"].astype(str).str.upper().eq("Y") & raw["BusinessSymbol"].notna()
    supplier_symbol = pd.Series([""] * len(raw), index=raw.index)
    supplier_symbol[mask_listed] = raw.loc[mask_listed, "BusinessSymbol"].astype(str)

    out = pd.DataFrame({
        "Year": raw["Year"].values,
        "Symbol": raw["Symbol"].values,
        "EndDate": raw["EndDate"].values,
        "Supplier_InstitutionID": raw["BusinessInstitutionID"].values,
        "Supplier_Name": raw["InstitutionName"].values,
        "Supplier_Symbol": supplier_symbol.values,
        "Purchase_Amount": pd.to_numeric(raw["PurchaseAmount"], errors="coerce").values,
        "Purchase_Proportion": pd.to_numeric(raw["ProportionOfTotalValue"], errors="coerce").values,
        "Customer_InstitutionID": [None] * len(raw),
        "Customer_Name": [None] * len(raw),
        "Customer_Symbol": [""] * len(raw),
        "Sales_Amount": [None] * len(raw),
        "Sales_Proportion": [None] * len(raw),
    }, columns=TARGET_COLUMNS)
    print(f"[purchase] {len(out)} 行, 年份 {sorted(out['Year'].unique())}")
    return out


def load_top5_sale() -> pd.DataFrame:
    """读取 2024-2025 前五大客户，转换为统一 schema。"""
    src = SUPPLY_DIR / "SC_TopFiveSaleInfo 2024-2025.xlsx"
    raw = pd.read_excel(src, sheet_name="sheet1")
    raw = raw[raw["Rank"].isin([1, 2, 3, 4, 5])].copy()
    raw["EndDate"] = raw["EndDate"].astype(str).str.strip()
    raw = raw[raw["EndDate"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    raw["Year"] = raw["EndDate"].apply(_extract_year)
    raw = raw.dropna(subset=["Year"])
    raw["Year"] = raw["Year"].astype(int)
    raw = raw.reset_index(drop=True)

    mask_listed = raw["IsListed"].astype(str).str.upper().eq("Y") & raw["BusinessSymbol"].notna()
    customer_symbol = pd.Series([""] * len(raw), index=raw.index)
    customer_symbol[mask_listed] = raw.loc[mask_listed, "BusinessSymbol"].astype(str)

    out = pd.DataFrame({
        "Year": raw["Year"].values,
        "Symbol": raw["Symbol"].values,
        "EndDate": raw["EndDate"].values,
        "Supplier_InstitutionID": [None] * len(raw),
        "Supplier_Name": [None] * len(raw),
        "Supplier_Symbol": [""] * len(raw),
        "Purchase_Amount": [None] * len(raw),
        "Purchase_Proportion": [None] * len(raw),
        "Customer_InstitutionID": raw["BusinessInstitutionID"].values,
        "Customer_Name": raw["InstitutionName"].values,
        "Customer_Symbol": customer_symbol.values,
        "Sales_Amount": pd.to_numeric(raw["SalesAmount"], errors="coerce").values,
        "Sales_Proportion": pd.to_numeric(raw["ProportionOfTotalValue"], errors="coerce").values,
    }, columns=TARGET_COLUMNS)
    print(f"[sale] {len(out)} 行, 年份 {sorted(out['Year'].unique())}")
    return out


def main() -> None:
    print("=" * 60)
    print("供应链数据合并工具")
    print("=" * 60)

    df_legacy = load_legacy()
    df_purchase = load_top5_purchase()
    df_sale = load_top5_sale()

    # 合并
    merged = pd.concat([df_legacy, df_purchase, df_sale], ignore_index=True)
    merged = merged[TARGET_COLUMNS]
    merged["Year"] = merged["Year"].astype(int)
    merged = merged.sort_values(["Year", "Symbol"]).reset_index(drop=True)

    # 统计
    print(f"\n[合并后] 总行数: {len(merged)}")
    print(f"  年份范围: {merged['Year'].min()}-{merged['Year'].max()}")
    print(f"  唯一公司数: {merged['Symbol'].nunique()}")
    for y in sorted(merged["Year"].unique()):
        cnt = len(merged[merged["Year"] == y])
        print(f"    {y}: {cnt} 行")

    # 写入
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_excel(OUTPUT_FILE, sheet_name=OUTPUT_SHEET, index=False)
    print(f"\n已写入: {OUTPUT_FILE}")
    print("完成!")


if __name__ == "__main__":
    main()
