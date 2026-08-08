"""
用 Tushare 重建财务特征与 KMV 伪标签，并导出规范三表。

用法（在项目根目录）:
  .\\.venv\\Scripts\\python.exe src\\rebuild_dataset.py --step all
  .\\.venv\\Scripts\\python.exe src\\rebuild_dataset.py --step financial
  .\\.venv\\Scripts\\python.exe src\\rebuild_dataset.py --step kmv
  .\\.venv\\Scripts\\python.exe src\\rebuild_dataset.py --step export
  .\\.venv\\Scripts\\python.exe src\\rebuild_dataset.py --step edges

说明:
  - Token 从项目根目录 .env 的 TUSHARE_TOKEN 读取（勿提交到 git）
  - 供应链边不依赖 Tushare，从 data/raw 的整合表生成
  - 支持断点续跑；全量约 9000+ 公司-年，耗时较长
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REBUILD_DIR = ROOT / "data" / "rebuild"
LEGACY_DIR = ROOT / "src" / "legacy"


def setup_env() -> None:
    load_dotenv(ROOT / ".env")
    import os

    if not os.getenv("TUSHARE_TOKEN"):
        raise SystemExit(
            "未找到 TUSHARE_TOKEN。请在项目根目录 .env 中配置，"
            "或先在当前终端设置环境变量后再运行。"
        )
    sys.path.insert(0, str(LEGACY_DIR))
    sys.path.insert(0, str(ROOT / "src"))


def find_raw_excel() -> Path:
    files = list(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"未在 {RAW_DIR} 找到原始供应链 Excel")
    # 优先「整合的供应链数据」
    for f in files:
        if "整合" in f.name or "供应链" in f.name:
            return f
    return files[0]


def backup_processed() -> Path | None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        "financial_indicators_robust.csv",
        "financial_indicators_robust_raw.csv",
        "kmv_analysis_results.csv",
        "kmv_analysis_results_raw.csv",
        "combined_edges.xlsx",
        "nodes.parquet",
        "edges.parquet",
        "labels.parquet",
    ]
    existing = [PROCESSED_DIR / n for n in targets if (PROCESSED_DIR / n).exists()]
    if not existing:
        print("processed 下无可备份旧文件")
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = PROCESSED_DIR / f"_backup_{stamp}"
    bak.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.copy2(p, bak / p.name)
    print(f"已备份旧数据到: {bak}")
    return bak


def step_financial(resume: bool = True) -> Path:
    from node_features import FinancialDataExtractor

    REBUILD_DIR.mkdir(parents=True, exist_ok=True)
    excel = find_raw_excel()
    out = REBUILD_DIR / "financial_indicators_robust.csv"
    print(f"[financial] excel={excel.name} -> {out}")
    extractor = FinancialDataExtractor()
    extractor.extract_financial_data_fast(
        excel_file_path=str(excel),
        output_file=str(out),
        resume=resume,
    )
    # 同步到 processed
    for name in [
        "financial_indicators_robust.csv",
        "financial_indicators_robust_raw.csv",
    ]:
        src = REBUILD_DIR / name
        if src.exists():
            shutil.copy2(src, PROCESSED_DIR / name)
            print(f"已更新 processed/{name}")
    return out


def step_kmv(resume: bool = True) -> Path:
    from kmv import KMVAnalyzer

    REBUILD_DIR.mkdir(parents=True, exist_ok=True)
    excel = find_raw_excel()
    out = REBUILD_DIR / "kmv_analysis_results.csv"
    print(f"[kmv] excel={excel.name} -> {out}")
    analyzer = KMVAnalyzer()
    analyzer.analyze_supply_chain_kmv(
        excel_file_path=str(excel),
        output_file=str(out),
        resume=resume,
    )
    for name in ["kmv_analysis_results.csv", "kmv_analysis_results_raw.csv"]:
        src = REBUILD_DIR / name
        if src.exists():
            shutil.copy2(src, PROCESSED_DIR / name)
            print(f"已更新 processed/{name}")
    return out


def _norm_symbol(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6) if s.isdigit() else s


def step_edges() -> Path:
    """从整合供应链表重建边表（不调用 Tushare）。"""
    excel = find_raw_excel()
    df = pd.read_excel(excel, sheet_name="供应链网络数据")
    rows = []
    for _, r in df.iterrows():
        year = int(r["Year"])
        core = _norm_symbol(r["Symbol"])
        if not core:
            continue
        if pd.notna(r.get("Supplier_Symbol")) and str(r.get("Supplier_Symbol")).strip():
            src = _norm_symbol(r["Supplier_Symbol"])
            if src:
                rows.append(
                    {
                        "source": src,
                        "target": core,
                        "weight": r.get("Purchase_Amount", 1.0),
                        "relationship": "supply",
                        "proportion": r.get("Purchase_Proportion"),
                        "year": year,
                    }
                )
        if pd.notna(r.get("Customer_Symbol")) and str(r.get("Customer_Symbol")).strip():
            dst = _norm_symbol(r["Customer_Symbol"])
            if dst:
                rows.append(
                    {
                        "source": core,
                        "target": dst,
                        "weight": r.get("Sales_Amount", 1.0),
                        "relationship": "sale",
                        "proportion": r.get("Sales_Proportion"),
                        "year": year,
                    }
                )
    edges = pd.DataFrame(rows)
    # 去重：同 year-source-target-relationship 保留权重大者
    if len(edges):
        edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0)
        edges = (
            edges.sort_values("weight", ascending=False)
            .drop_duplicates(["year", "source", "target", "relationship"], keep="first")
            .reset_index(drop=True)
        )
    REBUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_xlsx = REBUILD_DIR / "combined_edges.xlsx"
    out_parquet = REBUILD_DIR / "edges.parquet"
    edges.to_excel(out_xlsx, index=False)
    edges.to_parquet(out_parquet, index=False)
    shutil.copy2(out_xlsx, PROCESSED_DIR / "combined_edges.xlsx")
    shutil.copy2(out_parquet, PROCESSED_DIR / "edges.parquet")
    print(f"[edges] {len(edges)} 条边 -> processed/combined_edges.xlsx + edges.parquet")
    return out_xlsx


def step_export() -> None:
    """从重建/已有 csv 导出 nodes.parquet / labels.parquet。"""
    fin_path = PROCESSED_DIR / "financial_indicators_robust_raw.csv"
    if not fin_path.exists():
        fin_path = PROCESSED_DIR / "financial_indicators_robust.csv"
    kmv_path = PROCESSED_DIR / "kmv_analysis_results_raw.csv"
    if not kmv_path.exists():
        kmv_path = PROCESSED_DIR / "kmv_analysis_results.csv"

    if not fin_path.exists():
        raise FileNotFoundError("缺少财务特征文件，请先跑 --step financial")
    if not kmv_path.exists():
        raise FileNotFoundError("缺少 KMV 文件，请先跑 --step kmv")

    fin = pd.read_csv(fin_path)
    kmv = pd.read_csv(kmv_path)
    fin["symbol"] = fin["symbol"].map(_norm_symbol)
    kmv["symbol"] = kmv["symbol"].map(_norm_symbol)

    # 数值列尽量转回 float（formatted csv 里可能是字符串）
    for col in fin.columns:
        if col in ("symbol", "ts_code", "data_status"):
            continue
        fin[col] = pd.to_numeric(fin[col], errors="ignore")

    if "default_probability" in kmv.columns:
        def _prob(x):
            if pd.isna(x):
                return None
            if isinstance(x, str) and "%" in x:
                try:
                    return float(x.strip("%")) / 100.0
                except Exception:
                    return None
            return pd.to_numeric(x, errors="coerce")

        kmv["y"] = kmv["default_probability"].map(_prob)
    else:
        raise ValueError("KMV 结果缺少 default_probability 列")

    nodes = fin.copy()
    labels = kmv[["symbol", "year", "y"]].copy()
    if "risk_rating" in kmv.columns:
        labels["risk_rating"] = kmv["risk_rating"]
    labels["y_type"] = "kmv_edf"

    nodes_out = PROCESSED_DIR / "nodes.parquet"
    labels_out = PROCESSED_DIR / "labels.parquet"
    nodes.to_parquet(nodes_out, index=False)
    labels.to_parquet(labels_out, index=False)
    print(f"[export] nodes={len(nodes)} -> {nodes_out.name}")
    print(f"[export] labels={len(labels)} -> {labels_out.name}")
    if (PROCESSED_DIR / "edges.parquet").exists():
        print("[export] edges.parquet 已存在")
    else:
        print("[export] 提示: 尚未生成 edges.parquet，可运行 --step edges")


def main():
    parser = argparse.ArgumentParser(description="重建 FIN 数据集（Tushare）")
    parser.add_argument(
        "--step",
        choices=["all", "financial", "kmv", "edges", "export", "backup"],
        default="all",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有 rebuild 结果，从头拉取（仍会先备份 processed）",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="不备份 processed 旧文件",
    )
    args = parser.parse_args()
    resume = not args.no_resume

    setup_env()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REBUILD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ROOT={ROOT}")
    print(f"raw excel={find_raw_excel()}")
    print(f"resume={resume}")

    if args.step in ("all", "backup") and not args.skip_backup:
        backup_processed()
    if args.step == "backup":
        return

    if args.step in ("all", "edges"):
        step_edges()
    if args.step in ("all", "financial"):
        step_financial(resume=resume)
    if args.step in ("all", "kmv"):
        step_kmv(resume=resume)
    if args.step in ("all", "export"):
        # all 时若 kmv/fin 刚跑完则导出；单独 export 也可
        try:
            step_export()
        except FileNotFoundError as e:
            if args.step == "export":
                raise
            print(f"[export] 跳过: {e}")

    print("\n完成。")


if __name__ == "__main__":
    main()
