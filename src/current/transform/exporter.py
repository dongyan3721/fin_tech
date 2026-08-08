"""把 interim 产物转换为训练/可视化直接可用的三张 parquet。

输出（repository/processed/）：
- nodes.parquet：symbol,year + 10 个财务特征
- edges.parquet：source,target,weight,relationship,proportion,year
- labels.parquet：symbol,year + 各 labeler 产出的标签列（至少 default_probability）
"""
from __future__ import annotations

import pandas as pd

from src.current.config import CONFIG
from src.current.transform import schema
from src.current.transform.symbols import normalize_symbol


def export_nodes() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG.financial_interim)
    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol"] != ""]
    if "data_status" in df.columns:
        df = df[~df["data_status"].astype(str).str.startswith(("failed", "error"))]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    # 保留主键 + 特征列
    cols = schema.NODE_KEYS + [c for c in CONFIG.feature_columns if c in df.columns]
    df = df[cols].drop_duplicates(subset=schema.NODE_KEYS, keep="last")
    df = schema.validate_nodes(df)
    df.to_parquet(CONFIG.nodes_parquet, index=False)
    print(f"[export] nodes: {len(df)} 行 -> {CONFIG.nodes_parquet}")
    return df


def export_edges() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG.edges_interim)
    df = df.copy()
    df["source"] = df["source"].map(normalize_symbol)
    df["target"] = df["target"].map(normalize_symbol)
    df = df[(df["source"] != "") & (df["target"] != "")]
    df = schema.validate_edges(df)
    df.to_parquet(CONFIG.edges_parquet, index=False)
    print(f"[export] edges: {len(df)} 行 -> {CONFIG.edges_parquet}")
    return df


def export_labels() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG.labels_interim)
    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol"] != ""]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df.drop_duplicates(subset=schema.LABEL_KEYS, keep="last")
    df = schema.validate_labels(df)
    df.to_parquet(CONFIG.labels_parquet, index=False)
    print(f"[export] labels: {len(df)} 行 -> {CONFIG.labels_parquet}")
    return df


def export_all() -> None:
    export_nodes()
    export_edges()
    export_labels()
