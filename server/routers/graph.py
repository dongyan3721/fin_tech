"""图谱数据接口：/api/meta、/api/graph/years、/api/graph/{year}、/api/graph/locate/{symbol}。

图数据由 services/graph_cache 启动预加载（毫秒级命中），Neo4j 依赖已移除。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.services.data import list_runs, load_edges, load_labels, load_nodes
from server.services.graph_cache import get_graph_years, get_year_graph

router = APIRouter()


@router.get("/meta")
def meta():
    nodes = load_nodes()
    edges = load_edges()
    labels = load_labels()
    if nodes.empty:
        raise HTTPException(503, "processed/nodes.parquet 不存在或为空，请先运行 export")
    years = sorted(int(y) for y in edges["year"].unique()) if not edges.empty else []
    runs = list_runs()
    return {
        "n_companies": int(nodes["symbol"].nunique()),
        "n_node_rows": int(len(nodes)),
        "feature_years": [int(nodes["year"].min()), int(nodes["year"].max())],
        "n_edges": int(len(edges)),
        "graph_years": years,
        "has_hybrid_label": bool(not labels.empty and "st_level" in labels.columns),
        "runs": runs,
        "latest_run": runs[0] if runs else None,
    }


@router.get("/graph/years")
def graph_years():
    return {"years": get_graph_years()}


@router.get("/graph/locate/{symbol}")
def graph_locate(symbol: str):
    """定位某公司在哪些年份有供应链边（供图谱页搜索直达）。"""
    edges = load_edges()
    if edges.empty:
        raise HTTPException(503, "processed/edges.parquet 不存在或为空")
    sym = str(symbol).zfill(6)
    sub = edges[(edges["source"] == sym) | (edges["target"] == sym)]
    if sub.empty:
        return {"symbol": sym, "years": [], "latest": None, "edges_by_year": {}}
    years = sorted(int(y) for y in sub["year"].unique())
    per_year = {int(k): int(v) for k, v in sub.groupby("year").size().items()}
    return {"symbol": sym, "years": years, "latest": years[-1],
            "edges_by_year": per_year}


@router.get("/graph/{year}")
def graph_by_year(year: int):
    edges = load_edges()
    if edges.empty:
        raise HTTPException(503, "processed/edges.parquet 不存在或为空，请先运行 export")
    g = get_year_graph(year)
    if not g["links"]:
        raise HTTPException(404, f"{year} 年无供应链边数据")
    return g
