"""图谱数据接口：/api/meta、/api/graph/years、/api/graph/{year}。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.services.data import list_runs, load_edges, load_labels, load_nodes

router = APIRouter()


@router.get("/meta")
def meta():
    nodes = load_nodes()
    edges = load_edges()
    labels = load_labels()
    if nodes.empty:
        raise HTTPException(503, "processed/nodes.parquet 不存在或为空，请先运行 export")
    years = sorted(edges["year"].unique().tolist()) if not edges.empty else []
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


def _label_map_for_year(labels, year: int) -> dict:
    """该年 (symbol -> {rating, prob})；labels 为空返回空表。"""
    if labels.empty or "year" not in labels.columns:
        return {}
    ly = labels[labels["year"] == year]
    out = {}
    for sym, rating, prob in zip(ly["symbol"], ly.get("risk_rating"), ly.get("default_probability")):
        item = {}
        if rating is not None and str(rating) not in ("nan", "None"):
            item["rating"] = str(rating)
        try:
            if prob is not None and str(prob) != "nan":
                item["prob"] = float(prob)
        except (TypeError, ValueError):
            pass
        if item:
            out[str(sym)] = item
    return out


@router.get("/graph/years")
def graph_years():
    edges = load_edges()
    if edges.empty:
        return {"years": []}
    years = []
    for year, grp in edges.groupby("year"):
        n_nodes = int(len(set(grp["source"]) | set(grp["target"])))
        years.append({"year": int(year), "n_edges": int(len(grp)), "n_nodes": n_nodes})
    return {"years": sorted(years, key=lambda x: x["year"])}


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
    years = sorted(sub["year"].unique().tolist())
    per_year = {int(k): int(v) for k, v in sub.groupby("year").size().items()}
    return {"symbol": sym, "years": years, "latest": years[-1],
            "edges_by_year": per_year}


@router.get("/graph/{year}")
def graph_by_year(year: int):
    edges = load_edges()
    if edges.empty:
        raise HTTPException(503, "processed/edges.parquet 不存在或为空，请先运行 export")
    e = edges[edges["year"] == year]
    if e.empty:
        raise HTTPException(404, f"{year} 年无供应链边数据")

    labels = load_labels()
    label_map = _label_map_for_year(labels, year)

    nodes = []
    for sym in sorted(set(e["source"]) | set(e["target"])):
        info = label_map.get(sym, {})
        nodes.append({"id": sym, **info})

    links = []
    for row in e.itertuples(index=False):
        link = {
            "source": row.source,
            "target": row.target,
            "relationship": str(row.relationship),
            "weight": float(row.weight) if row.weight == row.weight else 1.0,
        }
        prop = getattr(row, "proportion", None)
        if prop is not None and prop == prop:  # 非 NaN
            link["proportion"] = float(prop)
        links.append(link)

    return {"year": int(year), "nodes": nodes, "links": links}
