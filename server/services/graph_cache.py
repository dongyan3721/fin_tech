"""年度图缓存：启动时预加载所有年份的供应链图，查询毫秒级。

数据全部来自 processed parquet（Neo4j 依赖已移除）。edges/labels 文件 mtime
变化后缓存自动失效，下次访问惰性重建。预加载过程带日志埋点，FastAPI 启动时打印。
"""
from __future__ import annotations

import threading
import time

from server.services.data import load_edges, load_labels
from src.current.config import CONFIG

_lock = threading.Lock()
_cache: dict[int, dict] = {}
_sig: tuple[float, float] | None = None  # (edges_mtime, labels_mtime)


def _signature() -> tuple[float, float]:
    e = CONFIG.edges_parquet.stat().st_mtime if CONFIG.edges_parquet.exists() else 0.0
    l = CONFIG.labels_parquet.stat().st_mtime if CONFIG.labels_parquet.exists() else 0.0
    return (e, l)


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


def _build_year_graph(edges, labels, year: int) -> dict:
    e = edges[edges["year"] == year]
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


def preload_all_graphs() -> dict:
    """启动时预加载所有年份的图进缓存（带日志埋点）。"""
    global _sig
    t0 = time.perf_counter()
    print("[preload] 开始预加载年度图缓存 ...", flush=True)
    edges = load_edges()
    if edges.empty:
        print("[preload] processed/edges.parquet 为空，跳过预加载（请先运行 cli export）", flush=True)
        return {"years": 0, "seconds": 0.0}
    labels = load_labels()
    years = sorted(int(y) for y in edges["year"].unique())
    print(f"[preload] 边 {len(edges)} 条（{years[0]}–{years[-1]} 共 {len(years)} 年），"
          f"标签 {len(labels)} 行", flush=True)

    built = 0
    with _lock:
        _cache.clear()
        for y in years:
            ty = time.perf_counter()
            g = _build_year_graph(edges, labels, y)
            _cache[y] = g
            built += 1
            print(f"[preload]   {y} 年: {len(g['nodes'])} 节点 / {len(g['links'])} 边"
                  f"（{(time.perf_counter() - ty) * 1000:.1f} ms）", flush=True)
        _sig = _signature()

    dt = time.perf_counter() - t0
    print(f"[preload] 完成：{built} 个年度图全部就绪，总耗时 {dt:.2f}s，"
          f"后续 /api/graph 查询毫秒级", flush=True)
    return {"years": built, "seconds": round(dt, 3)}


def _ensure_fresh() -> None:
    """parquet mtime 变化则使缓存失效（惰性重建）。"""
    global _sig
    with _lock:
        if _sig is not None and _sig != _signature():
            _cache.clear()
            _sig = None


def get_year_graph(year: int) -> dict:
    """取某年图：优先命中缓存，未命中（新年份/缓存失效）惰性构建。"""
    _ensure_fresh()
    with _lock:
        g = _cache.get(year)
    if g is not None:
        return g
    edges = load_edges()
    labels = load_labels()
    g = _build_year_graph(edges, labels, year)
    with _lock:
        _cache[year] = g
        _sig = _signature()
    return g


def get_graph_years() -> list[dict]:
    """各年份的节点/边规模（供图谱页年份滑杆）。"""
    _ensure_fresh()
    with _lock:
        if _cache:
            return [{"year": y,
                     "n_edges": len(_cache[y]["links"]),
                     "n_nodes": len(_cache[y]["nodes"])}
                    for y in sorted(_cache.keys())]
    # 缓存为空（未预加载/数据为空）时从边表直接统计
    edges = load_edges()
    if edges.empty:
        return []
    out = []
    for year, grp in edges.groupby("year"):
        out.append({"year": int(year),
                    "n_edges": int(len(grp)),
                    "n_nodes": int(len(set(grp["source"]) | set(grp["target"])))})
    return sorted(out, key=lambda x: x["year"])


def cache_stats() -> dict:
    """健康检查用：缓存年份个数与预加载状态。"""
    with _lock:
        return {"graph_years_cached": len(_cache),
                "graph_preloaded": _sig is not None and len(_cache) > 0}
