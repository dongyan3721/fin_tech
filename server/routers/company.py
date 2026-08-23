"""企业分析接口：/api/company/search、/api/company/{symbol}。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.current.config import CONFIG
from server.services.data import (load_edges, load_labels, load_nodes,
                                  load_run_csv, resolve_run, to_records)

router = APIRouter()


def _norm(sym) -> str:
    s = str(sym).strip().replace(".0", "")
    return s.zfill(6) if s.isdigit() else s


def _prob_to_rating(prob) -> str | None:
    if prob is None:
        return None
    thresholds = [(0.01, "AAA"), (0.05, "AA"), (0.1, "A"), (0.2, "BBB"),
                  (0.3, "BB"), (0.4, "B"), (0.6, "CCC")]
    for t, r in thresholds:
        if prob < t:
            return r
    return "D"


@router.get("/company/search")
def company_search(q: str = Query("")):
    q = str(q).strip()
    if not q:
        return {"items": []}
    symbols: set[str] = set()
    nodes = load_nodes()
    if not nodes.empty:
        symbols |= set(nodes["symbol"].astype(str).map(_norm))
    edges = load_edges()
    if not edges.empty:
        symbols |= set(edges["source"].astype(str).map(_norm))
        symbols |= set(edges["target"].astype(str).map(_norm))
    matched = sorted(s for s in symbols if q in s)[:20]
    return {"items": [{"symbol": s} for s in matched]}


@router.get("/company/{symbol}")
def company_detail(symbol: str, run: str | None = Query(None)):
    sym = _norm(symbol)

    financial = []
    nodes = load_nodes()
    if not nodes.empty:
        n = nodes[nodes["symbol"].astype(str).map(_norm) == sym].sort_values("year")
        cols = ["year"] + [c for c in CONFIG.feature_columns if c in n.columns]
        financial = to_records(n[cols])

    labels = []
    lab = load_labels()
    if not lab.empty:
        l = lab[lab["symbol"].astype(str).map(_norm) == sym].sort_values("year")
        lcols = ["year", "default_probability", "risk_rating"]
        for extra in ("st_level", "delisted"):
            if extra in l.columns:
                lcols.append(extra)
        labels = to_records(l[lcols])

    edges_out = []
    ed = load_edges()
    if not ed.empty:
        e = ed[(ed["source"].astype(str).map(_norm) == sym)
               | (ed["target"].astype(str).map(_norm) == sym)]
        for rec in to_records(e):
            src, tgt = _norm(rec["source"]), _norm(rec["target"])
            edges_out.append({
                "year": rec["year"],
                "relationship": str(rec["relationship"]),
                "peer": tgt if src == sym else src,
                "direction": "out" if src == sym else "in",
                "weight": rec.get("weight"),
                "proportion": rec.get("proportion"),
            })

    predictions = []
    try:
        rid = resolve_run(run)
        for fname, ycol in (("test_predictions.csv", "prediction_year"),
                            ("2025_predictions.csv", "prediction_year")):
            try:
                df = load_run_csv(rid, fname)
            except FileNotFoundError:
                continue
            sub = df[df["symbol"].astype(str).map(_norm) == sym]
            for rec in to_records(sub):
                p = rec.get("predicted_probability")
                yr = rec.get(ycol)
                predictions.append({
                    "year": yr,
                    "predicted_probability": p,
                    "predicted_rating": _prob_to_rating(p),
                })
    except (KeyError, RuntimeError):
        predictions = []
    predictions = sorted([p for p in predictions if p["year"] is not None],
                         key=lambda x: x["year"])

    if not financial and not labels and not edges_out:
        raise HTTPException(404, f"未找到公司 {sym} 的任何数据")

    return {
        "symbol": sym,
        "financial": financial,
        "labels": labels,
        "edges": edges_out,
        "predictions": predictions,
        "n_edges": len(edges_out),
    }
