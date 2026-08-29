"""模型产物接口：/api/model/metrics、/api/model/eval-log、/api/predictions/*。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from server.services.data import (load_run_csv, load_run_json, resolve_run,
                                  to_records)

router = APIRouter()


def _resolve_or_404(run: str | None) -> str:
    try:
        return resolve_run(run)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


def _load_or_404(loader, run: str, filename: str):
    try:
        return loader(run, filename)
    except FileNotFoundError as e:
        raise HTTPException(404, f"{run} 缺少 {filename}")


def _prob_to_rating(prob) -> str | None:
    if prob is None:
        return None
    thresholds = [(0.01, "AAA"), (0.05, "AA"), (0.1, "A"), (0.2, "BBB"),
                  (0.3, "BB"), (0.4, "B"), (0.6, "CCC")]
    for t, r in thresholds:
        if prob < t:
            return r
    return "D"


@router.get("/model/runs")
def model_runs():
    from server.services.data import list_runs

    runs = list_runs()
    return {"runs": runs, "latest": runs[0] if runs else None}


@router.get("/model/experiments")
def model_experiments():
    """汇总各 run 的关键配置与指标（读每个 run 的 metrics.json，比读 experiments_log 更稳）。"""
    from server.services.data import list_runs

    keys = ["temporal_encoder", "label_scheme", "label_transform", "graph_scheme",
            "hidden_dim", "epochs", "lr", "dropout",
            "r2", "r2_logit", "spearman", "ic", "auc", "ks", "n_default",
            "mse", "mae", "rating_accuracy", "n_test"]
    rows = []
    for rid in list_runs():
        try:
            m = load_run_json(rid, "metrics.json")
        except FileNotFoundError:
            continue
        row = {"run_id": rid}
        for k in keys:
            if k in m and m[k] is not None:
                row[k] = m[k]
        rows.append(row)
    return {"runs": rows}


@router.get("/model/metrics")
def model_metrics(run: str | None = Query(None)):
    rid = _resolve_or_404(run)
    metrics = _load_or_404(load_run_json, rid, "metrics.json")
    return {"run": rid, **metrics}


@router.get("/model/eval-log")
def model_eval_log(run: str | None = Query(None)):
    rid = _resolve_or_404(run)
    df = _load_or_404(load_run_csv, rid, "eval_log.csv")
    cols = [c for c in ["epoch", "mse", "mae", "r2", "r2_logit", "spearman", "ic",
                        "auc", "ks", "rating_accuracy"] if c in df.columns]
    return {"run": rid, "points": to_records(df[cols])}


@router.get("/inference")
def inference(run: str | None = Query(None),
              year: int = Query(..., ge=1990, le=2100),
              top: int = Query(50, ge=1, le=1000)):
    """实时推理：所选 run 的 checkpoint 对目标年（用其前 3 年特征）前向预测 TopN。"""
    rid = _resolve_or_404(run)
    try:
        from server.services.inference import predict_year

        return predict_year(rid, year, top)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/predictions/test")
def predictions_test(run: str | None = Query(None),
                     limit: int = Query(0, ge=0, description="0=全部")):
    rid = _resolve_or_404(run)
    df = _load_or_404(load_run_csv, rid, "test_predictions.csv")
    keep = [c for c in ["symbol", "prediction_year", "actual_probability",
                        "predicted_probability", "actual_rating",
                        "predicted_rating"] if c in df.columns]
    df = df[keep]
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    # 高风险优先
    if "actual_probability" in df.columns:
        df = df.sort_values("actual_probability", ascending=False)
    total = int(len(df))
    if limit > 0:
        df = df.head(limit)
    return {"run": rid, "total": total, "items": to_records(df)}


@router.get("/predictions/future")
def predictions_future(run: str | None = Query(None),
                       limit: int = Query(0, ge=0, description="0=全部")):
    rid = _resolve_or_404(run)
    df = _load_or_404(load_run_csv, rid, "2025_predictions.csv")
    keep = [c for c in ["symbol", "sequence_years", "prediction_year",
                        "predicted_probability"] if c in df.columns]
    df = df[keep]
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.sort_values("predicted_probability", ascending=False)
    total = int(len(df))
    if limit > 0:
        df = df.head(limit)
    items = []
    for rec in to_records(df):
        prob = rec.get("predicted_probability")
        rec["risk_rating"] = _prob_to_rating(prob)
        items.append(rec)
    return {"run": rid, "total": total, "items": items}
