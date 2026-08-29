"""实时推理服务：加载指定 run 的 checkpoint，对任意目标年做前向预测。

- 输入特征：nodes.parquet 中 (year-3, year-2, year-1) 连续三年财务特征，
  用 checkpoint 内保存的 scaler 归一化；
- 图结构：按「目标年 - graph_lag」年的供应链边建分年块对角图（与训练口径一致）；
- 输出：违约概率/风险评分降序 TopN；
- 缓存：模型按 run 缓存（首次调用含 torch 导入约 10s，之后毫秒级），
  结果按 (run, year, nodes_mtime) 缓存，重训导出后自动失效。
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import torch

from src.current.config import CONFIG
from src.current.models.tgc import TGCModel, ModelConfig
from src.current.train.dataset import build_graph_by_pred_year

_lock = threading.Lock()
_models: dict[str, dict] = {}
_results: dict[tuple, tuple] = {}


def _prob_to_rating(p: float) -> str:
    thresholds = [(0.01, "AAA"), (0.05, "AA"), (0.1, "A"), (0.2, "BBB"),
                  (0.3, "BB"), (0.4, "B"), (0.6, "CCC")]
    for t, r in thresholds:
        if p < t:
            return r
    return "D"


def _load_model(run: str) -> dict:
    with _lock:
        entry = _models.get(run)
    if entry is not None:
        return entry
    ckpt_path = CONFIG.outputs_dir / run / "model_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{run} 缺少 model_checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ModelConfig(
        temporal_encoder=ckpt["model_config"]["temporal_encoder"],
        hidden_dim=ckpt["model_config"]["hidden_dim"],
        dropout=ckpt["model_config"]["dropout"],
        temporal_kernel=ckpt["model_config"]["temporal_kernel"],
    )
    model = TGCModel(input_dim=ckpt["input_dim"], cfg=mc,
                     use_gcn=ckpt["use_gcn"], final_activation=ckpt["final_activation"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    entry = {
        "model": model,
        "mean": np.asarray(ckpt["scaler_mean"], dtype=np.float64),
        "scale": np.asarray(ckpt["scaler_scale"], dtype=np.float64),
        "feature_columns": ckpt["feature_columns"],
        "final_activation": ckpt["final_activation"],
    }
    with _lock:
        _models[run] = entry
    return entry


def predict_year(run: str, year: int, top: int = 50) -> dict:
    entry = _load_model(run)
    nodes_mtime = CONFIG.nodes_parquet.stat().st_mtime if CONFIG.nodes_parquet.exists() else 0.0
    key = (run, int(year), nodes_mtime)
    with _lock:
        cached = _results.get(key)
    if cached is not None:
        n, items = cached
        return {"run": run, "year": int(year), "n_companies": n, "items": items[:top]}

    nodes = pd.read_parquet(CONFIG.nodes_parquet)
    nodes["symbol"] = nodes["symbol"].astype(str)
    nodes["year"] = nodes["year"].astype(int)
    feats = [c for c in entry["feature_columns"] if c in nodes.columns]
    input_years = [year - 3, year - 2, year - 1]

    companies, seqs = [], []
    for company, grp in nodes.groupby("symbol"):
        avail = set(grp["year"].unique())
        if not all(y in avail for y in input_years):
            continue
        rows, ok = [], True
        for y in input_years:
            row = grp[grp["year"] == y]
            if row.empty:
                ok = False
                break
            rows.append(row[feats].fillna(0).values[0])
        if ok:
            companies.append(company)
            seqs.append(rows)
    if not companies:
        return {"run": run, "year": int(year), "n_companies": 0, "items": []}

    X = np.asarray(seqs, dtype=np.float64)
    n, t, f = X.shape
    Xs = ((X.reshape(-1, f) - entry["mean"]) / entry["scale"]).reshape(n, t, f)
    x = torch.tensor(Xs, dtype=torch.float)

    edges = pd.read_parquet(CONFIG.edges_parquet)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["year"] = edges["year"].astype(int)
    ei, ew = build_graph_by_pred_year(edges, companies, [year] * n, lag=1)

    with torch.no_grad():
        raw = entry["model"](x, ei, ew).cpu().numpy().flatten()
    prob = 1.0 / (1.0 + np.exp(-raw)) if entry["final_activation"] == "identity" else raw

    order = np.argsort(-prob)
    items = [{"rank": rank,
              "symbol": companies[idx],
              "predicted_probability": float(prob[idx]),
              "risk_rating": _prob_to_rating(float(prob[idx]))}
             for rank, idx in enumerate(order, 1)]

    with _lock:
        _results[key] = (n, items)
    return {"run": run, "year": int(year), "n_companies": n, "items": items[:top]}
