"""推理脚本：加载训练好的模型，预测指定年份的企业违约概率和评级。

用法示例:
    # 预测 2025 年（使用最新模型）
    python scripts/predict.py --year 2025

    # 指定模型 checkpoint 路径
    python scripts/predict.py --year 2025 --checkpoint repository/outputs/current_tgc_20260809_013010/model_checkpoint.pt

    # 输出 Top 50 高风险公司
    python scripts/predict.py --year 2025 --top 50

    # 输出到 CSV 文件
    python scripts/predict.py --year 2025 --output predictions_2025.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.current.config import CONFIG
from src.current.models.tgc import TGCModel, ModelConfig
from src.current.train.dataset import build_graph_by_pred_year
from src.current.transform.symbols import normalize_symbol


def _logit(x: np.ndarray) -> np.ndarray:
    xc = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(xc / (1 - xc))


def _prob(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _risk_rating(edf: float) -> str:
    thresholds = [(0.01, "AAA"), (0.05, "AA"), (0.1, "A"), (0.2, "BBB"),
                  (0.3, "BB"), (0.4, "B"), (0.6, "CCC")]
    for t, r in thresholds:
        if edf < t:
            return r
    return "D"


def find_latest_checkpoint() -> Path:
    """找到最新的 model checkpoint。"""
    outputs_dir = CONFIG.outputs_dir
    checkpoints = sorted(outputs_dir.glob("current_tgc_*/model_checkpoint.pt"),
                         key=lambda p: p.parent.name, reverse=True)
    if not checkpoints:
        raise FileNotFoundError("未找到任何 model checkpoint，请先运行训练。")
    return checkpoints[0]


def load_checkpoint(ckpt_path: Path) -> dict:
    """加载 checkpoint。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[load] checkpoint: {ckpt_path}")
    print(f"  input_dim={ckpt['input_dim']}, use_gcn={ckpt['use_gcn']}, "
          f"activation={ckpt['final_activation']}")
    print(f"  feature_columns={ckpt['feature_columns']}")
    return ckpt


def build_inference_data(pred_year: int, feature_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """构造推理数据：找到有连续 3 年数据的公司的特征序列。"""
    nodes = pd.read_parquet(CONFIG.nodes_parquet)
    nodes["symbol"] = nodes["symbol"].astype(str)
    nodes["year"] = nodes["year"].astype(int)

    input_years = [pred_year - 3, pred_year - 2, pred_year - 1]
    print(f"[data] 推理输入年份: {input_years} → 预测 {pred_year}")

    feats = [c for c in feature_columns if c in nodes.columns]
    if len(feats) != len(feature_columns):
        missing = set(feature_columns) - set(feats)
        print(f"[warn] 缺少特征列: {missing}")

    companies = []
    sequences = []

    for company in nodes["symbol"].unique():
        cf = nodes[nodes["symbol"] == company].sort_values("year")
        avail_years = set(cf["year"].unique())

        if not all(y in avail_years for y in input_years):
            continue

        fseq = []
        for y in input_years:
            row = cf[cf["year"] == y]
            if row.empty:
                break
            fseq.append(row[feats].fillna(0).values[0])

        if len(fseq) == len(input_years):
            companies.append(company)
            sequences.append(fseq)

    X = np.array(sequences, dtype=np.float64)
    print(f"[data] 有效样本: {len(companies)} 家公司")
    return X, companies


def run_inference(ckpt: dict, X: np.ndarray, companies: list[str],
                  edges: pd.DataFrame, pred_year: int,
                  logit_space: bool = True) -> pd.DataFrame:
    """运行推理。"""
    input_dim = ckpt["input_dim"]
    use_gcn = ckpt["use_gcn"]
    final_act = ckpt["final_activation"]
    model_cfg = ckpt["model_config"]

    mc = ModelConfig(
        temporal_encoder=model_cfg["temporal_encoder"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        temporal_kernel=model_cfg["temporal_kernel"],
    )

    model = TGCModel(input_dim=input_dim, cfg=mc, use_gcn=use_gcn,
                     final_activation=final_act)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler_mean = np.array(ckpt["scaler_mean"])
    scaler_scale = np.array(ckpt["scaler_scale"])
    n, t, f = X.shape
    X_scaled = (X.reshape(-1, f) - scaler_mean) / scaler_scale
    X_scaled = X_scaled.reshape(n, t, f)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float)

    pred_years = [pred_year] * n
    ei, ew = build_graph_by_pred_year(edges, companies, pred_years, lag=1)

    with torch.no_grad():
        raw = model(x_tensor, ei, ew).numpy().flatten()

    if final_act == "identity":
        prob = _prob(raw)
    else:
        prob = raw

    results = pd.DataFrame({
        "symbol": companies,
        "predicted_probability": prob,
        "risk_rating": [_risk_rating(p) for p in prob],
    })
    results = results.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TGC 企业风险推理脚本")
    p.add_argument("--year", type=int, default=2025, help="预测目标年份 (default: 2025)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="模型 checkpoint 路径 (default: 自动找最新)")
    p.add_argument("--top", type=int, default=20, help="输出 Top N 高风险公司 (default: 20)")
    p.add_argument("--output", type=str, default=None, help="输出 CSV 文件路径")
    p.add_argument("--all", action="store_true", help="输出所有公司（而非仅 Top N）")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("TGC 企业风险推理")
    print("=" * 60)
    print(f"  预测年份: {args.year}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    ckpt = load_checkpoint(ckpt_path)

    edges = pd.read_parquet(CONFIG.edges_parquet)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["year"] = edges["year"].astype(int)

    X, companies = build_inference_data(args.year, ckpt["feature_columns"])

    logit_space = ckpt.get("final_activation") == "identity"
    results = run_inference(ckpt, X, companies, edges, args.year, logit_space)

    print(f"\n预测完成: {len(results)} 家公司")
    print(f"  平均违约概率: {results['predicted_probability'].mean():.4f}")
    print(f"  最大违约概率: {results['predicted_probability'].max():.4f}")
    print(f"  最小违约概率: {results['predicted_probability'].min():.4f}")

    if args.all:
        display = results
    else:
        display = results.head(args.top)

    print(f"\n{'='*60}")
    print(f"{'排名':<4} {'代码':<8} {'违约概率':<12} {'评级':<6}")
    print("-" * 40)
    for i, row in display.iterrows():
        print(f"{i+1:<4} {row['symbol']:<8} {row['predicted_probability']:<12.6f} {row['risk_rating']:<6}")
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
