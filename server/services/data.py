"""parquet/csv 读取与 mtime 缓存（进程内单例，重训后自动失效）。"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.current.config import CONFIG

_cache: Dict[str, Tuple[float, object]] = {}
_lock = threading.Lock()


def _cached_read(path: Path, loader):
    key = str(path)
    mtime = path.stat().st_mtime if path.exists() else -1.0
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    df = loader() if mtime >= 0 else pd.DataFrame()
    with _lock:
        _cache[key] = (mtime, df)
    return df


def load_nodes() -> pd.DataFrame:
    return _cached_read(CONFIG.nodes_parquet, lambda: pd.read_parquet(CONFIG.nodes_parquet))


def load_edges() -> pd.DataFrame:
    df = _cached_read(CONFIG.edges_parquet, lambda: pd.read_parquet(CONFIG.edges_parquet))
    if not df.empty:
        df = df.copy()
        df["source"] = df["source"].astype(str)
        df["target"] = df["target"].astype(str)
        df["year"] = df["year"].astype(int)
    return df


def load_labels() -> pd.DataFrame:
    return _cached_read(CONFIG.labels_parquet, lambda: pd.read_parquet(CONFIG.labels_parquet))


def resolve_run(run: str | None) -> str:
    """解析 run 名；None 取最新。无效则抛 KeyError/RuntimeError。"""
    runs = list_runs()
    if run:
        if run not in runs:
            raise KeyError(f"run 不存在: {run}")
        return run
    if not runs:
        raise RuntimeError("outputs/ 下没有任何训练产物，请先训练")
    return runs[0]


def load_run_json(run: str, filename: str) -> dict:
    import json

    path = CONFIG.outputs_dir / run / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")
    return _cached_read(path, lambda: json.loads(path.read_text(encoding="utf-8")))


def load_run_csv(run: str, filename: str) -> pd.DataFrame:
    path = CONFIG.outputs_dir / run / filename

    def _load():
        # symbol 列必须是字符串，否则 read_csv 会把 000672 推断成整数丢前导零
        return pd.read_csv(path, dtype={"symbol": str})

    return _cached_read(path, _load)


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON 安全 records：NaN→None、numpy 标量→原生类型。"""
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                clean[k] = None
                continue
            try:
                if pd.isna(v):
                    clean[k] = None
                    continue
            except (TypeError, ValueError):
                pass
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            clean[k] = v
        out.append(clean)
    return out


def list_runs() -> list[str]:
    """outputs/ 下含 metrics.json 或 test_predictions.csv 的 run 目录名，新→旧。"""
    out_dir = CONFIG.outputs_dir
    if not out_dir.exists():
        return []
    runs = [p.name for p in sorted(out_dir.iterdir(), reverse=True)
            if p.is_dir() and ((p / "metrics.json").exists() or (p / "test_predictions.csv").exists())]
    return runs
