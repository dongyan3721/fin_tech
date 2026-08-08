"""从 processed parquet 构造 (N,3,10) 序列样本、时间切分与样本级图。

与 legacy 对齐：
- 连续 4 年，前 3 年特征 -> 第 4 年 KMV 标签；
- 训练集预测年 2007-2020，测试集 2021-2024；
- 2025 推演用 2022-2024 特征；
- 图：batch 行号即节点号，把公司级边展开到样本级，log1p+归一化权重，
  样本级边数不足 min_edges 时退回简化卷积（返回 None）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.current.config import CONFIG


@dataclass
class SplitData:
    X: np.ndarray                        # (N, T, F)
    y: np.ndarray                        # (N,)
    companies: List[str]
    pred_years: List[int]
    sequence_years: List[list]


@dataclass
class FutureData:
    X: np.ndarray
    companies: List[str]
    sequence_years: List[list]


@dataclass
class DatasetBundle:
    train: SplitData
    test: SplitData
    future: Optional[FutureData]
    nodes: pd.DataFrame
    labels: pd.DataFrame
    edges: pd.DataFrame
    feature_columns: List[str] = field(default_factory=lambda: list(CONFIG.feature_columns))


def _load() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_parquet(CONFIG.nodes_parquet)
    labels = pd.read_parquet(CONFIG.labels_parquet)
    edges = pd.read_parquet(CONFIG.edges_parquet) if CONFIG.edges_parquet.exists() else pd.DataFrame(
        columns=["source", "target", "weight", "relationship", "proportion", "year"])
    nodes["symbol"] = nodes["symbol"].astype(str)
    labels["symbol"] = labels["symbol"].astype(str)
    return nodes, labels, edges


def build_dataset() -> DatasetBundle:
    nodes, labels, edges = _load()
    feats = [c for c in CONFIG.feature_columns if c in nodes.columns]
    label_col = CONFIG.label_column
    seq_len = CONFIG.seq_len

    common = set(nodes["symbol"].unique()) & set(labels["symbol"].unique())

    sequences, ys, companies, pred_years, seq_years = [], [], [], [], []
    for company in common:
        cf = nodes[nodes["symbol"] == company].sort_values("year")
        ck = labels[labels["symbol"] == company].sort_values("year")
        fyears = sorted(cf["year"].unique())
        kyears = set(ck["year"].unique())
        for i in range(len(fyears) - seq_len):
            window = fyears[i:i + seq_len + 1]
            if any(window[j + 1] - window[j] != 1 for j in range(seq_len)):
                continue
            pred_year = window[seq_len]
            if pred_year not in kyears:
                continue
            fseq, yseq = [], []
            ok = True
            for y in window[:seq_len]:
                row = cf[cf["year"] == y]
                if row.empty:
                    ok = False
                    break
                fseq.append(row[feats].fillna(0).values[0])
                yseq.append(int(y))
            if not ok or len(fseq) != seq_len:
                continue
            lab = ck[ck["year"] == pred_year][label_col]
            if lab.empty or pd.isna(lab.iloc[0]):
                continue
            sequences.append(fseq)
            ys.append(float(lab.iloc[0]))
            companies.append(company)
            pred_years.append(int(pred_year))
            seq_years.append(yseq)

    sequences = np.array(sequences, dtype=np.float64) if sequences else np.empty((0, seq_len, len(feats)))
    ys = np.array(ys, dtype=np.float64)

    from src.current.config import TRAIN_PRED_YEARS, TEST_PRED_YEARS
    tr_lo, tr_hi = TRAIN_PRED_YEARS
    te_lo, te_hi = TEST_PRED_YEARS

    tr_idx = [i for i, py in enumerate(pred_years) if tr_lo <= py <= tr_hi]
    te_idx = [i for i, py in enumerate(pred_years) if te_lo <= py <= te_hi]

    def _subset(idx: List[int]) -> SplitData:
        return SplitData(
            X=sequences[idx] if idx else np.empty((0, seq_len, len(feats))),
            y=ys[idx] if idx else np.empty((0,)),
            companies=[companies[i] for i in idx],
            pred_years=[pred_years[i] for i in idx],
            sequence_years=[seq_years[i] for i in idx],
        )

    train = _subset(tr_idx)
    test = _subset(te_idx)
    future = _build_future(nodes, feats)

    print(f"[dataset] 样本总数 {len(sequences)}；训练 {len(train.X)}，测试 {len(test.X)}，"
          f"2025推演 {0 if future is None else len(future.X)}")
    return DatasetBundle(train=train, test=test, future=future,
                         nodes=nodes, labels=labels, edges=edges, feature_columns=feats)


def _build_future(nodes: pd.DataFrame, feats: List[str]) -> Optional[FutureData]:
    from src.current.config import FUTURE_INPUT_YEARS
    years = list(FUTURE_INPUT_YEARS)
    comps, seqs, syears = [], [], []
    for company in nodes["symbol"].unique():
        cf = nodes[nodes["symbol"] == company]
        avail = set(cf["year"].unique())
        if not all(y in avail for y in years):
            continue
        fseq = []
        for y in years:
            row = cf[cf["year"] == y]
            if row.empty:
                break
            fseq.append(row[feats].fillna(0).values[0])
        if len(fseq) == len(years):
            comps.append(company)
            seqs.append(fseq)
            syears.append(years)
    if not comps:
        return None
    return FutureData(X=np.array(seqs, dtype=np.float64), companies=comps, sequence_years=syears)


def build_graph_by_pred_year(edges: pd.DataFrame, company_names: List[str],
                             pred_years: List[int], lag: int = 1,
                             min_edges: Optional[int] = None
                             ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """按预测年建「分年块对角图」（推荐方案）。

    每个样本 i 有预测年 p_i，其图结构取「p_i - lag」年的供应链边（预测时点可得的最新结构）。
    只有 **同一预测年** 且公司在该年边表中相连的样本之间才连边，不同预测年的样本互不相连
    （块对角）。这样训练集和测试集用的是同一套「按预测年取边」的口径，避免 legacy 方案里
    训练用某一年、测试用另一年造成的图结构分布漂移。

    返回样本级 (edge_index, edge_weight)；总边数不足 min_edges 时返回 (None, None) 退回简化卷积。
    """
    if edges is None or edges.empty or "year" not in edges.columns:
        return None, None
    if min_edges is None:
        min_edges = CONFIG.model.min_edges_for_gcn

    from collections import defaultdict
    year_to_indices: dict = defaultdict(list)
    for idx, py in enumerate(pred_years):
        year_to_indices[int(py)].append(idx)

    pairs, weights = [], []
    edges_by_year = {int(y): grp for y, grp in edges.groupby("year")}
    for py, idxs in year_to_indices.items():
        graph_year = py - lag
        ge = edges_by_year.get(graph_year)
        if ge is None or ge.empty:
            continue
        sym_to_idx: dict = {}
        for i in idxs:
            sym_to_idx.setdefault(company_names[i], []).append(i)
        for _, row in ge.iterrows():
            s, t = row["source"], row["target"]
            if s not in sym_to_idx or t not in sym_to_idx:
                continue
            w = float(row["weight"]) if pd.notna(row.get("weight", 1.0)) else 1.0
            for si in sym_to_idx[s]:
                for ti in sym_to_idx[t]:
                    if si != ti:
                        pairs.append([si, ti])
                        weights.append(w)

    if len(pairs) < min_edges:
        return None, None

    w = np.log1p(np.maximum(np.asarray(weights, dtype=np.float64), 0.0))
    w = w / (w.max() if w.max() > 0 else 1.0)
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(w, dtype=torch.float)
    n_pred_years = len({int(p) for p in pred_years})
    print(f"[graph] 按预测年建图(lag={lag}): 覆盖 {n_pred_years} 个预测年, 样本级边={edge_index.shape[1]}")
    return edge_index, edge_weight


def build_graph(edges: pd.DataFrame, company_names: List[str],
                target_year: Optional[int] = None
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """[DEPRECATED] legacy 建图：整批共用「有效边最多的一年」拓扑。

    该方案会让训练集与测试集各自选到不同年份的拓扑（如训练 2013、测试 2020），
    造成 GNN 的图结构分布漂移、指标不稳。已被 ``build_graph_by_pred_year`` 取代，
    仅保留用于对照/复现 legacy 行为（config.model.graph_scheme="densest_legacy"）。
    """
    import warnings
    warnings.warn(
        "build_graph（整批最密年方案）已废弃，请使用 build_graph_by_pred_year；"
        "如需复现 legacy 行为，设 config.model.graph_scheme='densest_legacy'。",
        DeprecationWarning, stacklevel=2,
    )
    if edges is None or edges.empty or "year" not in edges.columns:
        return None, None

    nodes = list(company_names)
    node_set = set(nodes)
    sym_to_idx: dict = {}
    for idx, sym in enumerate(nodes):
        sym_to_idx.setdefault(sym, []).append(idx)

    # 选择建图年份：优先 target_year（若其有效边>0），否则选有效边最多的年
    year_counts = []
    for y, grp in edges.groupby("year"):
        valid = (grp["source"].isin(node_set) & grp["target"].isin(node_set)).sum()
        year_counts.append((int(y), int(valid)))
    if not year_counts:
        return None, None
    if target_year is not None and any(y == target_year for y, _ in year_counts):
        spec = next(c for y, c in year_counts if y == target_year)
        graph_year = target_year if spec > 0 else max(year_counts, key=lambda x: (x[1], x[0]))[0]
    else:
        graph_year = max(year_counts, key=lambda x: (x[1], x[0]))[0]

    ge = edges[edges["year"] == graph_year]
    pairs, weights = [], []
    for _, row in ge.iterrows():
        s, t = row["source"], row["target"]
        if s not in sym_to_idx or t not in sym_to_idx:
            continue
        w = float(row["weight"]) if pd.notna(row.get("weight", 1.0)) else 1.0
        for si in sym_to_idx[s]:
            for ti in sym_to_idx[t]:
                if si != ti:
                    pairs.append([si, ti])
                    weights.append(w)

    if len(pairs) < CONFIG.model.min_edges_for_gcn:
        return None, None

    w = np.log1p(np.maximum(np.asarray(weights, dtype=np.float64), 0.0))
    w = w / (w.max() if w.max() > 0 else 1.0)
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(w, dtype=torch.float)
    print(f"[graph] 建图年={graph_year}, 节点={len(nodes)}, 样本级边={edge_index.shape[1]}")
    return edge_index, edge_weight
