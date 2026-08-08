"""端到端编排：采集 -> 标签 -> 导出 -> 训练。

各阶段可单独调用，也可用 run_all 一键跑通。所有阶段读取 CONFIG，产出落 repository/。
"""
from __future__ import annotations

import pandas as pd

from src.current.config import CONFIG


def step_collect(resume: bool = True) -> None:
    """采集供应链边 + 财务特征 + 行情（后两者消耗 Tushare 额度）。"""
    from src.current.data import supply_chain
    from src.current.data.financial import FinancialCollector
    from src.current.data.market import MarketCollector
    from src.current.data.tushare_client import TushareClient

    print("=== [collect] 供应链边 ===")
    edges = supply_chain.collect_edges()
    edges.to_parquet(CONFIG.edges_interim, index=False)
    print(f"[collect] edges: {len(edges)} 行 -> {CONFIG.edges_interim}")

    combos = supply_chain.collect_symbol_year_universe()
    print(f"[collect] 需采集 (symbol, year) 全集: {len(combos)}")

    client = TushareClient()
    print("=== [collect] 财务特征 ===")
    FinancialCollector(client).collect(combos, resume=resume)
    print("=== [collect] 行情（KMV 所需）===")
    MarketCollector(client).collect(combos, resume=resume)


def step_edges_only() -> None:
    """仅从本地 Excel 重建边（不消耗 Tushare 额度）。"""
    from src.current.data import supply_chain
    edges = supply_chain.collect_edges()
    edges.to_parquet(CONFIG.edges_interim, index=False)
    print(f"[collect] edges: {len(edges)} 行 -> {CONFIG.edges_interim}")


def step_label() -> pd.DataFrame:
    from src.current.labels.base import generate_labels
    import src.current.labels  # noqa: F401  触发 labeler 注册
    return generate_labels(CONFIG.labels.active_labelers)


def step_export() -> None:
    from src.current.transform.exporter import export_all
    export_all()


def step_train() -> dict:
    from src.current.train.trainer import Trainer
    return Trainer().run()


def run_all(resume: bool = True) -> dict:
    step_collect(resume=resume)
    step_label()
    step_export()
    return step_train()
