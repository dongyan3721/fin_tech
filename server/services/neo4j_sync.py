"""Neo4j 数据同步服务（可迁移性核心）。

server 启动时调用 :func:`sync_neo4j`，把 processed parquet 的供应链图同步到本地 Neo4j：
图模型：
    (:Company {symbol, latest_rating, latest_prob, ever_st, node_size})
    (:Company)-[:SUPPLIES {year, weight, proportion, color}]->(:Company)
    (:Company)-[:SELLS_TO {year, weight, proportion, color}]->(:Company)

幂等：节点用 MERGE（symbol 唯一约束），关系先删后建。Neo4j 未配置/不可达时优雅跳过，
返回状态 dict，绝不抛出异常阻断 server 启动。
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from neo4j import GraphDatabase

from src.current.config import CONFIG, get_neo4j_config

REL_TYPE_MAP = {"supply": "SUPPLIES", "sale": "SELLS_TO"}


def _clean(d: dict) -> dict:
    """去掉 None/NaN 值（Neo4j 属性不允许 null）。"""
    return {k: v for k, v in d.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))}


def build_company_props(labels: pd.DataFrame, edges: pd.DataFrame) -> list[dict]:
    """节点属性：最新评级/概率 + 是否曾 ST/退市 + 节点大小。

    节点全集 = 标签公司 ∪ 边端点（无标签的上下游公司也要成节点，否则关系悬空）。
    """
    latest: dict[str, dict] = {}
    ever_st: set[str] = set()
    has_hybrid = not labels.empty and "st_level" in labels.columns
    for row in labels.sort_values("year").itertuples(index=False):
        sym = str(row.symbol)
        item = latest.setdefault(sym, {})
        rating = getattr(row, "risk_rating", None)
        prob = getattr(row, "default_probability", None)
        if rating is not None and str(rating) not in ("nan", "None"):
            item["latest_rating"] = str(rating)
        if prob is not None and str(prob) != "nan":
            try:
                item["latest_prob"] = float(prob)
            except (TypeError, ValueError):
                pass
        if has_hybrid:
            st_level = getattr(row, "st_level", 0) or 0
            delisted = getattr(row, "delisted", 0) or 0
            if int(st_level) > 0 or int(delisted) == 1:
                ever_st.add(sym)
    # 补齐只在边里出现的公司
    for col in ("source", "target"):
        for sym in edges[col].astype(str):
            latest.setdefault(sym, {})

    out = []
    for sym, props in latest.items():
        size = 10
        if "latest_prob" in props:
            try:
                size = max(10, min(60, int(round(props["latest_prob"] * 120))))
            except (TypeError, ValueError):
                pass
        out.append(_clean({"symbol": sym, **props, "ever_st": sym in ever_st,
                           "node_size": size}))
    return out


def sync_neo4j(config: Optional[dict] = None, wipe: bool = False,
               batch_size: int = 1000) -> dict:
    """把供应链图同步到 Neo4j，返回状态 dict（不抛异常）。

    Returns:
        {"status": "skipped"|"unreachable"|"error"|"ok", ...}
    """
    config = config or get_neo4j_config()
    uri = (config.get("uri") or "").strip()
    user = config.get("user") or "neo4j"
    password = (config.get("password") or "").strip()
    database = config.get("database") or "neo4j"

    if not uri or not password:
        return {"status": "skipped",
                "reason": "未配置 Neo4j（.env 需 NEO4J_URI / NEO4J_PASSWORD）"}

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as e:  # noqa: BLE001
        return {"status": "unreachable", "reason": str(e)}

    try:
        edges = pd.read_parquet(CONFIG.edges_parquet)
        labels = pd.read_parquet(CONFIG.labels_parquet)
        edges["source"] = edges["source"].astype(str)
        edges["target"] = edges["target"].astype(str)
        edges["year"] = edges["year"].astype(int)

        with driver.session(database=database) as s:
            if wipe:
                s.run("MATCH (n) DETACH DELETE n")

            s.run(
                "CREATE CONSTRAINT company_symbol IF NOT EXISTS "
                "FOR (c:Company) REQUIRE c.symbol IS UNIQUE"
            )

            # ---- 节点 ----
            companies = build_company_props(labels, edges)
            for i in range(0, len(companies), batch_size):
                s.run(
                    "UNWIND $rows AS r MERGE (c:Company {symbol: r.symbol}) SET c += r",
                    rows=companies[i:i + batch_size],
                )

            # ---- 关系（先删后建，保证与 parquet 一致）----
            s.run("MATCH ()-[r:SUPPLIES|SELLS_TO]->() DELETE r")
            total_rel = 0
            for i in range(0, len(edges), batch_size):
                chunk = []
                for row in edges.iloc[i:i + batch_size].itertuples(index=False):
                    rel_type = REL_TYPE_MAP.get(str(row.relationship))
                    if rel_type is None:
                        continue
                    props = _clean({
                        "year": int(row.year),
                        "weight": float(row.weight) if row.weight == row.weight else None,
                        "proportion": float(row.proportion) if row.proportion == row.proportion else None,
                        "color": "#4c8dff" if rel_type == "SUPPLIES" else "#36ad6a",
                    })
                    chunk.append({"source": row.source, "target": row.target,
                                  "rel_type": rel_type, "props": props})
                if not chunk:
                    continue
                for rel_type in ("SUPPLIES", "SELLS_TO"):
                    rows = [c for c in chunk if c["rel_type"] == rel_type]
                    if rows:
                        s.run(
                            f"UNWIND $rows AS r "
                            f"MATCH (a:Company {{symbol: r.source}}) "
                            f"MATCH (b:Company {{symbol: r.target}}) "
                            f"CREATE (a)-[rel:{rel_type}]->(b) SET rel = r.props",
                            rows=rows,
                        )
                        total_rel += len(rows)

            # ---- 校验（总量对比）----
            n_node = s.run("MATCH (c:Company) RETURN count(c) AS c").single()["c"]
            n_rel = s.run("MATCH ()-[r:SUPPLIES|SELLS_TO]->() RETURN count(r) AS c").single()["c"]

        parquet_edges = int(len(edges))
        consistent = int(n_rel) == parquet_edges
        return {
            "status": "ok",
            "database": database,
            "nodes": int(n_node),
            "relationships": int(n_rel),
            "parquet_edges": parquet_edges,
            "consistent": consistent,
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "reason": str(e)}
    finally:
        if driver is not None:
            driver.close()
