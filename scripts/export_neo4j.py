"""把 processed 图数据导入本地 Neo4j（图谱页 Neovis.js 数据源）。

图模型：
    (:Company {symbol, latest_rating, latest_prob, ever_st})
    (:Company)-[:SUPPLIES   {year, weight, proportion}]->(:Company)   # supply 边
    (:Company)-[:SELLS_TO   {year, weight, proportion}]->(:Company)   # sale 边

用法：
    .venv\\Scripts\\python.exe scripts/export_neo4j.py               # 导入 + 验证
    .venv\\Scripts\\python.exe scripts/export_neo4j.py --wipe        # 清空后导入
    .venv\\Scripts\\python.exe scripts/export_neo4j.py --verify-only  # 只做数量校验

连接配置（环境变量可覆盖）：NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
默认 bolt://localhost:7687, neo4j / grapheval2026。
幂等：节点用 MERGE（按 symbol 唯一约束），关系先删后建，可重复执行。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from neo4j import GraphDatabase

from src.current.config import CONFIG

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "grapheval2026")
DATABASE = "neo4j"

REL_TYPE_MAP = {"supply": "SUPPLIES", "sale": "SELLS_TO"}


def _clean(d: dict) -> dict:
    """去掉 None/NaN 值（Neo4j 属性不允许 null）。"""
    return {k: v for k, v in d.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}


def build_company_props(labels, edges) -> list[dict]:
    """节点属性：最新评级/概率 + 是否曾 ST/退市。

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
        # 节点大小（vis/neovis 用整数 value 映射）：按违约概率 10~60
        size = 10
        if "latest_prob" in props:
            try:
                size = max(10, min(60, int(round(props["latest_prob"] * 120))))
            except (TypeError, ValueError):
                pass
        out.append(_clean({"symbol": sym, **props, "ever_st": sym in ever_st,
                           "node_size": size}))
    return out


def run(wipe: bool, verify_only: bool, batch_size: int) -> int:
    import pandas as pd

    edges = pd.read_parquet(CONFIG.edges_parquet)
    labels = pd.read_parquet(CONFIG.labels_parquet)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["year"] = edges["year"].astype(int)

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print(f"[neo4j] 已连接 {URI} (db={DATABASE})")

    if verify_only:
        return verify(driver, edges)

    with driver.session(database=DATABASE) as s:
        if wipe:
            s.run("MATCH (n) DETACH DELETE n")
            print("[neo4j] 已清空全图")

        s.run(
            "CREATE CONSTRAINT company_symbol IF NOT EXISTS "
            "FOR (c:Company) REQUIRE c.symbol IS UNIQUE"
        )

        # ---- 节点 ----
        companies = build_company_props(labels, edges)
        for i in range(0, len(companies), batch_size):
            chunk = companies[i:i + batch_size]
            s.run(
                "UNWIND $rows AS r MERGE (c:Company {symbol: r.symbol}) SET c += r",
                rows=chunk,
            )
        print(f"[neo4j] 节点 MERGE 完成: {len(companies)}")

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
            # APOC 不一定可用，按类型分两条 UNWIND
            for rel_type in ("SUPPLIES", "SELLS_TO"):
                rows = [c for c in chunk if c["rel_type"] == rel_type]
                if rows:
                    s.run(
                        f"UNWIND $rows AS r "
                        f"MATCH (a:Company {{symbol: r.source}}) "
                        f"MATCH (b:Company {{symbol: r.target}}) "
                        f"CREATE (a)-[rel:{rel_type}]->(b) "
                        f"SET rel = r.props",
                        rows=rows,
                    )
                    total_rel += len(rows)
        print(f"[neo4j] 关系创建完成: {total_rel}")

    rc = verify(driver, edges)
    driver.close()
    return rc


def verify(driver, edges) -> int:
    """逐年校验 Neo4j 关系数与 edges.parquet 一致，返回退出码。"""
    parquet_counts = edges.groupby("year").size().to_dict()
    ok = True
    print(f"\n{'年份':<6}{'parquet':>10}{'neo4j':>10}  状态")
    print("-" * 36)
    with driver.session(database=DATABASE) as s:
        for year in sorted(parquet_counts):
            expected = int(parquet_counts[year])
            got = s.run(
                "MATCH ()-[r:SUPPLIES|SELLS_TO {year:$y}]->() RETURN count(r) AS c",
                y=int(year),
            ).single()["c"]
            flag = "OK" if got == expected else "MISMATCH"
            if got != expected:
                ok = False
            print(f"{year:<6}{expected:>10}{got:>10}  {flag}")
        n_node_db = s.run("MATCH (c:Company) RETURN count(c) AS c").single()["c"]
    print("-" * 36)
    print(f"节点总数: neo4j={n_node_db}")
    print("校验结果:", "全部一致 ✅" if ok else "存在不一致 ❌")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="供应链图导入 Neo4j")
    parser.add_argument("--wipe", action="store_true", help="导入前清空全图")
    parser.add_argument("--verify-only", action="store_true", help="只校验不导入")
    parser.add_argument("--batch-size", type=int, default=1000)
    return run(parser.parse_args().wipe, parser.parse_args().verify_only,
               parser.parse_args().batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
