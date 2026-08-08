"""Neo4j 导出插件（离线 CSV，供 neo4j-admin import 使用）。

出于内网防泄露约束，本插件只在本地生成 CSV，绝不建立任何外部/公网连接。
默认不在 active_exporters 中启用；需要时把 "neo4j" 加入 config.VizConfig.active_exporters。
"""
from __future__ import annotations

from pathlib import Path

from src.current.registry import VIZ_EXPORTERS
from src.current.viz.base import VizExporter


@VIZ_EXPORTERS.register("neo4j")
class Neo4jExporter(VizExporter):
    def handle(self, stage: str, data: dict, out_dir: Path) -> None:
        if stage != "graph":
            return
        nodes = data.get("nodes")
        edges = data.get("edges")
        neo_dir = out_dir / "neo4j"
        neo_dir.mkdir(parents=True, exist_ok=True)
        if nodes is not None and not nodes.empty:
            n = nodes.copy()
            n = n.rename(columns={"symbol": "symbol:ID"})
            n[":LABEL"] = "Company"
            n.to_csv(neo_dir / "nodes_neo4j.csv", index=False, encoding="utf-8-sig")
        if edges is not None and not edges.empty:
            e = edges.copy()
            e = e.rename(columns={"source": ":START_ID", "target": ":END_ID",
                                  "relationship": ":TYPE"})
            e.to_csv(neo_dir / "edges_neo4j.csv", index=False, encoding="utf-8-sig")
        print(f"[viz] Neo4j 离线 CSV 已导出到 {neo_dir}（本地文件，无任何外部连接）")
