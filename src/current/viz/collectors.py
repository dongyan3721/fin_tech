"""内置绘图数据收集插件：训练损失曲线、预测散点、图快照。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，避免服务器/内网环境报错
import matplotlib.pyplot as plt  # noqa: E402

from src.current.registry import VIZ_EXPORTERS
from src.current.viz.base import VizExporter


@VIZ_EXPORTERS.register("training_curve")
class TrainingCurveExporter(VizExporter):
    def handle(self, stage: str, data: dict, out_dir: Path) -> None:
        if stage != "training":
            return
        losses = data.get("losses") or []
        if not losses:
            return
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.title("Training Loss (current TGC)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(out_dir / "training_loss.png")
        plt.close()


@VIZ_EXPORTERS.register("scatter")
class ScatterExporter(VizExporter):
    def handle(self, stage: str, data: dict, out_dir: Path) -> None:
        if stage != "evaluation":
            return
        results = data.get("results")
        if results is None or results.empty:
            return
        plt.figure(figsize=(8, 6))
        plt.scatter(results["actual_probability"], results["predicted_probability"], alpha=0.6)
        plt.plot([0, 1], [0, 1], "r--", linewidth=2)
        plt.xlabel("Actual Default Probability")
        plt.ylabel("Predicted Default Probability")
        plt.title("Actual vs Predicted (Test Set)")
        plt.savefig(out_dir / "prediction_scatter.png")
        plt.close()


@VIZ_EXPORTERS.register("graph_snapshot")
class GraphSnapshotExporter(VizExporter):
    """导出建图用到的节点/边快照，便于二次绘图或核对。"""

    def handle(self, stage: str, data: dict, out_dir: Path) -> None:
        if stage != "graph":
            return
        nodes = data.get("nodes")
        edges = data.get("edges")
        if nodes is not None and not nodes.empty:
            nodes.to_csv(out_dir / "graph_nodes.csv", index=False, encoding="utf-8-sig")
        if edges is not None and not edges.empty:
            edges.to_csv(out_dir / "graph_edges.csv", index=False, encoding="utf-8-sig")
