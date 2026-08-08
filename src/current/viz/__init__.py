"""绘图数据收集插件（可插拔）。

导入本包即触发各 exporter 注册到 registry.VIZ_EXPORTERS。训练流程在关键阶段
（训练结束/评估/建图）通过 VizManager 分发数据，已启用的插件各取所需产出图或数据文件。
新增插件：继承 VizExporter 并 @VIZ_EXPORTERS.register("<name>")，在
config.VizConfig.active_exporters 里启用。
"""
from src.current.viz import collectors  # noqa: F401  触发注册
from src.current.viz import neo4j_export  # noqa: F401  触发注册（预留）
from src.current.viz.base import VizExporter, VizManager

__all__ = ["VizExporter", "VizManager"]
