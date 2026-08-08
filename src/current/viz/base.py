"""绘图数据收集插件基类与分发器。

阶段(stage)约定：
- "training": data={"losses": list[float]}
- "evaluation": data={"results": DataFrame[actual_probability, predicted_probability, ...]}
- "graph": data={"nodes": DataFrame, "edges": DataFrame}
插件只处理自己关心的 stage，其他忽略。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from src.current.config import CONFIG
from src.current.registry import VIZ_EXPORTERS


class VizExporter(ABC):
    @abstractmethod
    def handle(self, stage: str, data: dict, out_dir: Path) -> None:
        raise NotImplementedError


class VizManager:
    def __init__(self, active: Optional[List[str]] = None) -> None:
        names = active or CONFIG.viz.active_exporters
        self.exporters: List[VizExporter] = []
        for name in names:
            try:
                self.exporters.append(VIZ_EXPORTERS.create(name))
            except KeyError as e:
                print(f"[viz] 跳过未注册插件 {name}: {e}")

    def dispatch(self, stage: str, data: dict, out_dir: Path) -> None:
        for exp in self.exporters:
            try:
                exp.handle(stage, data, out_dir)
            except Exception as e:  # 插件失败不影响主流程
                print(f"[viz] {exp.__class__.__name__} 处理 {stage} 出错: {e}")
