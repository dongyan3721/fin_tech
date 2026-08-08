"""src.current —— 独立于 legacy 的供应链金融风控新实现。

设计目标（对照 README 与开发计划）：
- 数据收集 / 格式转换 / KMV 伪标签 / TGC 训练预测，端到端复现 legacy 效果；
- 用注册表(registry) + ABC 基类预留四类插入点：
  风险标签、时序模型、Agent 集成、绘图数据收集插件；
- 不 import、不复制 src/legacy 的任何代码。

约定：所有新采集/产出数据落在项目根目录的 ``repository/``。
"""

from src.current.config import CONFIG

__all__ = ["CONFIG"]
