"""风险标签模块（可插拔）。

导入本包即触发各 labeler 注册到 registry.LABELERS。新增风险标签只需：
1) 新建一个继承 RiskLabeler 的类；2) 用 @LABELERS.register("<name>") 注册；
3) 在 config.LabelConfig.active_labelers 里启用。
"""
from src.current.labels import kmv  # noqa: F401  触发注册
from src.current.labels import market_garch  # noqa: F401  触发注册（插入点 stub）
from src.current.labels import st  # noqa: F401  触发注册（方案D ST/退市事件）
from src.current.labels.base import RiskLabeler, generate_labels

__all__ = ["RiskLabeler", "generate_labels"]
