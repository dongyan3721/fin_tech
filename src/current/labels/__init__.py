"""风险标签模块（可插拔）。

导入本包即触发各 labeler / 标签方案注册到 registry。扩展方式：
1) 底层标签器：继承 ``RiskLabeler`` 并 ``@LABELERS.register("<name>")``；
2) 标签方案（可配置对象）：继承 ``LabelScheme`` 并 ``@LABEL_SCHEMES.register("<name>")``，
   然后在 ``config.LabelConfig.label_scheme`` 里切换（默认 "kmv" 基线，可选 "hybrid"）。
"""
from src.current.labels import kmv  # noqa: F401  触发注册（KMV 标签器）
from src.current.labels import market_garch  # noqa: F401  触发注册（插入点 stub）
from src.current.labels import st  # noqa: F401  触发注册（方案D ST/退市事件标签器）
from src.current.labels import schemes  # noqa: F401  触发注册（kmv / hybrid 方案）
from src.current.labels.base import LabelScheme, RiskLabeler, generate_labels

__all__ = ["RiskLabeler", "LabelScheme", "generate_labels"]