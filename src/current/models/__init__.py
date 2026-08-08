"""模型模块：时序编码器（可插拔）+ TGC 网络。

导入本包即触发时序编码器注册到 registry.TEMPORAL_ENCODERS。
新增时序模型：继承 TemporalEncoder 并 @TEMPORAL_ENCODERS.register("<name>")，
然后在 config.ModelConfig.temporal_encoder 里切换。
"""
from src.current.models import temporal  # noqa: F401  触发注册
from src.current.models.base import TemporalEncoder
from src.current.models.tgc import TGCModel

__all__ = ["TemporalEncoder", "TGCModel"]
