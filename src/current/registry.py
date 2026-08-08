"""通用注册表：为四类插入点（标签/时序模型/绘图/Agent）提供统一的“加文件+注册”扩展方式。

用法::

    from src.current.registry import LABELERS

    @LABELERS.register("kmv")
    class KMVLabeler(RiskLabeler):
        ...

    labeler = LABELERS.create("kmv", ...)      # 实例化
    names = LABELERS.keys()                      # 已注册名列表
"""
from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """装饰器：把类/工厂以 key 注册进表。"""
        def _wrap(obj: T) -> T:
            if key in self._registry:
                raise KeyError(f"[{self.name}] 重复注册: {key!r}")
            self._registry[key] = obj
            return obj
        return _wrap

    def get(self, key: str) -> T:
        if key not in self._registry:
            raise KeyError(
                f"[{self.name}] 未注册: {key!r}。已注册: {sorted(self._registry)}"
            )
        return self._registry[key]

    def create(self, key: str, *args, **kwargs):
        """按 key 取出并实例化（假定注册对象是可调用的类/工厂）。"""
        return self.get(key)(*args, **kwargs)

    def keys(self) -> Iterable[str]:
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry


# 四类插入点各自的全局注册表
LABELERS: Registry = Registry("labelers")            # 风险标签插入点
TEMPORAL_ENCODERS: Registry = Registry("temporal")    # 时序模型插入点
VIZ_EXPORTERS: Registry = Registry("viz")             # 绘图数据收集插件
AGENT_HOOKS: Registry = Registry("agents")            # Agent 集成插入点
