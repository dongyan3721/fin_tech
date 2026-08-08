"""Agent 集成插入点（P0 为 no-op）。

导入本包即注册 AgentHook 到 registry.AGENT_HOOKS。四个 hook 对应
`LLM增强TGC供应链风控_设计文档.md` 的四个增强点：特征增强/关系发现/标签矫正/反思。
"""
from src.current.agents import noop  # noqa: F401  触发注册
from src.current.agents.base import AgentHook

__all__ = ["AgentHook"]
