"""默认 no-op Agent：所有 hook 原样返回，作为 P0 占位与扩展模板。"""
from __future__ import annotations

from src.current.agents.base import AgentHook
from src.current.registry import AGENT_HOOKS


@AGENT_HOOKS.register("noop")
class NoOpAgent(AgentHook):
    pass
