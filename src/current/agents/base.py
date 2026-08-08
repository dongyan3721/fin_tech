"""Agent 集成插入点：AgentHook 抽象基类。

四个 hook 对应设计文档的四个增强点，均为“可选、幂等”的插入：
- enhance_features(nodes)   文本/舆情等特征增强 -> 返回可能新增列的 nodes
- discover_relations(edges) LLM 关系发现 -> 返回可能补充的 edges
- correct_labels(labels)    标签矫正 -> 返回修正后的 labels
- reflect(metrics)          反思：基于评估结果给出改进建议（记录/回写）

P0 使用 NoOpAgent，全部原样返回，不改变流程。真正接入 LLM 时新增一个
AgentHook 子类并 @AGENT_HOOKS.register(...)，在 config.AgentConfig 中启用即可。
"""
from __future__ import annotations

from abc import ABC

import pandas as pd


class AgentHook(ABC):
    def enhance_features(self, nodes: pd.DataFrame) -> pd.DataFrame:
        return nodes

    def discover_relations(self, edges: pd.DataFrame) -> pd.DataFrame:
        return edges

    def correct_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        return labels

    def reflect(self, metrics: dict) -> None:
        return None
