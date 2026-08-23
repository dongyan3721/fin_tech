"""市场风险标签插入点（PPT 提到的 GARCH + 折算系数思路）——当前为 stub。

思路（待实现）：
1) 用行情/期货价格拟合 GARCH(1,1) 得到条件波动率 σ_t；
2) 结合折算系数/利率风险，映射为市场风险度量或违约概率修正项；
3) 产出列如 market_risk_probability，与 KMV 标签在 labels 表并列。

实现后把返回值改为真实 DataFrame，并新建一个组合它的 LabelScheme
（或直接作为底层 labeler 被 kmv/hybrid 方案引用）即可纳入标签生成，无需改动其他代码。
"""
from __future__ import annotations

import pandas as pd

from src.current.labels.base import LabelContext, RiskLabeler
from src.current.registry import LABELERS


@LABELERS.register("market_garch")
class MarketGarchLabeler(RiskLabeler):
    output_column = "market_risk_probability"

    def generate(self, ctx: LabelContext) -> pd.DataFrame:  # noqa: ARG002
        print("[labels] market_garch 尚未实现（插入点占位），返回空表。")
        return pd.DataFrame(columns=["symbol", "year", self.output_column])
