"""数据收集模块：Tushare 客户端 + 财务/行情/供应链三类采集器。

设计为可插拔：供应链边目前来自本地整合 Excel，未来可新增其他边源（插入点见
``supply_chain.py`` 的 EDGE_SOURCES 说明）。
"""
