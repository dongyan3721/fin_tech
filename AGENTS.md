# 项目记忆

## Python 解释器
- 使用本项目内的 venv 环境解释器：`D:\Data\CodeRepository\py\GraghRiskEvaluate\.venv\Scripts\python.exe`
- 所有 Python 命令都必须通过 `.venv\Scripts\python.exe` 执行，不能使用系统全局 Python。
- 示例：
  - 运行 CLI：`.venv\Scripts\python.exe -m src.current.cli <command>`
  - 数据整合：`.venv\Scripts\python.exe scripts/merge_supply_data.py`

## 项目概述
基于供应链图（TGC，时序图卷积网络）的公司风险预测评级系统。
- 用上市公司供应链交易关系建图、财务指标做节点特征、KMV 违约概率做监督标签。
- 通过 TGC 模型预测下一年度违约概率。

## 执行流程
```bash
.venv\Scripts\python.exe -m src.current.cli collect   # 采集边+财务+行情(Tushare)
.venv\Scripts\python.exe -m src.current.cli label     # KMV 标签生成
.venv\Scripts\python.exe -m src.current.cli export    # 导出三张 parquet
.venv\Scripts\python.exe -m src.current.cli train     # 训练+评估+推演
```

## 快捷脚本
```bash
# 训练脚本（支持参数化 + Agent 开关 + 时序模型替换 + 自定义输出名）
.venv\Scripts\python.exe scripts/train.py --epochs 300 --lr 0.001 --agent
.venv\Scripts\python.exe scripts/train.py --epochs 500 --hidden-dim 128 --temporal gru --name exp_gru_d128
.venv\Scripts\python.exe scripts/train.py --temporal lstm --agent --name exp_lstm_agent

# 时序编码器选项：gated_conv(默认) / gru / lstm / transformer
# 详细用法见 scripts/train.py 文件头注释

# 推理脚本（加载 checkpoint 预测指定年份）
.venv\Scripts\python.exe scripts/predict.py --year 2025 --top 20
.venv\Scripts\python.exe scripts/predict.py --year 2025 --output predictions.csv --all
.venv\Scripts\python.exe scripts/predict.py --year 2025 --checkpoint repository/outputs/exp_gru_d128/model_checkpoint.pt
```

## 关键配置
- 供应链数据源：`repository/supply/`（三个 Excel，覆盖 2001-2025）
- 数据整合脚本：`scripts/merge_supply_data.py` → 输出到 `data/raw/整合的供应链数据.xlsx`
- Tushare 限频：200 QPM（低于此值易触发账号侧限流导致 60s 退避）
- 产出目录：`repository/{raw,interim,processed,outputs}/`

## 增强点 D：反思 Agent（Reflection Agent）
- 实现文件：`src/current/agents/reflection.py`
- 核心机制：难例重加权（Hard-example Reweighting）+ LLM 诊断 + Tushare 标签交叉验证
- 训练流程：第一轮均匀权重训练 → 反思分析误差 → 第二轮加权训练
- LLM 配置：SiliconFlow Qwen3.5-27B，需设置 `extra_body={"enable_thinking": False}`（思考模型返回空内容）
- 加权策略：高资产负债率样本 ×1.2、低流动比率样本 ×1.15、高误差年份邻近训练年 ×1.08
- Tushare 标签验证：通过 `namechange` 接口查询 ST 状态交叉验证 KMV 标签
- 配置入口：`config.AgentConfig.reflection`（ReflectionConfig）
- 产物：`repository/outputs/current_tgc_<时间戳>/reflection_report.json`