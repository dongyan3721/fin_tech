# 快速上手（Mini README）

> 5 分钟看懂如何部署、配置、训练和扩展本项目。
> 完整背景与算法说明见 [README.md](README.md)；前端设计见 [docs/前端开发计划.md](docs/前端开发计划.md)。

**一句话**：用上市公司供应链交易关系建图 + 财务指标做节点特征，训练 TGC（
时序图卷积）预测企业下一年度违约概率/市场风险；FastAPI + Vue3 提供可视化。

**推荐上手顺序**：§4 数据准备 → §6 一键训练 → §5 启动前后端查看结果。
（前端「总览 / 风险评测 / 方案对比」三个页面展示的就是训练产物，跳过 §6 直接开前端，
只有「供应链图谱 / 企业分析」两个页面有数据。）

---

## 1. 前置条件

| 项 | 要求 | 用途 |
|---|---|---|
| Python | 3.12（项目自带 `.venv`） | 训练/后端 |
| Node.js / npm | ≥ 18 / ≥ 9 | 前端构建 |
| Tushare Token | 必需（仅采集阶段消耗额度） | 财务/行情/期货数据 |
| GPU（可选） | CUDA 11.8 | 训练提速，CPU 也可跑 |

## 2. 安装

```bash
# Python 依赖（项目根目录）
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 前端依赖
cd frontend/current && npm install
```

## 3. 环境变量（根目录 `.env`，必配）

```ini
# 必需：Tushare 令牌（采集财务/行情/期货数据用）
TUSHARE_TOKEN=你的令牌

# 可选：反思 Agent 的 LLM 诊断（不配则跳过 LLM 环节）
SILICON_ENDPOINT=https://api.siliconflow.cn/v1
SILICON_APIKEY=你的密钥
SILICON_MODEL=Qwen/Qwen3.5-27B
```

> 无需任何数据库连接配置。Tushare 限频 200 次/分钟，采集结果全部磁盘缓存，重跑只补缺失。

## 4. 数据准备（首次部署执行一次）

```bash
.venv\Scripts\python.exe scripts/merge_supply_data.py   # 合并供应链 Excel → data/raw/
.venv\Scripts\python.exe -m src.current.cli collect      # 采集边+财务+行情+风险事件+期货/行业（耗额度，可断点续跑）
.venv\Scripts\python.exe -m src.current.cli label        # 生成标签（默认 kmv 方案）
.venv\Scripts\python.exe -m src.current.cli export       # 导出 processed/{nodes,edges,labels}.parquet
```

> 提示：`collect` 一步即包含市场风险标签所需的全部原始数据（36 个商品期货日线/换月映射 + 申万行业成员，
> 首跑约 10 分钟，之后走缓存秒级）。`label --scheme market` 阶段只做 GARCH 计算，不再联网。
> 拿到现成 `repository/processed/*.parquet` 可跳过本节。

## 5. 启动前后端

**开发模式**（前后端分离，前端热更新）：

```bash
bash server/api.sh start                  # 后端 API :8000（启动时预加载年度图缓存）
cd frontend/current && npm run dev        # 前端 :5173（/api 自动代理到 8000）
# 打开 http://localhost:5173
```

**生产模式**（单进程，FastAPI 托管前端构建产物）：

```bash
cd frontend/current && npm run build      # 生成 dist/
bash server/api.sh start                  # 直接打开 http://127.0.0.1:8000
```

**服务管理**：`bash server/api.sh {start|stop|restart|status}`；端口被占可 `bash server/kill_api.sh`。

启动日志里会看到预加载埋点（属正常）：

```
[preload] 边 7657 条（2001–2025 共 25 年），标签 17785 行
[preload]   2025 年: 994 节点 / 778 边（89.3 ms）
[preload] 完成：25 个年度图全部就绪，总耗时 0.38s，后续 /api/graph 查询毫秒级
```

> 注意：首次调用实时推理接口（`/api/inference`）需加载 torch，约 10 秒，之后毫秒级。

> ⚠️ **重要：大部分页面依赖训练产物**。「总览 / 风险评测 / 方案对比」读取的是
> `repository/outputs/<run>/` 下的训练产物（指标、预测明细、checkpoint），**必须先完成 §6
> 的一键训练**才有数据展示；「供应链图谱 / 企业分析」只依赖 §4 的 processed parquet，
> export 完即可用。若启动前端后这几个页面图表为空，先跑一遍训练再刷新。

## 6. 一键训练

```bash
# 典型：LSTM + 500 轮 + KMV 基线标签
.venv\Scripts\python.exe scripts/train.py --temporal lstm --epochs 500

# 市场风险标签训练（标签方案 + 监督目标列一起指定）
.venv\Scripts\python.exe scripts/train.py --temporal lstm --epochs 500 \
    --label-scheme market --target-column market_risk_label --name exp_market

# 启用反思 Agent（难例重加权 + LLM 诊断）
.venv\Scripts\python.exe scripts/train.py --epochs 300 --agent --name exp_agent
```

常用参数：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--epochs` / `--lr` | 训练轮数 / 学习率 | 500 / 1e-3 |
| `--temporal` | 时序编码器：`gated_conv` / `gru` / `lstm` / `transformer` | gated_conv |
| `--label-scheme` | 标签方案：`kmv` / `hybrid` / `market` / `mix` | kmv |
| `--target-column` | 监督目标列：`default_probability` / `market_risk_label` / `composite_risk_label` | 随方案 |
| `--agent` | 启用反思 Agent（增强点 D） | 关 |
| `--name` | 输出目录名（`repository/outputs/<name>/`） | 时间戳 |
| `--eval-every` | 每 N 轮评估，自动保存最佳模型（0=关） | 100 |
| `--log` | 日志文件路径 | 控制台 |

产物：`repository/outputs/<name>/`（metrics.json、checkpoint、预测明细、图表）；训练前会按所选方案自动重新生成标签。

## 7. 可插拔扩展

所有扩展点在 `src/current/registry.py`，模式统一：**继承基类 + 装饰器注册 + 配置切换**。

**① 新增时序模型**（`src/current/models/temporal.py`）：

```python
from src.current.models.base import TemporalEncoder
from src.current.registry import TEMPORAL_ENCODERS

@TEMPORAL_ENCODERS.register("my_encoder")          # 输入输出均为 (N, T, C)
class MyEncoder(TemporalEncoder):
    def __init__(self, dim, dropout=0.3):
        super().__init__(dim, dropout)
        self.layer = ...                            # 你的网络

    def forward(self, x):
        return ...                                  # (N,T,C) -> (N,T,C)

# 使用：scripts/train.py --temporal my_encoder
```

**② 新增标签方案**（`src/current/labels/schemes.py`）：

```python
from src.current.labels.base import LabelContext, LabelScheme
from src.current.registry import LABEL_SCHEMES

@LABEL_SCHEMES.register("my_label")
class MyLabelScheme(LabelScheme):
    def generate(self, ctx: LabelContext):
        # ctx.financial / ctx.market / ctx.events 为 interim 数据
        # 返回必须含 [symbol, year, <目标列>] 的 DataFrame
        ...

# 使用：
#   cli label --scheme my_label                 # 生成标签
#   scripts/train.py --label-scheme my_label --target-column <你的目标列>
```

**③ 底层标签器**（如需复用的标签计算单元，`src/current/labels/`）：
继承 `RiskLabeler` 并 `@LABELERS.register("name")`，供标签方案组合调用（参考 `kmv.py` / `st.py` / `market_garch.py`）。

## 8. 常用命令速查

| 命令 | 作用 |
|---|---|
| `bash server/api.sh start / stop / status` | 后端服务管理 |
| `cd frontend/current && npm run dev` | 前端开发服务 |
| `npm run build`（前端目录内） | 生产构建 |
| `-m src.current.cli collect / label / export / train` | 数据流水线四步 |
| `scripts/train.py --temporal lstm --epochs 500` | 一键训练 |
| `scripts/predict.py --year 2026 --top 20` | 离线推理指定年份 |
| `repository/outputs/experiments_log.csv` | 历次训练台账（前端「方案对比」页同源） |
