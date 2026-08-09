# TGC 供应链金融风控系统架构文档

> 本文档基于项目实际代码编写，描述数据流转、模型结构、评估体系和反思 Agent 机制。

---

## 目录

1. [数据存储格式](#1-数据存储格式)
2. [张量形状与流动](#2-张量形状与流动)
3. [图卷积（GCN）详解](#3-图卷积gcn详解)
4. [KMV 标签计算](#4-kmv-标签计算)
5. [评估指标解析](#5-评估指标解析)
6. [反思 Agent（增强点 D）](#6-反思-agent增强点-d)

---

## 1. 数据存储格式

### 1.1 三张核心 Parquet 表

系统产出三张 Parquet 表，存储在 `repository/processed/`：

| 表名 | 文件 | 行数 | 说明 |
|------|------|------|------|
| nodes | `nodes.parquet` | 17,675 | 公司×年份的财务特征节点 |
| edges | `edges.parquet` | 7,657 | 供应链有向边（供应商→核心→客户） |
| labels | `labels.parquet` | 14,633 | 公司×年份的 KMV 违约概率标签 |

### 1.2 节点表 Schema（`nodes.parquet`）

| 列名 | 类型 | 说明 |
|------|------|------|
| `symbol` | str | 6 位股票代码（主键 1） |
| `year` | int | 年份（主键 2） |
| `debt_to_asset_ratio` | float | 资产负债率 = total_liab / total_assets |
| `current_ratio` | float | 流动比率 = current_assets / current_liab |
| `quick_ratio` | float | 速动比率 = (current_assets - inventory) / current_liab |
| `interest_coverage_ratio` | float | 利息保障倍数 |
| `total_assets` | float | 总资产（元） |
| `total_liab` | float | 总负债（元） |
| `current_assets` | float | 流动资产（元） |
| `current_liab` | float | 流动负债（元） |
| `revenue` | float | 营业收入（元） |
| `operate_profit` | float | 营业利润（元） |

**代码位置**: `src/current/transform/schema.py:16-18`

### 1.3 边表 Schema（`edges.parquet`）

| 列名 | 类型 | 说明 |
|------|------|------|
| `source` | str | 源节点股票代码 |
| `target` | str | 目标节点股票代码 |
| `weight` | float | 边权重（交易金额，已 log1p 归一化） |
| `relationship` | str | 关系类型：`supply`（供应商→核心）或 `sale`（核心→客户） |
| `proportion` | float | 交易占比 |
| `year` | int | 年份 |

**代码位置**: `src/current/transform/schema.py:21`

### 1.4 标签表 Schema（`labels.parquet`）

| 列名 | 类型 | 说明 |
|------|------|------|
| `symbol` | str | 6 位股票代码（主键 1） |
| `year` | int | 年份（主键 2） |
| `default_probability` | float | KMV 违约概率（核心标签） |
| `distance_to_default` | float | 违约距离 DD |
| `asset_value` | float | 资产价值 = market_cap + total_liab |
| `default_point` | float | 违约点 = total_liab × 0.7 |
| `risk_rating` | str | 信用评级 AAA~D |

**代码位置**: `src/current/labels/kmv.py:64-72`

---

## 2. 张量形状与流动

### 2.1 数据准备阶段

```
nodes.parquet (17675, 12)  +  labels.parquet (14633, 7)
            ↓                           ↓
            └──────── build_dataset() ──┘
                            ↓
              滑动窗口构造序列样本（连续 4 年：前 3 年特征 → 第 4 年标签）
                            ↓
         ┌──────────────────┼──────────────────┐
         ↓                  ↓                  ↓
   train: (991, 3, 10)  test: (513, 3, 10)  future: (258, 3, 10)
   y_train: (991,)      y_test: (513,)      (无标签)
```

**代码位置**: `src/current/train/dataset.py:59-126`

- 输入特征 F=10（财务指标）
- 时间步 T=3（连续 3 年）
- 样本数 N=991（训练）/ 513（测试）/ 258（推演）

### 2.2 图构建阶段

```
edges.parquet (7657, 6)  +  companies list
            ↓
    build_graph_by_pred_year()
            ↓
    按预测年分组，取 pred_year - lag 年的边
            ↓
    edge_index: (2, E)   # E = 样本级边数，如 408
    edge_weight: (E,)    # log1p + min-max 归一化
```

**代码位置**: `src/current/train/dataset.py:153-206`

- `edge_index` 的索引指向 batch 行号（样本即节点）
- 不同预测年的样本互不相连（块对角结构）

### 2.3 模型前向传播

```
输入: x = (N, T=3, F=10)
      edge_index = (2, E)
      edge_weight = (E,)

      ↓ [input_proj: Linear(10 → 64)]
      
      h = (N, 3, 64)
      
      ↓ [SpatioTemporalBlock]
      ↓   ├─ temporal: GatedConvEncoder
      ↓   │    h = (N, 3, 64)
      ↓   │    h.transpose(1,2) → (N, 64, 3)
      ↓   │    Conv1d(64, 128, kernel=3, padding=1) → (N, 128, 3)
      ↓   │    .transpose(1,2) → (N, 3, 128)
      ↓   │    chunk(2, dim=-1) → gate=(N,3,64), filt=(N,3,64)
      ↓   │    sigmoid(gate) * tanh(filt) → (N, 3, 64)
      ↓   │    + residual + LayerNorm → (N, 3, 64)
      ↓   │
      ↓   └─ spatial: GCNConv(64, 64) 对每个时间步
      ↓        for t in range(3):
      ↓            x_t = h[:, t, :]  # (N, 64)
      ↓            x_t = ReLU(GCNConv(x_t, edge_index, edge_weight))  # (N, 64)
      ↓        cat → (N, 3, 64)
      ↓        + residual + LayerNorm → (N, 3, 64)
      
      ↓ [pool: AdaptiveAvgPool1d(1)]
      ↓   x.transpose(1,2) → (N, 64, 3)
      ↓   pool → (N, 64, 1)
      ↓   squeeze → (N, 64)
      
      ↓ [head: MLP]
      ↓   Linear(64 → 32) → ReLU → Dropout → Linear(32 → 1)
      ↓   → (N, 1)
      
      ↓ [final_activation]
      ↓   "sigmoid" → (N, 1) 概率值 [0, 1]
      ↓   "identity" → (N, 1) logit 值 (-∞, +∞)
      
输出: pred = (N,)  # squeeze 掉最后一维
```

**代码位置**: `src/current/models/tgc.py:63-86`, `src/current/models/temporal.py:17-29`

### 2.4 形状变化汇总表

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `x` | (N, 3, 10) | 3 年 × 10 特征 |
| 投影后 | `h` | (N, 3, 64) | hidden_dim=64 |
| 时序编码后 | `h` | (N, 3, 64) | 残差 + LayerNorm |
| 空间编码后 | `h` | (N, 3, 64) | GCN + 残差 + LayerNorm |
| 池化后 | `h` | (N, 64) | 时间维平均 |
| MLP 输出 | `out` | (N, 1) | 线性头 |
| 激活后 | `pred` | (N,) | sigmoid 或 identity |

---

## 3. 图卷积（GCN）详解

### 3.1 从供应链数据到图的边

#### 原始数据

供应链数据来自上市公司年报披露的前五大供应商和客户：

```
整合的供应链数据.xlsx
├── Year: 2023
├── Symbol: 000002 (万科，核心公司)
├── Supplier_Symbol: 600048 (保利发展，供应商)
├── Purchase_Amount: 1500000000 (采购额 15 亿)
├── Customer_Symbol: 000001 (平安银行，客户)
└── Sales_Amount: 800000000 (销售额 8 亿)
```

#### 边的定义

代码位置: `src/current/data/supply_chain.py:25-70`

```
supply 边: 供应商 → 核心公司 (风险从上游传导)
  source = Supplier_Symbol (600048)
  target = Symbol (000002)
  weight = Purchase_Amount (15 亿)

sale 边: 核心公司 → 客户 (风险向下游传导)
  source = Symbol (000002)
  target = Customer_Symbol (000001)
  weight = Sales_Amount (8 亿)
```

#### 样本示例

假设训练集有 5 个样本（公司×预测年）：

| 样本索引 | 公司 | 预测年 | 含义 |
|---------|------|--------|------|
| 0 | 000002 (万科) | 2024 | 用 2021-2023 特征预测 2024 风险 |
| 1 | 600048 (保利) | 2024 | 用 2021-2023 特征预测 2024 风险 |
| 2 | 000001 (平安) | 2024 | 用 2021-2023 特征预测 2024 风险 |
| 3 | 000002 (万科) | 2023 | 用 2020-2022 特征预测 2023 风险 |
| 4 | 600397 (安源煤业) | 2024 | 用 2021-2023 特征预测 2024 风险 |

**关键理解**：样本索引 0-4 既是 batch 维度，也是图的节点。`edge_index` 指向这些索引。

### 3.2 edge_index 的构建过程

代码位置: `src/current/train/dataset.py:153-206`

#### 步骤 1：按预测年分组

```python
# 输入:
companies = ["000002", "600048", "000001", "000002", "000397"]
pred_years = [2024, 2024, 2024, 2023, 2024]

# 分组结果:
year_to_indices = {
    2024: [0, 1, 2, 4],  # 预测 2024 的样本
    2023: [3],            # 预测 2023 的样本
}
```

#### 步骤 2：取对应年份的边

对于预测年 2024，取 `2024 - lag = 2023` 年的边（lag=1）：

```
edges.parquet (2023 年):
source   target   weight   relationship
600048   000002   1.5e9    supply    (保利 → 万科)
000002   000001   0.8e9    sale      (万科 → 平安)
600397   000002   0.3e9    supply    (安源 → 万科)
```

#### 步骤 3：将公司级边展开为样本级边

```python
# 建立 公司 → 样本索引 的映射
sym_to_idx = {
    "000002": [0, 3],  # 万科在样本 0 和 3
    "600048": [1],     # 保利在样本 1
    "000001": [2],     # 平安在样本 2
    "000397": [4],     # 安源在样本 4
}

# 对于 2024 年的边，展开为样本级:
边 600048 → 000002:
  source 样本: sym_to_idx["600048"] = [1]
  target 样本: sym_to_idx["000002"] = [0, 3]
  展开: (1→0), (1→3)

边 000002 → 000001:
  source 样本: [0, 3]
  target 样本: [2]
  展开: (0→2), (3→2)

边 600397 → 000002:
  source 样本: [4]
  target 样本: [0, 3]
  展开: (4→0), (4→3)
```

#### 步骤 4：构建 edge_index 和 edge_weight

```python
# 收集所有边 (仅 2024 年，2023 年只有一个样本无法形成边)
pairs = [[1,0], [1,3], [0,2], [3,2], [4,0], [4,3]]
weights = [1.5e9, 1.5e9, 0.8e9, 0.8e9, 0.3e9, 0.3e9]

# 权重归一化: log1p + min-max
w = log1p(weights) / max(log1p(weights))
  = [0.72, 0.72, 0.57, 0.57, 0.33, 0.33]

# 转为 PyG 格式
edge_index = tensor([[1, 1, 0, 3, 4, 4],   # source
                     [0, 3, 2, 2, 0, 3]])  # target
edge_weight = tensor([0.72, 0.72, 0.57, 0.57, 0.33, 0.33])
```

#### 最终图结构

```
节点 = 样本 (batch 行号)
边 = 供应链关系 (按预测年-1 年的结构)

    600048(1) ──0.72──→ 000002(0)
        │                  │
        │                  │
        └──0.72──→ 000002(3)
                   │
                   └──0.57──→ 000001(2)
                   
    600397(4) ──0.33──→ 000002(0)
        │
        └──0.33──→ 000002(3)
```

**注意**：样本 3（万科 2023）属于 2023 预测年，与 2024 年的样本不相连（块对角结构）。

### 3.3 GCNConv 的数学原理

代码位置: `src/current/models/tgc.py:39`, `torch_geometric.nn.GCNConv`

#### 消息传递范式

GCN 的核心是**消息传递**（Message Passing）：每个节点聚合其邻居的信息来更新自身表示。

```
对于节点 i，其第 (l+1) 层的表示 h_i^(l+1) 计算如下:

h_i^(l+1) = σ( Σ_j (α_ij × h_j^(l)) )

其中:
  - j 遍历 i 的所有邻居（包括 i 自身，如果有自环）
  - α_ij 是归一化系数（考虑节点度数）
  - σ 是激活函数（ReLU）
```

#### GCNConv 的具体实现

PyG 的 `GCNConv` 实现了 Kipf & Welling (2017) 的 GCN：

```
H^(l+1) = σ( D^(-1/2) × A_hat × D^(-1/2) × H^(l) × W^(l) )

其中:
  - H^(l): 第 l 层的节点特征矩阵 (N, F)
  - A_hat = A + I: 添加自环的邻接矩阵
  - D: 度矩阵，D_ii = Σ_j A_hat_ij
  - W^(l): 可学习的权重矩阵 (F_in, F_out)
  - σ: ReLU 激活
```

#### 在本项目中的应用

```python
# tgc.py:51-58 对每个时间步做 GCN
for t in range(seq_len):  # seq_len = 3
    x_t = x[:, t, :]              # (N, 64) 第 t 个时间步的特征
    x_t = F.relu(
        self.spatial(x_t, edge_index, edge_weight)  # GCNConv
    )
    x_t = F.dropout(x_t, p=0.3, training=self.training)
    outs.append(x_t.unsqueeze(1))

x = torch.cat(outs, dim=1)        # (N, 3, 64)
```

#### 具体计算示例

以节点 0（万科 2024）为例，经过 GCNConv 后：

```
输入:
  h_0 = [0.12, -0.34, 0.56, ...]  (64 维)
  h_1 = [0.23, -0.11, 0.45, ...]  (保利，邻居)
  h_4 = [0.08, -0.22, 0.33, ...]  (安源，邻居)

邻居集合 (包括自身): {0, 1, 4}
边权重: w_10=0.72, w_40=0.33, w_00=1.0 (自环)

归一化系数 (简化):
  α_10 = 0.72 / sqrt(deg(0) × deg(1)) = 0.72 / sqrt(3 × 2) = 0.29
  α_40 = 0.33 / sqrt(3 × 2) = 0.13
  α_00 = 1.0 / sqrt(3 × 3) = 0.33  (自环)

聚合:
  h_0_new = ReLU(α_00 × h_0 + α_10 × h_1 + α_40 × h_4) × W
          = ReLU(0.33×h_0 + 0.29×h_1 + 0.13×h_4) × W
```

**物理意义**：节点 0（万科）的新表示融合了：
- 自身财务特征（33%）
- 保利（供应商）的特征（29%）
- 安源（供应商）的特征（13%）
- 经过线性变换 W

这就是**风险传导**的数学实现：供应商的风险特征通过边传递到核心公司。

### 3.4 为什么这样设计？

| 设计选择 | 理由 |
|---------|------|
| 按预测年建图 | 避免训练/测试的图结构分布漂移 |
| lag=1 | 用预测年-1 年的边，确保"预测时点可得" |
| 块对角结构 | 不同预测年的样本互不干扰 |
| log1p 权重归一化 | 压缩大额交易的权重，避免单条边主导 |
| 样本即节点 | batch 维度与图节点维度对齐，简化实现 |

---

## 4. KMV 标签计算

### 4.1 理论基础

KMV 模型源自 Merton (1974) 的结构化信用风险模型，核心思想：

> 当公司资产价值跌破其负债（违约点）时，发生违约。违约概率取决于资产价值与违约点的距离，以及资产价值的波动率。

### 4.2 计算公式

```
资产价值 A = 股权市值(market_cap) + 总负债(total_liab)
违约点 DPT = 总负债 × 0.7
资产波动率 σ = asset_volatility (来自行情数据，下限 0.1)

违约距离 DD = (A - DPT) / (A × σ)
违约概率 EDF = Φ(-DD)    # Φ 为标准正态分布 CDF
```

**代码位置**: `src/current/labels/kmv.py:55-62`

### 4.3 计算流程

```
financial.parquet              market.parquet
     ↓                              ↓
[symbol, year, total_liab]   [symbol, year, market_cap, asset_volatility]
     └──────────┬─────────────────┘
                ↓ merge on (symbol, year)
                
         asset_value = market_cap + total_liab
         default_point = total_liab × 0.7
         vol = asset_volatility.clip(lower=0.1)
         
         dd = (asset_value - default_point) / (asset_value × vol)
         edf = scipy.stats.norm.cdf(-dd)
         
         risk_rating = 根据 edf 映射到 AAA~D
         
                ↓
         labels.parquet
```

### 4.4 评级映射

| EDF 区间 | 评级 | 含义 |
|---------|------|------|
| < 0.01 | AAA | 极低风险 |
| 0.01 ~ 0.05 | AA | 很低风险 |
| 0.05 ~ 0.10 | A | 低风险 |
| 0.10 ~ 0.20 | BBB | 中等风险 |
| 0.20 ~ 0.30 | BB | 较高风险 |
| 0.30 ~ 0.40 | B | 高风险 |
| 0.40 ~ 0.60 | CCC | 极高风险 |
| ≥ 0.60 | D | 违约 |

**代码位置**: `src/current/labels/kmv.py:20-26`

### 4.5 标签分布特征

实际数据中，KMV 违约概率呈现**高度右偏分布**：

- 均值 ≈ 0.05（5%）
- 中位数 ≈ 0.02（2%）
- 75 分位数 ≈ 0.07（7%）
- 最大值 ≈ 0.37（37%）

**这是 R² 低下的根本原因**（详见第 4 节）。

---

## 5. 评估指标解析

### 5.1 指标一览

| 指标 | 公式 | 评估能力 | 代码位置 |
|------|------|---------|---------|
| **MSE** | `mean((y - ŷ)²)` | 概率空间整体拟合误差 | trainer.py:262 |
| **RMSE** | `sqrt(MSE)` | 同 MSE，量纲与概率一致 | trainer.py:263 |
| **MAE** | `mean(\|y - ŷ\|)` | 平均绝对误差，对异常值鲁棒 | trainer.py:264 |
| **R²(prob)** | `1 - SS_res/SS_tot` (概率空间) | 模型解释概率空间方差的能力 | trainer.py:265 |
| **R²(logit)** | `1 - SS_res/SS_tot` (logit 空间) | 模型解释对数几率空间方差的能力 | trainer.py:266 |
| **Spearman** | `corr(rank(y), rank(ŷ))` | 风险排序能力（Rank IC） | trainer.py:267 |
| **IC** | `corr(y, ŷ)` (Pearson) | 线性相关强度 | trainer.py:268 |

### 5.2 各指标评估的能力

#### MSE / RMSE / MAE — 绝对误差

- **衡量**：预测概率与真实概率的绝对偏差
- **适用场景**：需要精确概率估计时（如定价）
- **局限**：对极端值敏感（MSE）或不够敏感（MAE）

#### R²(prob) — 概率空间可决系数

```
R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

- **衡量**：模型解释概率空间方差的比例
- **物理意义**：R²=0.5 表示模型解释了 50% 的真实方差
- **问题**：当标签方差极小时，分母极小，R² 极易为负

#### R²(logit) — 对数几率空间可决系数

```
logit(p) = log(p / (1-p))
R²(logit) = 1 - Σ(logit(y) - logit(ŷ))² / Σ(logit(y) - logit(ȳ))²
```

- **衡量**：模型解释对数几率空间方差的能力
- **优势**：将 [0, 1] 的概率展开到 (-∞, +∞)，**对细粒度区分更敏感**
- **适用场景**：本项目的评级切片很细（AAA~D 共 8 级），R²(logit) 更能反映模型区分相邻评级的能力

#### Spearman — 排序相关系数

```
Spearman = corr(rank(y), rank(ŷ))
```

- **衡量**：预测排名与真实排名的相关性
- **物理意义**：Spearman=0.5 表示模型能正确区分 75% 的样本对
- **优势**：对异常值鲁棒，关注相对排序而非绝对值
- **适用场景**：**风控场景最核心的指标**，因为风控决策依赖排名而非绝对概率

#### IC (Pearson) — 信息系数

```
IC = corr(y, ŷ)
```

- **衡量**：预测值与真实值的线性相关强度
- **物理意义**：IC=0.3 表示中等程度的线性相关
- **与 Spearman 的区别**：IC 受异常值影响更大

### 5.3 为什么 R² 如此低？

本项目 R²(prob) 通常在 -0.2 ~ 0.0 之间，R²(logit) 在 -0.1 ~ 0.05 之间。这是**正常且符合预期的**，原因如下：

#### 原因 1：标签方差极小

```
真实违约概率分布：
  mean ≈ 0.05, std ≈ 0.04
  
总平方和 SS_tot = Σ(y - ȳ)² ≈ N × 0.04² = 0.0016N

即使模型完美预测，残差平方和 SS_res 也很难小于 SS_tot 的 50%
→ R² 上限约为 0.5
```

#### 原因 2：标签本身含噪声

KMV 是简化模型（未严格迭代 Merton），标签本身存在误差：
- 资产价值用 `market_cap + total_liab` 近似，未考虑资产流动性折扣
- 波动率用历史日收益率估计，存在估计误差
- 违约点用 `0.7 × total_liab` 经验公式，非最优

**噪声标签会压低 R² 的理论上限**。

#### 原因 3：任务本质是"排序"而非"回归"

风控的核心需求是**区分高风险与低风险公司**，而非精确预测违约概率：
- 如果模型给出 A 公司 0.08、B 公司 0.02，而真实是 A=0.10、B=0.01
- MSE 会惩罚这个偏差，但排序是正确的（A > B）
- **Spearman=0.50 已经能支撑有效的风控决策**

#### 原因 4：类别不平衡

绝大多数公司违约概率 < 0.1（低风险），少数 > 0.2（高风险）。模型倾向于预测中位数附近的值，导致：
- 对低风险公司预测准确（占多数）
- 对高风险公司预测偏低（漏报）
- 整体 R² 被高风险样本的偏差拉低

### 5.4 指标优先级建议

| 优先级 | 指标 | 理由 |
|--------|------|------|
| 1 | **Spearman** | 排序能力是风控决策的核心 |
| 2 | **R²(logit)** | 对细粒度评级区分最敏感 |
| 3 | **IC** | 线性相关强度 |
| 4 | **MAE** | 平均误差，对异常值鲁棒 |
| 5 | MSE / RMSE | 对异常值敏感，参考即可 |

---

## 6. 反思 Agent（增强点 D）

### 6.1 设计目标

标准 TGC 训练对所有样本均匀拟合，对误报/漏报缺乏针对性修正。反思 Agent 的目标：

> **分析模型预测误差，识别系统性偏差模式，通过样本加权让模型"针对性补短板"。**

### 6.2 核心机制：难例重加权

```
┌─────────────────────────────────────────────────────────────┐
│ 第一轮训练（均匀权重 w=1.0）                                  │
│   loss = MSE(pred, y)                                        │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 误差分析 (ReflectionAgent.analyze_errors)                    │
│                                                              │
│ 输入: test_preds (513 samples)                               │
│                                                              │
│ 按维度分组统计 MAE:                                           │
│   • 预测年 (2021/2022/2023/2024)                             │
│   • 财务特征 (高/低资产负债率, 高/低流动比率)                  │
│   • 图位置 (节点度数 0-1 / 2-5 / ≥5)                         │
│                                                              │
│ 输出: error_summary dict                                     │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 标签交叉验证 (ReflectionAgent.verify_labels)                 │
│                                                              │
│ 通过 Tushare namechange 接口查询 ST/*ST 状态                  │
│ 比对 KMV 标签与市场事实:                                      │
│   • KMV 低风险 + ST 状态 → 标签冲突 → 降权                   │
│   • KMV 高风险 + ST 状态 → 标签一致 → 保持                   │
│                                                              │
│ 输出: label_verification dict with conflicts list            │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ LLM 诊断 (ReflectionAgent.llm_diagnose)                      │
│                                                              │
│ 将误差统计摘要发送给 Qwen3.5-27B，请求:                        │
│   1. 分析误差集中在哪些年份/特征区间                           │
│   2. 判断标签验证冲突的含义                                    │
│   3. 输出加权建议 (JSON: year_rules, feature_rules)          │
│                                                              │
│ 输出: llm_diagnosis dict                                     │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 样本权重计算 (ReflectionAgent.compute_sample_weights)        │
│                                                              │
│ 基于误差分析，生成训练样本权重:                                │
│                                                              │
│ 规则 1: 高误差年份传播                                        │
│   if year_error_test[2021] > overall_ae × 1.2:               │
│       train_years [2019, 2020] × 1.08                        │
│                                                              │
│ 规则 2: 高资产负债率                                          │
│   if train_debt > median:                                    │
│       weight × 1.2                                           │
│                                                              │
│ 规则 3: 低流动比率                                            │
│   if train_cr < median:                                      │
│       weight × 1.15                                          │
│                                                              │
│ 规则 4: 标签冲突降权                                          │
│   if symbol in conflicts:                                    │
│       weight × 0.5                                           │
│                                                              │
│ 最终权重 clip 到 [0.3, 3.0]，归一化均值为 1.0                  │
│                                                              │
│ 输出: weights array (N_train,)                               │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 第二轮训练（加权 MSELoss）                                    │
│                                                              │
│ 原始 loss:                                                   │
│   loss = mean((pred - y)²)                                   │
│                                                              │
│ 加权 loss:                                                   │
│   loss = mean(w × (pred - y)²)                               │
│       = Σ(w_i × (pred_i - y_i)²) / N                         │
│                                                              │
│ 其中 w_i 为第 i 个训练样本的权重                              │
│                                                              │
│ 效果: 难例样本 (w > 1) 的误差被放大，模型更关注它们           │
└─────────────────────────────────────────────────────────────┘
```

**代码位置**: 
- 误差分析: `src/current/agents/reflection.py:58-113`
- 标签验证: `src/current/agents/reflection.py:118-171`
- LLM 诊断: `src/current/agents/reflection.py:176-287`
- 权重计算: `src/current/agents/reflection.py:292-401`
- 加权训练: `src/current/train/trainer.py:226-248`

### 6.3 加权训练的实现细节

```python
# trainer.py: _train_loop 中的加权 loss 实现

def _train_loop(self, x, y, edge_index, edge_weight,
                sample_weight: Optional[np.ndarray] = None):
    opt = torch.optim.Adam(...)
    losses = []
    self.model.train()
    
    use_weighted = sample_weight is not None
    if use_weighted:
        w_t = torch.tensor(sample_weight, dtype=torch.float).view(-1, 1)  # (N, 1)
    
    for epoch in range(epochs):
        opt.zero_grad()
        out = self.model(x, edge_index, edge_weight)  # (N, 1)
        
        if use_weighted:
            # 逐样本 MSE
            per_sample = (out - y).pow(2)           # (N, 1)
            loss = (per_sample * w_t).mean()         # 加权平均
        else:
            loss = nn.functional.mse_loss(out, y)    # 均匀 MSE
        
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    
    return losses
```

### 6.4 Agent 的实际效果（300 epoch 实验）

| 指标 | 基线（无 Agent） | 反思 R1（重训前） | 反思 R2（重训后） | 变化 |
|------|-----------------|------------------|------------------|------|
| MSE | 0.002571 | **0.002390** | 0.002630 | R1 ↓7.0% |
| MAE | 0.03203 | 0.03246 | **0.03160** | R2 ↓1.3% |
| R²(prob) | -0.174 | **-0.0917** | -0.201 | R1 ↑47% |
| R²(logit) | **0.0357** | -0.0868 | -0.0208 | 基线最佳 |
| Spearman | 0.494 | 0.484 | **0.502** | R2 ↑1.6% |
| IC | 0.386 | **0.428** | 0.384 | R1 ↑10.9% |

**分析**:

1. **第一轮（反思前）**：300 epoch 训练充分后，模型状态很好，MSE/R²/IC 均优于基线
2. **反思重训**：加权策略提升了 MAE 和 Spearman（排序能力），但 MSE/R² 有所下降
3. **R²(logit)**：基线最佳（0.0357），反思两轮均为负值，说明加权策略改变了 logit 空间的拟合

**结论**：反思 Agent 在排序指标（Spearman）上有微弱增益，但整体不如让模型自然训练。加权策略可能需要更精细的调整。

---

## 附录：关键代码文件索引

| 模块 | 文件 | 说明 |
|------|------|------|
| 配置 | `src/current/config.py` | 路径、常量、超参 |
| 数据采集 | `src/current/data/tushare_client.py` | Tushare API 封装 |
| 财务特征 | `src/current/data/financial.py` | 10 个财务指标计算 |
| 行情数据 | `src/current/data/market.py` | 市值、波动率计算 |
| 供应链边 | `src/current/data/supply_chain.py` | 从 Excel 构建边表 |
| KMV 标签 | `src/current/labels/kmv.py` | 违约概率计算 |
| 数据导出 | `src/current/transform/exporter.py` | 三表导出 parquet |
| 数据集构建 | `src/current/train/dataset.py` | 序列样本 + 图构建 |
| 模型定义 | `src/current/models/tgc.py` | TGC 网络结构 |
| 时序编码器 | `src/current/models/temporal.py` | GatedConv/GRU/LSTM/Transformer |
| 训练评估 | `src/current/train/trainer.py` | 训练循环 + 评估指标 |
| 反思 Agent | `src/current/agents/reflection.py` | 增强点 D 实现 |
| CLI 入口 | `src/current/cli.py` | 命令行接口 |

---

*文档生成时间：2026-08-09 | 基于代码版本 v1.0*
