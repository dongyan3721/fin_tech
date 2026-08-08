# 基于 LLM Agent 的 TGC 供应链金融风控模型增强方案

> 团队：图灵风控 ｜ 核心模型：TGC（时序图卷积神经网络）
> 设计目标：**在完全保留现有 TGC 图神经网络架构的前提下**，引入基于 LLM 的 Agent 作为"可插拔外挂增强模块"，提升下一年度企业违约/风险预测的**准确率与各项性能指标**。
> 参考文献：Fatemi & Hu, *FinVision: A Multi-Agent Framework for Stock Market Prediction*, ICAIF'24.

---

## 0. 设计定位

FinVision 中 Agent 的所有产物（新闻摘要、技术分析、反思结论）最终都作为**输入喂给决策模型**。本方案沿用该逻辑，但把"决策模型"替换为我们的 **TGC**：

- **TGC 结构保持不变**（主干、主体）。
- **LLM Agent 不做最终决策**，只负责：在**输入端**给 TGC 提供更优的特征 `X`、邻接 `A`、标签 `Y`；在**输出端**进行误差纠正。
- Agent 与 TGC **解耦、可插拔**，每个增强点都能单独开关，便于做消融实验证明增益。

四个注入点（按对准确率贡献从高到低）：

| 编号 | 注入位置 | Agent 角色 | 对应 FinVision 机制 | TGC 侧改动 |
|------|----------|-----------|---------------------|-----------|
| **A** | 特征侧 `X` | 文本特征 Agent | Summarize Agent | 特征矩阵维度扩充（拼接） |
| **B** | 结构侧 `A` | 关系发现 Agent | Technical Analyst（读结构） | 邻接矩阵增补边/权重 |
| **C** | 标签侧 `Y` | 标签校正 Agent | Reflection（复核信号） | 标签降噪/软化，无结构改动 |
| **D** | 训练侧 loss | 反思 Agent | Reflection Module（消融第一） | loss 重加权 / 残差修正 |

---

## 1. 整合架构

```
              LLM Agent 外设层（内网私有化部署）
   ┌───────────────┬────────────────┬────────────────┐
   │ 文本特征 Agent │ 关系发现 Agent  │ 标签校正 Agent  │
   └───────┬───────┴───────┬────────┴────────┬───────┘
           ▼ (A)           ▼ (B)             ▼ (C)
      X' 增强特征      A' 增强邻接         Y' 降噪标签
           └───────────────┼──────────────────┘
                           ▼
                ┌──────────────────────┐
                │    TGC 核心（主干）    │
                │  时序模块 + GCN + 残差 │  ← 图侧保持；时序侧可替换（见 §1.1）
                └──────────┬───────────┘
                           ▼  预测 ŷ
              Reflection Agent (D) ── 误差归因 → 难例重加权 / 残差修正
                           └────────── 反哺下一轮训练（闭环）
```

**一句话概括**：TGC 负责"算准风险传导"，LLM Agent 负责"在输入端补信息、在输出端纠错误"。

### 1.1 备注：时序模块可更换，建议单独做对比实验

> 记录（会话共识）：TGC 主干应拆成 **「时序建模」+「图卷积 GCN」** 两段。当前 `src/legacy/train_tgc.py` 里时序段是 `TemporalGatedConv`（门控 1D 卷积），**不是** GCN；GCN 只在后续按时间步聚合供应链邻居时发挥作用。PPT 文字曾写 GRU，与代码选型不一致，二者同属「时间模块」角色。

**时序这一块可以换成其它模型做消融/对比实验**，图卷积与预测头可先不动，例如：

| 时序模块候选 | 说明 |
|--------------|------|
| TemporalGatedConv（现状） | 门控 Conv1d，短窗滤波 |
| GRU / LSTM | 与 PPT 表述一致，按年逐步编码 |
| Transformer Encoder | 3 个时间步 self-attention |
| 简单基线 | 三年特征 concat / 平均后直接进 GCN |

约束与预期：

- 接口保持输出形状 `(N, T, H)`，以便继续「对每个 t 做 GCN」。
- 当前 `T=3` 极短，换时序骨架**未必**带来大幅指标跃迁，但有利于理解贡献拆解，并与汇报材料对齐。
- 实验时应固定数据切分、图构建与标签，**只改时序模块**，记入消融表（见 §6）。

该方向属于 **TGC 主干内部的可替换实验**，与 LLM Agent 外挂增强（A–D）正交，可并行推进。

---

## 2. 增强点 A：文本特征 Agent（特征侧 `X`）

### 2.1 目标
现有节点特征仅为硬财务指标（资产负债率、现金比率等）。违约往往**先有文本征兆**（控股股东高质押、审计非标、核心客户流失），这些信号**领先于财务报表**。用 LLM Agent 把非结构化文本转成"软特征"拼接进 `X`，为 TGC 补充领先信号。

### 2.2 数据源
年报 MD&A、公司公告、监管处罚、诉讼记录、行业舆情等（内网可获取部分）。

### 2.3 做法
1. LLM Agent 对每个企业-年度，抽取结构化风险要点；
2. 用内网 embedding 模型把要点编码为定长向量 `v_text ∈ R^d`；
3. 与原财务特征拼接：`X'[i] = [X_fin[i] ‖ v_text[i]]`。

### 2.4 简单示例

输入（某企业 2023 年公告文本片段）：
```
关于控股股东股份质押比例达 92% 的公告；
年度审计报告被出具"保留意见"；
第一大客户占营收 41%，报告期内应收账款激增。
```

Agent 抽取的结构化要点（示例）：
```json
{
  "controlling_shareholder_pledge_ratio": 0.92,
  "audit_opinion": "qualified",
  "customer_concentration_top1": 0.41,
  "receivable_surge_flag": 1,
  "risk_summary": "高质押 + 非标审计 + 客户集中度过高"
}
```
→ 经 embedding 得到 `v_text`（如 32 维），拼接到该节点特征。

### 2.5 伪代码

```python
def build_text_features(firm_id, year, corpora, llm_agent, embed_model, d=32):
    # 1. LLM 抽取结构化风险要点（内网私有模型）
    prompt = TEXT_FEATURE_PROMPT.format(firm=firm_id, year=year)
    risk_points = llm_agent.extract(prompt, corpora[firm_id][year])  # dict / 文本

    # 2. 编码为定长向量
    v_text = embed_model.encode(risk_points["risk_summary"])          # R^d
    return v_text

# 与原财务特征拼接，得到增强后的特征矩阵 X'
def augment_node_features(X_fin, firm_ids, years, ...):
    X_aug = []
    for i, fid in enumerate(firm_ids):
        v = build_text_features(fid, years[i], corpora, llm_agent, embed_model)
        X_aug.append(np.concatenate([X_fin[i], v]))   # [X_财务 ‖ X_文本]
    return np.stack(X_aug)                              # 直接喂给 TGC

# TGC 侧唯一改动：输入维度 in_dim = fin_dim + d
```

### 2.6 预期指标影响
主要改善 **Recall（漏报率）**，因补充了财务数据缺失的领先信号；对 AUC 亦有正向贡献。

---

## 3. 增强点 B：关系发现 Agent（结构侧 `A`）

### 3.1 目标
现有边仅来自"前五大采购商/客户"交易数据，会漏掉大量真实关联。TGC 的风险传导完全依赖图结构——**图越完整，传导越准**。用 Agent 挖出非交易型关联作为增补边。

### 3.2 可挖掘的关系类型
共同实控人、担保链、股权关联、高管交叉任职、重大诉讼关联等。

### 3.3 做法
Agent 从公告/工商/新闻中识别关系三元组 `(源企业, 关系类型, 目标企业)`，映射为新增边或对已有边的权重修正，写入邻接矩阵 `A'`。

### 3.4 简单示例

Agent 从公告识别：
```
企业 002284 为企业 000030 提供 5 亿元连带责任担保
```
→ 生成一条高风险传导边：
```
源节点 002284 → 目标节点 000030
关系类型 = 担保  边权重（风险传导系数）= 0.8
```
（相比交易边，担保边的风险传导强度更高，可赋更大权重。）

### 3.5 伪代码

```python
RELATION_WEIGHT = {"担保": 0.8, "共同实控人": 0.7, "股权关联": 0.6, "诉讼": 0.5}

def discover_relations(corpora, firm_ids, llm_agent):
    edges = []
    for text in corpora:
        triples = llm_agent.extract_relations(text)  # [(src, rel, dst), ...]
        for src, rel, dst in triples:
            if src in firm_ids and dst in firm_ids and rel in RELATION_WEIGHT:
                edges.append((src, dst, RELATION_WEIGHT[rel]))
    return edges

def augment_adjacency(A, edges, firm_index):
    A_aug = A.copy()
    for src, dst, w in edges:
        i, j = firm_index[src], firm_index[dst]
        A_aug[i, j] = max(A_aug[i, j], w)   # 增补/加强边，避免覆盖已有交易边
    return A_aug                             # 供 TGC 每个时间切片使用
```

### 3.6 预期指标影响
显著提升对**链式/隐性风险**的捕捉，改善 AUC 与 KS；是团队"供应链风险传导"卖点的直接强化。

---

## 4. 增强点 C：标签校正 Agent（标签侧 `Y`）

### 4.1 目标
KMV/GARCH 标签为纯模型计算，含噪声（如市场异常波动导致 KMV 误判）。监督学习中**标签质量是精度天花板**。用 Agent 交叉核对标签与真实风险事件，对冲突标签做软修正/降权。

### 4.2 做法
Agent 比对模型标签与真实事件（违约、暴雷、退市、重大处罚）：
- 标签与事实**一致** → 保持，可提升样本权重；
- 标签与事实**冲突** → 软修正标签值，或降低该样本训练权重（降噪）。

### 4.3 简单示例

```
企业 X：KMV 标签 = 低风险(EDF=0.02)
真实事件：该企业次年发生债券实质性违约
→ 冲突！ 软修正标签 EDF: 0.02 → 0.5，或对该样本降权（视为噪声/异常）
```

### 4.4 伪代码

```python
def correct_labels(Y, firm_ids, years, events, llm_agent):
    Y_corr, sample_weight = Y.copy(), np.ones(len(Y))
    for k, (fid, yr) in enumerate(zip(firm_ids, years)):
        verdict = llm_agent.check_consistency(
            label=Y[k], event=events.get((fid, yr)))   # "一致"/"冲突"/"未知"
        if verdict == "冲突":
            Y_corr[k] = soft_adjust(Y[k], events[(fid, yr)])  # 软修正
            sample_weight[k] = 0.5                              # 或降权
        elif verdict == "一致":
            sample_weight[k] = 1.2                              # 可信样本加权
    return Y_corr, sample_weight   # 喂给 TGC 的监督信号 + 样本权重
```

### 4.5 预期指标影响
稳定提升 **AUC/KS**，因监督信号更接近真实风险分布。

---

## 5. 增强点 D：反思 Agent（训练侧 loss）⭐ 重点

> FinVision 的消融实验证明 **Reflection 模块贡献最大**。本增强点改动最小（仅动 loss 或加一层残差），却最可能显著提升指标，建议优先落地。

### 5.1 目标
标准 TGC 训练对预测数据均匀拟合，对**误报/漏报缺乏针对性修正**。反思 Agent 对预测错的样本做误差归因，并转化为训练信号，让 TGC "针对性补短板"。

### 5.2 两种实现方式（可择一或结合）

**方式一：难例重加权（hard-example reweighting）**
反思 Agent 归纳误判模式（如"某行业系统性低估""供应链末端节点漏报"），据此提高对应样本在下一轮训练中的 loss 权重。

**方式二：残差修正（stacking）**
让反思 Agent / 轻量模型学习 TGC 的**残差**（真实值 − TGC 预测），最终预测 = TGC 输出 + 残差修正项。TGC 完全不动。

### 5.3 简单示例（误差归因）

```
本轮预测复盘：
- 漏报集中在"电子制造"行业的供应链末端节点（Recall 仅 0.55）
- 误报集中在现金流充裕但高质押企业
反思结论 → 下一轮：对"电子制造 + 末端节点"样本 loss 权重 ×2
```

### 5.4 伪代码

```python
# ---- 方式一：难例重加权 ----
def reflect_and_reweight(model, val_data, llm_agent, base_w):
    preds = model.predict(val_data.X, val_data.A)
    errors = collect_errors(preds, val_data.y)          # 误报/漏报样本
    # LLM 对错误样本归因，返回需加权的规则
    rules = llm_agent.diagnose(errors)                  # e.g. {"电子制造&末端": 2.0}
    new_w = base_w.copy()
    for k, sample in enumerate(val_data):
        for cond, factor in rules.items():
            if match(sample, cond):
                new_w[k] *= factor
    return new_w                                        # 用于下一轮加权训练

# 训练循环（TGC 结构不变，仅 loss 加权）
for epoch in range(E):
    for batch in loader:
        pred = tgc(batch.X, batch.A)                    # TGC 前向：完全不变
        loss = weighted_loss(pred, batch.y, batch.w)    # 唯一改动：样本权重
        loss.backward(); opt.step()
    if epoch % R == 0:
        w = reflect_and_reweight(tgc, val_set, llm_agent, w)

# ---- 方式二：残差修正（TGC 冻结）----
def residual_correction(tgc, corrector, X, A, y_true=None):
    base = tgc.predict(X, A)                            # 冻结的 TGC
    if y_true is not None:                              # 训练 corrector
        corrector.fit(features_of(X, A), y_true - base) # 学残差
    return base + corrector.predict(features_of(X, A))  # 最终预测
```

### 5.5 预期指标影响
对 **F1 / Recall / KS** 提升最明显（针对性修正类别不平衡下的难例），且不改 TGC 结构。

---

## 6. 消融实验设计（用于证明每个增强点的增益）

违约预测为**类别不平衡**问题，重点关注 **AUC / KS / Recall**，不要只看 Accuracy。

| 方案 | AUC | KS | F1 | Recall | Accuracy |
|------|-----|----|----|--------|----------|
| TGC（Baseline，原系统：TemporalGatedConv + GCN） | — | — | — | — | — |
| TGC-GRU（仅替换时序模块为 GRU，图侧不变） | | | | | |
| TGC-LSTM / TGC-Transformer（可选） | | | | | |
| TGC + A（文本特征） | | | | | |
| TGC + A + B（+关系边） | | | | | |
| TGC + A + B + C（+标签降噪） | | | | | |
| **TGC + A + B + C + D（全量，+反思）** | | | | | |

逐项叠加可清晰量化每个 Agent 的独立贡献，论证严谨（对齐 FinVision 的消融范式）。  
时序模块替换行与 Agent 增强行建议**分开报告**，避免把「换 RNN」和「加 LLM」的增益混在一起。

---

## 7. 落地优先级（按投入产出比）

1. **D（反思重加权/残差修正）**：改动最小、论文证明增益最大 → 首选。
2. **A（文本特征拼接）**：最直观补充领先信号，工程成熟。
3. **B（关系边发现）**：增益大但数据挖掘成本较高。
4. **C（标签降噪）**：锦上添花，稳定提升。

正交项（可随时插入）：**时序模块替换实验**（§1.1）——改动面小、与 Agent 无关，适合在基线稳定后先做一轮 GRU vs Conv 对比。

---

## 8. 合规提醒（内网环境）

上述所有 Agent 均涉及企业财务、交易、关联关系等**敏感数据**。在内网防泄露要求下：

- **禁止**将数据传输至公网大模型 API（构成数据外泄）；
- 所有 LLM 与 embedding 模型须采用**内网私有化/本地化部署**（如内网部署的开源模型），确保**数据不出内网**；
- 不得为调用外部模型建立任何内网穿透/端口映射/公网暴露通道。

---

*文档结束。如需，可针对增强点 D 或 A 进一步展开为可运行的模块级实现。*
