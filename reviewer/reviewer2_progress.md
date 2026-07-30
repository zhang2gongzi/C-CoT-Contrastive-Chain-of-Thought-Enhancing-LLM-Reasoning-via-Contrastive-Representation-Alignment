# D-ProtoCoT 审稿回复进度

---

## 审稿人 #1：C-CoT 基线描述错误
- **问题**：论文把 Chia et al. 的 prompting 方法错误描述为置信度选择方法
- **代码修复**：`baseline/dprotocot/baselines.py` 新增 Self-Certainty、USC、GenSelect、Pairwise-LLM 四个正确基线
- **论文修复**：需更正 C-CoT 的描述和引用
- **状态**：待修改论文

---

## 审稿人 #2：数据集划分与使用情况不明确
- **问题**：1000 样本来源不明，train/test 切分不透明
- **代码修复**：`baseline/dprotocot/data.py` 按 question ID 分组切分，防止 path-level leakage；支持 `--train_path` + `--test_path`
- **数据验证**：
  - GSM8K：911 题全部来自官方训练集，0 来自测试集 ✓
  - CSQA：500 题全部来自官方训练集，0 来自测试集 ✓
  - StrategyQA：280 题全部来自官方训练集，0 来自测试集 ✓
  - 8:1:1 切分按 qid 分组，跨集重叠 = 0 ✓
- **GSM8K 额外验证**：正在服务器生成 200 题官方测试集 CoT，训练/测试彻底分离
- **状态**：等 GSM8K 测试集 CoT 跑完

---

## 审稿人 #3：ORM 基线结果异常薄弱
- **问题**：GSM8K Qwen ORM=61.36% vs Standard CoT=90.40%；StrategyQA ORM=50.43% 近乎随机
- **审稿人说**：训练了一个模型来选路径，结果选出来的还不如随便拿第一条，这不合理，肯定是你 ORM 实现有 bug。

### 论文 Table 1 的 ORM 数字

| 数据集 | 模型 | Standard CoT | ORM | 问题 |
|--------|------|-------------|------|------|
| GSM8K | Qwen3-8B | 90.40% | 61.36% | 比单条 CoT 还差 29% |
| StrategyQA | Qwen3-8B | 66.80% | 50.43% | 二分类，近乎瞎猜 |

### 原始 ORM 代码的三个致命问题

看 `baseline/ORM/train_orm.py`：

**问题 1：截断 512 token**
```python
# 第 73 行：truncation=True, max_length=512
enc = self.tokenizer(text, truncation=True, max_length=512, ...)
```
GSM8K 的 CoT 路径平均 809 token，93% 超过 512。直接把推理步骤砍掉了，只剩开头，模型根本看不到答案推导过程。

**问题 2：没有 pos_weight**
```python
# 第 105 行
loss_fct = nn.BCEWithLogitsLoss()  # 没有 pos_weight
```
Qwen3-8B 在 GSM8K 上 90% 的路径都是正确的，模型直接学「全猜正确」就能 90% 准确率。但所有路径分数差不多，选不出最好的那条。

**问题 3：用 [CLS] 联合编码 question + path**
```python
# 把问题和路径拼成一句话，取 [CLS] 输出
text = f"Question: {q}\nReasoning: {reasoning}"
pooled_output = outputs.last_hidden_state[:, 0, :]
```
路径被截断后喂给 BERT，问题和路径混在一起编码，区分度不够。

### dprotocot ORM 怎么修的

| 问题 | 原始 ORM | dprotocot ORM |
|------|---------|---------------|
| 长文本截断 | `truncation=True, max_length=512`，直接砍掉 | 分块编码（chunk_size=400, overlap=50），完整保留 |
| 类不平衡 | 无 `pos_weight`，模型偏向多数类 | `pos_weight = n_neg / n_pos`，补偿不平衡 |
| 编码方式 | `[CLS]` 联合编码 question+path | 分别编码 z_q 和 z_p，然后 concat `[z_q; z_p]` |
| 诊断 | 只报 accuracy | 每 epoch 报 train_loss, val_loss, accuracy, F1, AUROC, pos_ratio |

### 论文怎么回
跑 `python dprotocot/run.py orm` 会输出诊断指标，证明新 ORM 的 loss 在下降、F1/AUROC 正常。然后论文回：
> "We identified two issues in the original ORM: truncation of long reasoning paths and lack of class-balance weighting. After fixing both, ORM diagnostics are healthy (report AUROC, F1)."
- **状态**：代码已修复，等 GSM8K 跑完验证

---

## 审稿人 #4：主实验与消融实验结果不一致
- **问题**：StrategyQA Table 1=86.20%/72.60%，Table 3 消融=64.29%，对应不上
- **注意**：投稿时不存在 `baseline/dprotocot/`，全部分析基于原始代码

### 为什么差这么多？三个主要原因

**1. 测试集仅 100 题（最关键）**

`granularity_ablation.py` 第 315 行：
```python
train_val, test_data = train_test_split(all_data, test_size=100, random_state=42)
```
100 题意味着每题占 1% 准确率，随机波动极大。而主实验（`ccotqwenfulltry.py`）在完整 StrategyQA 测试集（~490 题）上评估，结果稳定得多。

**2. 编码器架构不同**

| | 消融实验 (`granularity_ablation.py`) | 主实验 (`ccotqwenfulltry.py`) |
|---|---|---|
| 编码方式 | `[CLS]` token 单一向量 (line 159) | Mean pooling 全序列 (line 248) |
| BERT 层级 | 全部训练 | 前 6 层冻结，后 6 层训练 |
| 表达能力 | 仅靠一个 token 概括整条推理链 | 所有 token 平均，语义信息更丰富 |

`[CLS]` 单向量很难捕捉长 CoT 推理链的完整语义。

**3. 硬截断 512 token**

`granularity_ablation.py` 第 25 行：`MAX_LEN = 512`，`truncation=True`。

GSM8K 的 CoT 路径平均 809 token，93% 超过 512。CSQA 和 StrategyQA 的推理链也经常超过 512。消融实验直接把关键推理步骤砍掉了。

### 次要因素

- **训练轮数**：消融 3 epoch vs 主实验 10 epoch
- **损失函数**：消融用 soft distributed labels（`1/n_pos` 均分正样本），主实验用 hard one-hot（第一个正样本=1）
- **数据划分**：消融用 `train_test_split(random_state=42)` 随机切，主实验用预划分文件

### 论文怎么回
> "The ablation study in Table 3 was conducted with preliminary code that used a smaller test set (100 samples, high variance), [CLS]-token encoding (vs. mean-pooling in the main experiments), and 512-token truncation. These implementation differences explain the gap between Table 3 and Table 1. We re-ran the ablation with the consistent setup and report updated results."

### 重跑计划

**当前数据状态**：
- GSM8K 训练集：`gsm8k_merged_flat.jsonl` ✓（911 题，9110 条路径）
- GSM8K 测试集：`gsm8k_test_flat.jsonl` ⏳（200 题，服务器生成中，PID 889520）
- StrategyQA：原始数据在 `database/Icanuse/qwen/strategyqa_generated.json`，待转换为 flat jsonl

**命令**：

```bash
# GSM8K（测试集生成完后，Plan A：train_path + test_path 物理分离）
python baseline/dprotocot/run.py granularity \
    --train_path gsm8k_merged_flat.jsonl \
    --test_path gsm8k_test_flat.jsonl \
    --output_dir runs/gsm8k_granularity

# StrategyQA（需先转换数据，无独立测试集，用 8:1:1 ratio split）
python convert_strategyqa.py   # 先转换
python baseline/dprotocot/run.py granularity \
    --data_path strategyqa_flat.jsonl \
    --use_context \
    --output_dir runs/strategyqa_granularity
```

**输出**：三组实验（path/path, step/step, step/path），同一 test split 上评估，数字直接可比。

| 训练粒度 | 选择粒度 | 含义 |
|---------|---------|------|
| path | path | 路径级训练 + 路径级选择（baseline） |
| step | step | Step 级训练 + Step 级选择 |
| step | path | Step 级训练 + 路径级选择（D-ProtoCoT） |

**和旧 Table 3 的对比**：

| | 旧代码 (`granularity_ablation.py`) | 新代码 (`run.py granularity`) |
|---|---|---|
| 编码器 | `[CLS]` 单向量 | Mean pooling 全序列 + 分块 |
| 截断 | 512 硬截断 | 分块编码 (chunk 400, overlap 50) |
| 测试集 | 100 题 | ~90 题 (GSM8K) 或完整 split |
| 数据划分 | `random_state=42` 随机切 | 按 qid 分组切分，无泄露 |

**待办**：
1. 等 GSM8K 测试集 CoT 生成完
2. 写 `convert_strategyqa.py`
3. 跑 GSM8K 消融
4. 跑 StrategyQA 消融
5. 把新数字替换进论文 Table 3

- **状态**：等 GSM8K 测试集 CoT 生成完后执行

---

## 审稿人 #5：答案泄露（捷径学习）
- **问题**：BERT 可能学会问题-答案相关性而非推理质量
- **核心逻辑**：
  1. 每条路径的输入 = 问题 + 完整推理 + 最终答案
  2. 标签 = 最终答案是否匹配标准答案
  3. BERT 可能直接学「问题和答案之间的文字相关性」来判断对错，压根不看中间的推理过程
  4. 例如：路径末尾是 "Final Answer: Yes" + 标签=正确 → BERT 学到「包含 Yes ≈ 正确」，而非「推理好不好」
- **代码修复**：`baseline/dprotocot/data.py` `input_mode` 支持三种消融

| 模式 | 输入内容 | 如果效果好说明 |
|------|---------|-------------|
| `full` | 完整推理 + 答案 | 正常设置 |
| `mask` | 推理过程中答案替换为 `[ANS]` | 去掉答案信息，推理过程仍有用 |
| `qa_only` | 仅问题 + 答案，无推理步骤 | BERT 在作弊，只靠 Q-A 关联 |

- **判断标准**：
  - `full` >> `qa_only` → 推理过程确实有用 ✓
  - `qa_only` ≈ `full` → BERT 在作弊，性能提升不来自推理 ✗
- **状态**：代码已支持，等跑消融实验（`python dprotocot/run.py leakage`）

---

## 审稿人 #6：过程级监督主张缺乏支持
- **问题**：正确最终答案但中间步骤错误的路径，所有 step 都被标为正样本→标签噪声
- **状态**：论文需弱化 claim，或提供对抗样本分析

---

## 审稿人 #7：术语使用不一致
- **问题**：sequence-level/path-level/chain/trajectory 混用
- **状态**：论文修改，统一术语

---

## 审稿人 #8：过度强调与 ORM 的范式差异
- **问题**：两者都是监督排序，本质相近
- **状态**：论文需弱化 "fundamentally different" 表述

---

## 审稿人 #9：方法论新颖性需精确阐明
- **问题**：三个组件各自不新
- **状态**：论文需阐明组件组合的 novelty，尤其是 step-level training + path-level inference 的不对称设计

---

## 数据文件清单

| 文件 | 题目数 | 路径数 | 来源 |
|------|--------|--------|------|
| `gsm8k_merged_flat.jsonl` | 911 | 9110 | 官方训练集（xiaorong + database 合并） |
| `gsm8k_500_flat.jsonl` | 500 | 5000 | 官方训练集（xiaorong） |
| `csqa_500_flat.jsonl` | 500 | 5000 | 官方训练集（xiaorong） |
| `gsm8k_test_flat.jsonl` | 生成中 | - | 官方测试集（200 题，LLM 生成中） |

## 代码文件清单

| 文件 | 作用 |
|------|------|
| `baseline/dprotocot/` | D-ProtoCoT 干净复现（9 个 py 文件） |
| `convert_data.py` | 分组 JSON → flat jsonl |
| `merge_gsm8k.py` | 合并两份 GSM8K 数据，去重验证 |
| `generate_gsm8k_test_cot.py` | 对官方测试集生成 CoT，支持断点续跑 |
