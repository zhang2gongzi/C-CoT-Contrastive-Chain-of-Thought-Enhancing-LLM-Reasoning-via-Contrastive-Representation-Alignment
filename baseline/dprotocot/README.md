# D-ProtoCoT 代码说明

本文件夹是基于论文 *D-ProtoCoT: Prototype-Based Path Selection for Chain-of-Thought Reasoning* 的**干净复现**，共 9 个 Python 文件。代码结构性修复了审稿人提出的多个方法论问题（类平衡 ORM、答案泄漏消融、统一测试集切分、step-level InfoNCE + 动态原型）。

---

## 文件总览

```
dprotocot/
├── config.py      # 全局配置（超参数、路径、消融开关）
├── data.py        # 数据加载、按题分组、答案提取与泄漏控制
├── encoder.py     # MultiGranularEncoder：分块编码 + 多粒度表示
├── losses.py      # Step-level InfoNCE 对比损失
├── prototype.py   # 动态原型构建 + 路径选择
├── train.py       # 编码器训练循环
├── orm.py         # ORM 基线（修复版，类平衡 + 诊断指标）
├── evaluate.py    # 统一评估（同一测试集、多种方法对比）
├── baselines.py   # 新基线：Self-Certainty / USC / GenSelect / Pairwise-LLM
├── llm_utils.py   # LLM 调用工具（vLLM / Together / DeepInfra / OpenAI）
└── run.py         # CLI 入口（main / orm / leakage / granularity / baselines）
```

---

## 1. config.py — 全局配置

**作用**：所有可调参数集中管理，通过 CLI 覆盖。

**不需要输入/输出**，只是一个 dataclass 配置对象。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bert_model` | `${MODEL_DIR}/bert-base-uncased` | 预训练 BERT 路径 |
| `data_path` | — | 单文件 jsonl，自动按比例切分 |
| `train_path / test_path` | None | 官方训练/测试集文件（二选一） |
| `split_ratio` | (0.8, 0.1, 0.1) | train/val/test 比例 |
| `lr` | 2e-5 | 学习率 |
| `batch_size` | 16 | 每 batch 题目数 |
| `epochs` | 3 | 训练轮数 |
| `temperature` | 0.07 | InfoNCE 温度 |
| `chunk_size / chunk_overlap` | 400 / 50 | 长文本分块参数 |
| `input_mode` | `"full"` | 泄漏消融：full / mask / qa_only |
| `train_repr` | `"step"` | 训练粒度：step / path |
| `select_repr` | `"path"` | 选择粒度：step / path |
| `use_context` | False | 是否拼接 context 字段 |
| `subset_questions` | None | 限制题目数（None=全用） |

**预期 jsonl 格式**（一行一条推理路径）：

```json
{
  "raw_example": {
    "id": "q_001",
    "question": "What is 2+2?",
    "context": "optional context",
    "label": 4
  },
  "cot": "Step 1: ...\nStep 2: ...\nSo the answer is 4.",
  "gold_label": 4,
  "is_correct": 1
}
```

相同 `raw_example.id` 的行属于同一道题的 K 条采样路径。

---

## 2. data.py — 数据加载与泄漏控制

**作用**：读取 jsonl、按题分组、按题切分、答案提取与掩码、提供编码器输入文本。

### 输入

| 输入 | 来源 | 格式 |
|------|------|------|
| `cfg.data_path`（或 `cfg.train_path + cfg.test_path`） | `config.py` | jsonl 文件路径 |
| `cfg.input_mode` | `config.py` | `"full"` / `"mask"` / `"qa_only"` |

### 输出

| 函数 | 返回值 | 说明 |
|------|--------|------|
| `load_splits(cfg)` | `(train_groups, val_groups, test_groups)` | 三个 list，每个元素是一个 dict（一道题的所有路径） |
| `trainable_questions(groups)` | `List[dict]` | 过滤出同时有正确和错误路径的题目 |
| `question_text(group, cfg)` | `str` | 题目文本（含 context 可选） |
| `path_text(cot, group, cfg)` | `str` | 路径文本（经 input_mode 处理） |
| `extract_final_answer(cot)` | `str 或 None` | 从 CoT 文本提取最终答案 |
| `apply_input_mode(cot, cfg)` | `str` | 根据 input_mode 变换路径文本 |

### 三种 input_mode 的效果

| mode | 编码器看到的路径文本 | 用途 |
|------|---------------------|------|
| `full` | 原始 CoT 全文（含最终答案） | 主实验 |
| `mask` | 最终答案被替换为 `[ANS]` 占位符 | 泄漏消融 |
| `qa_only` | 仅问题 + 答案，无推理步骤 | 泄漏消融上限 |

### 切分逻辑

- 按 `raw_example.id` 分组后切分 → 同一题的 K 条路径**不会跨 train/test**
- 支持两种模式：官方 train/test 文件，或单文件按 `split_ratio` 切分
- **打印 test set 题数**（回应审稿人对测试集规模的质疑）

---

## 3. encoder.py — 多粒度编码器

**作用**：将题目文本和推理路径编码为向量表示，支持 token / step / path 三种粒度。核心模型是 fine-tunable 的 `bert-base-uncased`。

### 输入

| 输入 | 格式 | 说明 |
|------|------|------|
| `cfg.bert_model` | 路径字符串 | 预训练 BERT 路径 |
| 文本（题目或路径） | `str` | 任意长度 |

### 输出

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `encode_text_pooled(text)` | 1 个 str | `[H]` tensor | 题目向量（mean-pooled） |
| `encode_path(text)` | 1 个 str | `(step_mat [M, H], path_emb [H])` | 单条路径的 step 矩阵 + path 向量 |
| `encode_paths(texts)` | `List[str]`（K 条路径） | `(list of [M_i, H], path_mat [K, H])` | K 条路径的 step 矩阵列表 + path 矩阵 |

### 关键设计：分块编码

- 长文本（>512 token）用滑动窗口分块（`chunk_size=400`, `chunk_overlap=50`）
- 每块独立编码，重叠部分取平均
- Step 切分：用 `\n` 分隔符 + token offset mapping 将字符级 step 边界映射到 token 索引
- Step 表示 = mean-pool 该 step 对应 token 的 hidden states
- Path 表示 = mean-pool 该路径所有 step 的表示

---

## 4. losses.py — 对比损失

**作用**：计算 Step-level InfoNCE 损失。正样本 = 正确路径的每个 step，分母 = 所有路径的所有 step。

### 输入

| 参数 | 格式 | 说明 |
|------|------|------|
| `z_q` | `[H]` tensor | 题目向量 |
| `step_mats` | `List[[M_i, H]]` | K 条路径各自的 step 矩阵 |
| `is_correct` | `List[int]`（K 个 0/1） | 每条路径是否正确 |
| `tau` | float | 温度系数 |
| `train_repr` | `"step"` 或 `"path"` | 训练粒度 |

### 输出

| 返回值 | 说明 |
|--------|------|
| 标量 loss | 若该题无正样本返回 None，不计入训练 |

### train_repr 两种模式

- **step**（默认）：每个 step 是独立候选单元，正样本 = 所有正确路径的所有 step
- **path**：每条路径 mean-pool 为一个向量，正样本 = 所有正确路径

---

## 5. prototype.py — 动态原型与路径选择

**作用**：在推理时，用 K 条路径的表示构建动态原型，选对齐度最高的路径。**不需要正确答案标签**。

### 输入

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `build_prototype(z_q, path_embs)` | `z_q [H]`, `path_embs [K, H]` | `(prototype [H], weights [K])` | softmax 加权构建原型 |
| `select_path(z_q, path_embs)` | 同上 | `int`（路径索引） | 选对齐度最高的路径 |
| `select_path_centroid(path_embs)` | `path_embs [K, H]` 仅此 | `int` | 无加权质心基线 |

### 计算流程

```
w_i   = softmax( sim(z_q, z_i) )          # 所有 K 条路径参与，无标签
p_q   = sum_i w_i * z_i                   # 动态 per-question 原型
a_i   = sim(z_i, p_q)                      # 对齐分数
c*    = argmax_i a_i                        # 选中的路径
```

---

## 6. train.py — 编码器训练

**作用**：用 InfoNCE 损失训练 MultiGranularEncoder，梯度累积，验证集监控。

### 输入

| 输入 | 格式 | 说明 |
|------|------|------|
| `cfg` | `Config` | 配置对象 |
| `train_groups` | `List[dict]` | 训练集（一道题一个 dict，含所有路径） |
| `val_groups` | `List[dict]` | 验证集（可选） |

### 输出

| 返回值 | 说明 |
|--------|------|
| 训练好的 `MultiGranularEncoder` | 同时保存到 `cfg.output_dir` |

### 训练逻辑

- 每道题单独计算 InfoNCE 损失
- 梯度累积 `cfg.batch_size` 道题后做一次 optimizer step
- 每 epoch 打印 contrastive loss + val loss
- 只对有正负样本的题目计算损失（`trainable_questions`）

---

## 7. orm.py — ORM 基线（修复版）

**作用**：Outcome Reward Model，预测单条路径是否正确。**修复了原始代码的问题**：使用 BCEWithLogitsLoss + pos_weight 对抗类别不平衡，报告完整诊断指标。

### 输入（训练时）

| 输入 | 说明 |
|------|------|
| `cfg` | 配置对象 |
| `train_groups` | 所有训练题（不需要过滤，ORM 用全部路径） |
| `val_groups` | 验证题 |

### 输出

| 函数 | 返回值 | 说明 |
|------|--------|------|
| `train_orm(...)` | `ORM` 模型 | 训练好的 ORM |
| `eval_orm_diagnostics(model, cfg, groups)` | dict | `{loss, acc, F1, AUROC}` 四项诊断指标 |
| `orm_select(model, cfg, group)` | `int` | 选预测正确概率最高的路径索引 |

### ORM 结构

```
z_q  = encoder(text_question)       # [H]
z_p  = encoder(path_text)           # [H]
feat = [z_q; z_p]                   # [2H]
logit = Linear(2H, 1)(feat)         # 标量 logit
```

### 修复点

- 使用 `pos_weight = n_neg / n_pos` 补偿正负样本不平衡
- 每 epoch 报告 path-level accuracy、F1、AUROC
- 选择逻辑用 `argmax sigmoid(logit)`（正确符号方向）

---

## 8. evaluate.py — 统一评估

**作用**：在**同一个测试集**上评估所有方法，直接可比。

### 输入

| 输入 | 格式 | 说明 |
|------|------|------|
| `cfg` | `Config` | 配置对象 |
| `test_groups` | `List[dict]` | 测试集 |
| `trained_encoder` | `MultiGranularEncoder` | 训练好的编码器 |
| `orm_model` | `ORM`（可选） | 若提供，追加 ORM 结果 |

### 输出

| 返回值 | 说明 |
|--------|------|
| `dict` | `{方法名: accuracy%, "_test_questions": N}` |

### 评估的方法

| 方法 | 选择逻辑 | 需要训练 |
|------|---------|---------|
| **Standard CoT** | 取第一条路径 | 否 |
| **Self-Consistency** | 提取答案后多数投票 | 否 |
| **Self-Certainty-BERT** | BERT 语义相似度：选与其他路径最一致的 | 否（冻结 BERT） |
| **Raw-BERT + Centroid** | 冻结 BERT + 无加权质心最近邻 | 否 |
| **D-ProtoCoT** | 训练好的编码器 + 动态原型 | 是 |
| **ORM**（可选） | 训练好的 ORM 选最高分路径 | 是（ORM） |
| **USC**（可选，需 LLM） | LLM 阅读 K 条路径后选出最佳 | 否（prompting） |
| **GenSelect**（可选，需 LLM） | 推理模型深度分析后选择 | 否（prompting） |

---

## 10. baselines.py — 新基线选择器

**作用**：实现审稿人要求的近三年推理时路径选择基线，统一接口 `selector(group, **kw) -> int`。

### 实现的基线

| 基线 | 论文 | 核心思路 | 需要什么 |
|------|------|---------|---------|
| `sel_self_certainty_logprobs` | Self-Certainty, NeurIPS 2025 | 选 log-probability 最高的路径（长度归一化） | jsonl 含 `logprobs` 字段 |
| `sel_self_certainty_bert` | Self-Certainty (BERT 近似) | 选与其他路径 BERT 余弦相似度最高的路径 | 任意 BERT encoder |
| `sel_usc` | USC, 2023 | LLM 阅读 K 条路径，选出最佳 | `llm_call` 函数 |
| `sel_genselect` | GenSelect, ICML 2025 | 推理模型深度分析后选择 | `llm_call` 函数（推荐 QwQ/DeepSeek-R1） |
| `sel_pairwise_llm` | Pairwise 锦标赛 | 两两对比 + 淘汰赛，O(K) 次调用 | `llm_call` 函数 |

### 接口

```python
# BERT-based（不需要 LLM）
from baselines import sel_self_certainty_bert
idx = sel_self_certainty_bert(group, encoder=encoder, cfg=cfg)

# LLM-based（需要 LLM API）
from baselines import make_llm_selectors
selectors = make_llm_selectors(llm_call=my_llm_fn)
idx = selectors["USC"](group)
```

---

## 11. llm_utils.py — LLM 调用工具

**作用**：为 USC / GenSelect 等 LLM-based 基线提供统一的 API 调用接口。

### 支持的后端

```python
from llm_utils import get_llm

# vLLM 本地服务（默认）
llm = get_llm("vllm", base_url="http://localhost:8000/v1", model="qwen3-8b")

# Together AI（LLaMA-3.1-70B）
llm = get_llm("together", model="meta-llama/Llama-3.1-70B-Instruct-Turbo")

# DeepInfra
llm = get_llm("deepinfra", model="meta-llama/Llama-3.1-70B-Instruct")

# OpenAI
llm = get_llm("openai", model="gpt-4o")

# 测试 mock（始终返回 Path 1）
llm = get_llm("dry-run")
```

所有后端返回统一的 `LLMCall` 协议：`fn(messages: List[dict]) -> str`。

---

## 12. run.py — CLI 入口

**作用**：命令行启动五种实验模式。

### 命令

```bash
# 1. 主实验：Standard CoT / Self-Consistency / Self-Certainty-BERT / Centroid / D-ProtoCoT
python run.py main --data_path /path/to/data.jsonl --use_context --output_dir runs/exp1

# 2. 主实验 + 修复版 ORM
python run.py orm --data_path /path/to/data.jsonl --output_dir runs/exp2

# 3. 答案泄漏消融（自动跑 full / mask / qa_only 三轮）
python run.py leakage --data_path /path/to/data.jsonl

# 4. 表示粒度消融（path/path, step/step, step/path 三轮）
python run.py granularity --data_path /path/to/data.jsonl --use_context

# 5. 完整基线对比（含 USC / GenSelect 等 LLM 基线）
python run.py baselines \
    --data_path /path/to/data.jsonl \
    --llm_backend vllm --llm_model qwen3-8b \
    --llm_base_url http://localhost:8000/v1
python run.py baselines \
    --data_path /path/to/data.jsonl \
    --llm_backend together --llm_model meta-llama/Llama-3.1-70B-Instruct-Turbo

# 使用官方训练/测试集
python run.py main --train_path train.jsonl --test_path test.jsonl
```

### 常用可选参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--bert_model` | str | 覆盖 BERT 路径 |
| `--epochs` | int | 训练轮数 |
| `--batch_size` | int | batch 大小 |
| `--lr` | float | 学习率 |
| `--temperature` | float | InfoNCE 温度 |
| `--seed` | int | 随机种子 |
| `--device` | str | 设备（cuda/cpu） |
| `--subset_questions` | int | 限制题目数（快速调试用） |
| `--input_mode` | str | full/mask/qa_only（单次指定） |
| `--train_repr / --select_repr` | str | step/path（粒度消融用） |
| `--llm_backend` | str | vllm / together / deepinfra / openai / dry-run |
| `--llm_model` | str | LLM 模型名 |
| `--llm_base_url` | str | API 地址（覆盖默认值） |
| `--llm_api_key` | str | API key（或设环境变量） |
| `--include_pairwise` | flag | 附加 Pairwise-LLM 基线 |

---

## 数据流全貌

```
jsonl 文件
  │
  ▼
data.py: load_splits()        ← 按题分组 + 按题切分
  │
  ├─→ train_groups  ──→  train.py: train_encoder()  ──→  训练好的 encoder
  ├─→ val_groups    ──→  train.py: _val_loss()           （保存到 output_dir）
  └─→ test_groups   ──→  evaluate.py: evaluate_all()  ──→  结果 dict
                              │
                              ├─ Standard CoT
                              ├─ Self-Consistency
                              ├─ Self-Certainty-BERT
                              ├─ Raw-BERT + Centroid
                              ├─ D-ProtoCoT        ← 训练好的 encoder
                              ├─ ORM (可选)        ← train_orm() 训练的 ORM
                              ├─ USC (可选)        ← LLM API 调用
                              ├─ GenSelect (可选)  ← LLM API 调用
                              └─ Pairwise-LLM (可选)
```
