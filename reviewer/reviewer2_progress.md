# D-ProtoCoT 审稿回复进度

---

## 审稿人 #1：C-CoT 基线描述错误
- **问题**：论文把 Chia et al. (2023) 错误描述为"基于置信度选择路径"。但 Chia et al. 实际是**对比 CoT prompting**（生成时提供正/负示范帮模型避错），不是候选路径选择方法。审稿人要求：说清 C-CoT 指代的确切算法、给实现细节和代码来源、改正引用。

### 代码溯源：Table 1 的 "C-CoT" 基线到底是什么
查代码发现有两个不同的 "C-CoT"：
1. `baseline/ccotPrompting/`（contrastive_cot_gsm8k.py 等）——**真正的 Chia et al. 对比 prompting**。
2. `baseline/commonsenseQA/ccot/`（train_prototype.py + run_ccot_qwen.py）——**Table 1 实际用的基线**。

`run_ccot_qwen.py:41-71` 实测逻辑：
- 用**冻结 BERT** 取每条路径的 **[CLS]** 向量（`max_length=256, truncation=True`）。
- `train_prototype.py` 把训练集所有**正确路径的 [CLS] 向量平均**成一个**固定的全局正样本原型**（`positive_prototype.npy`）。
- 推理时选与该静态原型**余弦相似度最高**的路径。

**结论**：这个基线既不是 Chia et al. 的对比 prompting，也**根本没用 token 概率/置信度**。它实际是"**冻结 BERT [CLS] + 静态全局原型**的余弦选择"，正好是 **D-ProtoCoT 的消融版**（静态全局原型 vs 动态逐题原型；冻结 vs 对比训练）。

### 论文里的错误位置（cas-dc-template.tex）
- **101行**：`confidence-based methods \citep{chia2023...} that rely on model-internal token probabilities` — Chia 引用错。
- **143行**：`Confidence-based selection ... token probabilities or entropy \citep{chia2023...}` — Chia 引用错。
- **164行**：`contrastive chain-of-thought prompting \citep{chia2023...}` — **正确，保留**。
- **343行**：`\item[C-CoT] \citep{chia2023...} selects reasoning paths based on confidence estimation.` — 基线定义错。
- **382行**：Table 1 里的 `C-CoT` 行（数字保留，名字要改）。
- **399行**：讨论说"C-CoT 靠 token-level confidence 在算术任务强"——**错的**，它没用 token 概率。

### ⚠️ 未定：Table 1 的 C-CoT 到底是哪份代码跑的
查代码发现 "C-CoT" 相关实现**至少四份**，不止两份：
1. `baseline/ccotPrompting/`（contrastive_cot_gsm8k.py）——真 Chia prompting（拼正/负示范生成答案，非选择）。
2. `baseline/commonsenseQA/ccot/`（train_prototype.py + run_ccot_qwen.py）——静态正样本原型 [CLS] 余弦选择。
3. `baseline/strategyQAccot/ccotllama/`——有 encoder/trainer/**infonce**，疑似**训练过的对比选择器**（若 Table 1 用这个，则与 D-ProtoCoT 高度相似，改名/描述完全不同）。
4. `baseline/commonsenseQA/ccotgpt/ccot_seq_infonce_train.py`——序列 InfoNCE 训练。

**用户回忆：可能跑了两种**（即 Table 1 六个 C-CoT 数字可能来自不同方法的混合，本身就是个一致性问题）。

**确定成立的结论**：论文对 C-CoT 的文字描述（"selects reasoning paths based on confidence estimation" + 引 Chia + token probabilities）在上述四种里**没有一种对得上**，审稿人指出的描述错误成立。

**待办**：比对各结果文件准确率 vs Table 1 六个数字（LLaMA: GSM8K 87.00 / CSQA 64.40 / StrategyQA 64.40；Qwen: GSM8K 68.20 / CSQA 75.60 / StrategyQA 62.70），确认每个 cell 是哪份代码跑的，再决定改名与描述。**改 tex 前必须先定这个。**

### ✅ 定稿方案（用户确认：跑了两种 → Table 1 拆成两行）
用户回忆：原 "C-CoT" 用了两套东西，缩写撞车导致引用写错。最终 Table 1 拆两行。

**行1 — Static-Prototype（保留现有数字）**
- 是什么：`baseline/commonsenseQA/ccot/` 静态原型选择——冻结 BERT [CLS]，正确训练路径 [CLS] 均值当全局原型，余弦 argmax 选路径。
- 用户自实现的 in-house 基线。查证网上**没有**缩写 C-CoT 的静态原型选择论文（Compressed CoT / Compositional CoT / CDW-CoT / Chia 都不是）→ 原引用是撞名误引。
- **引用处理（用户选 a）**：基线行**不引任何外部文献**，如实描述成 D-ProtoCoT 消融版（去掉对比训练 + 动态原型）。原型背景 `snell2017prototypical` **只放方法章节**，不压基线定义（避免二次误引：Snell 是学习嵌入+多类+欧氏，此处冻结+单原型+余弦，不同）。
- 论文作用：GSM8K 偶然高、CSQA/StrategyQA 掉 → 反衬动态设计（呼应 Figure 2）。

**行2 — C-CoT（Chia et al. 2023，重跑）**
- 是什么：真 Chia 对比 CoT prompting（`baseline/ccotPrompting/contrastive_cot_gsm8k.py` 为原型），in-context 给正/负示范引导生成，**生成时技术，不在共享 10 条采样路径里选择**。
- 引 `chia2023contrastivechainofthoughtprompting`。
- **需重跑**：现有脚本散乱、用老模型（Llama-2-7b、PARARULE），未覆盖 GSM8K/CSQA/StrategyQA × Qwen3-8B/LLaMA-3.1-8B。写统一脚本 `newrun/ccot_prompting.py` 跑 6 格。
- **必加脚注**：C-CoT 是生成时 prompting 基线，结果是它自己生成答案的准确率，非共享采样路径上的选择——与 Self-Consistency/ORM/Static-Prototype/D-ProtoCoT 口径不同。

**tex 具体改动**
- 343行：删原 C-CoT 定义，改成 Static-Prototype（无引用）+ C-CoT（引 Chia）两条 `\item`。
  > `\item[Static-Prototype]` A frozen BERT encoder embeds each candidate path via its [CLS] token. A single global prototype is precomputed as the mean [CLS] embedding of all correct training paths, and the path with the highest cosine similarity to this static prototype is selected. This is an ablated variant of D-ProtoCoT without contrastive training or the dynamic per-question prototype.
  > `\item[C-CoT]` \citep{chia2023...} A prompting-based method that supplies both valid and invalid reasoning demonstrations in-context to steer generation away from erroneous reasoning. It is a generation-time technique, not a candidate-path selection method.
- 101、143行：被错安到 Chia 的"置信度/token 概率"→ 换 `xiong2024llmsexpressuncertaintyempirical` / `sultan-astudillo-2025-confidence` / `leang2025picsar`。
- 164行：Chia 保留（描述正确）。
- 399行：原"C-CoT 靠 token-confidence 在算术强"→ 改讲 Static-Prototype（偶然高、泛化差）。
- 373行表 + 382行：C-CoT 行拆成 Static-Prototype + C-CoT 两行；加脚注。
- 方法章节：动态原型处引 `snell2017prototypical` 背景。

### 已澄清的实现细节（供 rebuttal）
- Static-Prototype 编码器：冻结 bert-base-uncased，[CLS] 池化，max_length=256。
- 原型：训练集正确路径 [CLS] 均值，静态全局，不随问题变化。选择：余弦 argmax。
- 代码来源：`baseline/commonsenseQA/ccot/train_prototype.py`、`run_ccot_qwen.py`。

- **状态**：方案定稿（拆两行，Static-Prototype 不引外部 + C-CoT 引 Chia 重跑）。待：写 `newrun/ccot_prompting.py` 重跑脚本 → 改 tex

---

## 审稿人 #2：数据集划分与使用情况不明确
- **问题**：1000 样本来源不明，train/test 切分不透明
- **代码修复**：`baseline/dprotocot/data.py` 按 question ID 分组切分，防止 path-level leakage；支持 `--train_path` + `--test_path`
- **数据验证**：
  - GSM8K：911 题全部来自官方训练集，0 来自测试集 ✓
  - CSQA：500 题全部来自官方训练集，0 来自测试集 ✓
  - StrategyQA：280 题全部来自官方训练集，0 来自测试集 ✓
  - 8:1:1 切分按 qid 分组，跨集重叠 = 0 ✓
- **GSM8K 额外验证**：200 题官方测试集 CoT 已生成 ✓，训练/测试彻底分离
- **状态**：数据准备完毕，可开始跑实验

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
- GSM8K 测试集：`gsm8k_test_flat.jsonl` ✓（200 题，已生成完毕）
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
1. ~~等 GSM8K 测试集 CoT 生成完~~ ✓
2. ~~写 `convert_strategyqa.py`~~ ✓（`newrundata/strategyqa_flat.jsonl`）
3. 跑 GSM8K 消融
4. 跑 StrategyQA 消融
5. 把新数字替换进论文 Table 3

- **状态**：数据准备完毕，可开始跑消融实验

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
- **状态**：草稿就绪（软化 claim + 复用 M4 AUC=0.78；**与审稿人1 Q1 同一问题、同一套证据，口径必须一致**）
- **不用新实验**：M4 AUC 已在 `reviewer1_q3_gsm8k.json` 跑出（GSM8K 200题 K=10）。

### 版本 A：Rebuttal 段落（英文，回信用）

> **Reply to Reviewer #2, Concern #6.**
>
> The reviewer is correct, and we thank them for the precise example. Our supervision is outcome-derived: path-level labels come from final-answer matching and are broadcast to every step, so a path that reaches the correct answer through a flawed step is nominally labeled positive — a genuine source of step-level label noise. We have accordingly **weakened the process-level supervision claim throughout the paper**: we no longer assert that the encoder detects localized reasoning errors, and instead describe it as attuned to *step-level semantic consistency with the question*.
>
> We nonetheless clarify why the method remains effective despite this shared label noise, which is a matter of training *geometry* rather than cleaner labels. Step-level InfoNCE aligns each of the $|\mathcal{P}|\cdot M$ steps of a correct path to the question independently; occasional step-level noise (a flawed step inside a nominally-positive path) is therefore a minority signal, diluted first when a path's steps are pooled into a path embedding and again when path embeddings are aggregated into the dynamic prototype. Systematically flawed paths, whose majority of steps deviate, remain separable. Empirically, on 200 GSM8K questions the learned alignment predicts path correctness with an **AUC of 0.78**, showing the outcome-level noise does not overwhelm the learned signal. We have not overclaimed step-level error detection, and we report this AUC as the honest evidence of what the representation does capture.

### 版本 B：正文改动（tex 落地）

> \emph{We acknowledge that, because path labels derive from final-answer matching and are broadcast to every step, a path that reaches the correct answer through a flawed step is nominally labeled positive, introducing step-level label noise. We therefore weaken our process-level claim: the encoder is not a localized-error detector but is trained toward step-level semantic consistency with the question. The benefit over outcome-level supervision is a denser training geometry rather than cleaner labels: step-level InfoNCE supplies $|\mathcal{P}|\cdot M$ positive pairs and aligns every step independently, so occasional step-level noise is averaged out under path- and prototype-level aggregation. Empirically the resulting alignment still predicts path correctness with an AUC of $0.78$ on GSM8K, indicating the label noise does not overwhelm the learned signal.}

**注意事项**：
- 与审稿人1 Q1 **同措辞、同数字（AUC 0.78，仅 GSM8K）**，避免两位审稿人对照出口径不一致。
- 主动承认标签噪声（接住审稿人的 7+2=8 例子），只声称"步-问语义一致性"，不声称"检测局部错误"。
- 配合审稿人1 Q1 表中列的四处 tex 软化（258/280/459/499 行）一起改。

---

## 审稿人 #7：术语使用不一致
- **问题**：sequence-level/path-level/chain/trajectory 混用；"reasoning chain / path / trajectory" 互换使用未说明是否同义。要求 problem formulation 定义统一术语，全文/公式/图/表一致。
- **状态**：草稿就绪（纯改文字，定义术语表 + 全文替换）；**不用实验**。

### 统一术语表（拟在 Problem Formulation 首次定义）
| 规范术语 | 含义 | 替换掉的混用词 |
|---------|------|---------------|
| **reasoning path** $r$ | 一次采样得到的完整 CoT（问题→若干步→答案） | reasoning chain, trajectory, sequence |
| **step** $s_i$ | 一条 path 内的单个推理步 | (保持) |
| **step-level representation** $z_{s}$ | 单个 step 的编码向量（训练目标） | — |
| **path-level representation** $z_{r}$ | 整条 path 的编码向量（step 向量池化，推理时用） | sequence-level representation |
| **dynamic prototype** $c_x$ | 逐题、相似度加权聚合的 path 原型 | (保持) |

**执行**：全文把 "reasoning chain / trajectory" 统一成 **reasoning path**；把 "sequence-level representation" 统一成 **path-level representation**（$z_r$）；step 级一律 "step-level representation"（$z_s$）。公式、图注、表头同步。

### 版本 A：Rebuttal 段落（英文，回信用）
> **Reply to Reviewer #2, Concern #7.**
>
> We thank the reviewer for catching this inconsistency. We have added a **terminology paragraph to the Problem Formulation** that fixes a single canonical vocabulary and use it consistently throughout the text, equations, figures, and tables. Specifically, we use **reasoning path** for a complete sampled chain-of-thought (replacing the interchangeable "reasoning chain" and "trajectory"), **step** for an individual reasoning step within a path, **step-level representation** ($z_s$) for the encoding of a single step (the training target), and **path-level representation** ($z_r$) for the pooled encoding of a whole path (used at inference); we no longer use "sequence-level representation." We have swept the manuscript to remove the mixed usages the reviewer identified.

### 版本 B：正文改动（tex 落地）
> \paragraph{Terminology.} Throughout, a \emph{reasoning path} $r$ denotes one complete sampled chain-of-thought for a question $x$, composed of \emph{steps} $\{s_1,\dots,s_M\}$. We write $z_{s}$ for a \emph{step-level representation} (the encoding of a single step, used as the contrastive training target) and $z_{r}$ for a \emph{path-level representation} (the pooled encoding of an entire path, used at inference). We use these terms exclusively and avoid the interchangeable use of ``reasoning chain,'' ``trajectory,'' and ``sequence-level representation.''

**注意事项**：定义一次、全文替换即可，无实验。落地时 grep `chain`/`trajectory`/`sequence-level` 逐个替换。

---

## 审稿人 #8：过度强调与 ORM 的范式差异
- **问题**：论文称 D-ProtoCoT 与 ORM "fundamentally different paradigm"。但从功能看两者都是用正/负标注路径训练辅助模型再排序/选择，只是打分形式不同（ORM 输出正确概率标量；D-ProtoCoT 用表示相似度/原型对齐）。要求更谨慎描述差异，别夸大成范式不同。
- **状态**：草稿就绪（软化 "fundamentally different"，承认同属监督排序，精确定位真正差异）；**不用实验**。

### tex 中需软化的位置
- **109行 / 302行**：`fundamentally different paradigm` → 改为承认同属 supervised path ranking，差异在打分形式与表示几何。
- （落地时 grep `fundamentally different` / `paradigm` 确认无遗漏。）

### 版本 A：Rebuttal 段落（英文，回信用）
> **Reply to Reviewer #2, Concern #8.**
>
> We agree with the reviewer's framing and have revised the manuscript to avoid overstatement. D-ProtoCoT and ORM indeed **share the same supervision and functional purpose**: both train an auxiliary model from positively- and negatively-labeled reasoning paths and use it to rank/select candidates. We no longer describe the two as a "fundamentally different paradigm." Instead we state the distinction precisely: ORM produces a per-path scalar correctness probability, whereas D-ProtoCoT scores a path by its **similarity to a question-specific dynamic prototype in a contrastively-learned representation space**. The practical consequences of this scoring choice — dense step-level positive pairs during training and a per-question adaptive selection criterion at inference — are what we now argue for, rather than a categorical difference in paradigm.

### 版本 B：正文改动（tex 落地）
> \emph{D-ProtoCoT and ORM share the same supervision signal (path labels from final-answer matching) and the same functional role (ranking candidate paths with a trained auxiliary model). The difference is in the scoring formulation rather than the paradigm: ORM emits a scalar correctness probability per path, whereas D-ProtoCoT scores paths by similarity to a question-specific dynamic prototype in a contrastively-aligned representation space. This yields dense step-level positive pairs at training time and a per-question adaptive selection criterion at inference, which is where our gains originate.}

**注意事项**：
- 主动承认"同监督、同功能"，接住审稿人的话，只把差异收在"打分形式 + 表示几何"。
- 与审稿人 #9（novelty）口径统一：真正贡献是 step 级稠密训练 + 逐题动态原型，不是"范式不同"。
- 删/改所有 "fundamentally different" 断言。

---

## 审稿人 #9：方法论新颖性需精确阐明
- **问题**：三个组件（对比微调 encoder、相似度加权、加权中心最近邻选择）各自在对比学习/原型建模里很常见。要求精确指出 novelty 到底在哪：是 outcome→step 监督传播、还是 step 训练/path 推理的不对称、还是动态原型、还是这些的组合。
- **状态**：草稿就绪（明确 novelty 定位）；条 4 granularity 消融数字回来后可补一句实证佐证。

### novelty 精确定位（三点组合，非单一组件）
1. **不对称的训练/推理粒度**：step-level 训练（$|\mathcal{P}|\cdot M$ 个正对，稠密监督）+ path-level 推理（池化后选择）。这是核心，条 4 的 granularity 消融（step/path 优于 path/path、step/step）正是它的实证支撑。
2. **逐题动态原型**：原型不是全局固定（对比 Static-Prototype 基线，见条 1），而是对每道题按相似度加权聚合当前采样路径——把 outcome 监督"传播"到测试时的 per-question 选择准则。
3. **outcome→step 表示传播**：仅用最终答案标签，却在 step 表示层得到稠密对齐（配合条 6 的 AUC 0.78 佐证学到的信号有效）。

### 版本 A：Rebuttal 段落（英文，回信用）
> **Reply to Reviewer #2, Concern #9.**
>
> We thank the reviewer and have sharpened the novelty statement. We do not claim any single component is new; the contribution is a **specific combination and the design choice that ties it together**. Concretely: (i) an **asymmetric training/inference granularity** — we train with a step-level contrastive objective (giving $|\mathcal{P}|\cdot M$ dense positive pairs from only outcome labels) but select at the path level after pooling; (ii) a **question-specific dynamic prototype** that aggregates the current candidate paths by similarity at inference, rather than a fixed global prototype; and (iii) the resulting **propagation of outcome-level supervision into step-level representations**. The asymmetric-granularity choice (i) is the primary contribution, and our granularity ablation (Concern #4) isolates it by showing that step-level training with path-level selection outperforms both symmetric variants. The dynamic prototype (ii) is what distinguishes us from the static-prototype baseline (Concern #1), which uses a single global centroid and degrades on CSQA/StrategyQA. We have rewritten the contributions paragraph to state this precisely and to avoid implying that the individual building blocks are themselves novel.

### 版本 B：正文改动（tex 落地，改 contributions/Method 引言）
> \emph{Our contribution is not any individual building block — contrastive encoder fine-tuning, similarity weighting, and centroid-based selection are each standard — but their combination under one design choice: an \textbf{asymmetric training/inference granularity}. We supervise at the step level (yielding $|\mathcal{P}|\cdot M$ dense positive pairs from outcome-only labels) yet select at the path level via a \textbf{question-specific dynamic prototype} that aggregates the current candidates rather than a fixed global centroid. This propagates outcome-level supervision into step-level representations while keeping selection adaptive per question. Our granularity ablation isolates the asymmetric design, and the contrast with a static-prototype variant isolates the dynamic prototype.}

**注意事项**：
- 明说"单个组件不新"，接住审稿人，把 novelty 收在**组合 + 不对称粒度设计**。
- novelty 的实证靠山：条 4（granularity 消融，证不对称有用）+ 条 1（Static-Prototype 对比，证动态原型有用）+ 条 6（AUC 0.78，证 outcome→step 传播有效）。**三条实验/证据是同一套 novelty 论证的支柱**。
- 条 4 数字回来后，把"granularity ablation outperforms both symmetric variants"换成具体百分比。

---

## 数据文件清单

| 文件 | 题目数 | 路径数 | 来源 |
|------|--------|--------|------|
| `gsm8k_merged_flat.jsonl` | 911 | 9110 | 官方训练集（xiaorong + database 合并） |
| `gsm8k_500_flat.jsonl` | 500 | 5000 | 官方训练集（xiaorong） |
| `csqa_500_flat.jsonl` | 500 | 5000 | 官方训练集（xiaorong） |
| `gsm8k_test_flat.jsonl` | 200 | 2000 | 官方测试集（LLM 已生成） |

## 代码文件清单

| 文件 | 作用 |
|------|------|
| `baseline/dprotocot/` | D-ProtoCoT 干净复现（9 个 py 文件） |
| `convert_data.py` | 分组 JSON → flat jsonl |
| `merge_gsm8k.py` | 合并两份 GSM8K 数据，去重验证 |
| `generate_gsm8k_test_cot.py` | 对官方测试集生成 CoT，支持断点续跑 |

---

## 待跑实验清单（本机服务器，`/home2/zzl/...` 默认路径）

> 本机跑：`run.py` 的 `--bert_model` 默认 `/home2/zzl/model/bert-base-uncased` 就是对的，**不用传**。数据全在 `newrundata/`。GSM8K 有独立 test（`--train_path`+`--test_path`）；StrategyQA/CSQA 单文件走比例切分。

| 条 | 实验 | 脚本 | 回应 |
|----|------|------|------|
| 3 | ORM 诊断 | `run.py orm` | ORM 太差是 bug（看 F1/AUROC） |
| 4 | granularity 消融 | `run.py granularity` | 主/消融不一致（修 Table 3） |
| 5 | leakage 消融 | `run.py leakage` | 答案泄露（full vs qa_only） |
| 1 | C-CoT 重跑 | `newrun/ccot_prompting.py` | 基线描述错（真 Chia prompting） |

### GSM8K / Qwen（示例）
```bash
# 条 3 ORM 诊断 → [ORM] TEST diagnostics: loss/path_acc/F1/AUROC + 四方法表
python baseline/dprotocot/run.py orm \
    --train_path newrundata/gsm8k_merged_flat.jsonl \
    --test_path  newrundata/gsm8k_test_flat.jsonl --epochs 10

# 条 4 granularity 消融 → 三组 path/path, step/step, step/path（同 test split，可比）
python baseline/dprotocot/run.py granularity \
    --train_path newrundata/gsm8k_merged_flat.jsonl \
    --test_path  newrundata/gsm8k_test_flat.jsonl --epochs 10

# 条 5 leakage 消融 → full / mask / qa_only 三种输入（看 full >> qa_only）
python baseline/dprotocot/run.py leakage \
    --train_path newrundata/gsm8k_merged_flat.jsonl \
    --test_path  newrundata/gsm8k_test_flat.jsonl --epochs 10

# 条 1 C-CoT 重跑（真 Chia 对比 prompting，非路径选择）
python newrun/ccot_prompting.py --dataset gsm8k \
    --data_path newrundata/gsm8k_test_flat.jsonl \
    --model_path /home2/zzl/model/Qwen3-8B \
    --output ccot_gsm8k_qwen.json
```

### 扩到完整 Table 1/3（3 数据集 × 2 模型）
- **StrategyQA**：单文件 + `--use_context`，无独立 test → 比例切分。
  ```bash
  python baseline/dprotocot/run.py orm \
      --data_path newrundata/strategyqa_flat.jsonl --use_context --epochs 10
  ```
- **CSQA**：单文件（`csqa_500_flat.jsonl`）；条 1 的 ccot_prompting 额外要 `--csqa_choices`（官方带选项文件）。
- **LLaMA 半张表**：把数据换成 `*_llama_flat.jsonl`（`gsm8k_llama_flat.jsonl` / `csqa_llama_flat.jsonl` / `strategyqa_llama_flat.jsonl`），`ccot_prompting.py` 的 `--model_path` 换 LLaMA-3.1-8B。

### 数据文件对照（`newrundata/`）
| 数据集 | Qwen | LLaMA | 独立 test |
|--------|------|-------|-----------|
| GSM8K | `gsm8k_merged_flat.jsonl`(911) | `gsm8k_llama_flat.jsonl` | `gsm8k_test_flat.jsonl`(200) |
| CSQA | `csqa_500_flat.jsonl`(500) | `csqa_llama_flat.jsonl` | 无（比例切分） |
| StrategyQA | `strategyqa_flat.jsonl` | `strategyqa_llama_flat.jsonl` | 无（比例切分，加 `--use_context`） |

---

## 附录：审稿人 #2 原话

The paper proposes D-ProtoCoT, a framework for chain-of-thought reasoning path selection. The manuscript explores reasoning-path selection from a representation-learning perspective and has a certain degree of research value. The reported results also suggest that D-ProtoCoT can outperform Self-Consistency under several settings. However, the current version still contains a number of substantial issues, including an incorrectly described baseline, unclear dataset usage, unexpectedly weak ORM results, inconsistencies between the main and ablation results, possible answer leakage, and insufficient support for the claim of process-level supervision. The novelty of the method also needs to be articulated more clearly. The following are some points which can be further improved in the new version. 

In its current form, I do not believe the manuscript is ready for acceptance. The authors should carefully address the above concerns, provide additional experiments, and revise the methodological claims accordingly.

1. The manuscript includes C-CoT as a comparison method and describes it as follows: “C-CoT (Chia et al., 2023) selects reasoning paths based on confidence estimation.” However, the method proposed by Chia et al. is actually a prompting-based approach that provides both valid and invalid reasoning demonstrations to help the language model avoid reasoning errors during generation. It is not a confidence-based candidate-path selection method. Therefore, the authors should clarify the exact algorithm referred to as C-CoT in this manuscript and provide the corresponding implementation details, code source, and correct reference. If a different confidence-based method was implemented, the method name and citation should be revised accordingly.

2. The authors state that D-ProtoCoT and ORM are trained on 1,000 samples from each dataset using an 8:1:1 train/validation/test split. However, the official training and test set sizes reported in the appendix are different from this setting. It is therefore unclear how the original dataset splits were used and on which subset the results in Table 1 were obtained. The authors should clearly explain the source of the 1,000 samples and the exact use of the training, validation, and test sets. I also recommend repeating the experiments using the official train/test splits to improve comparability and avoid possible data leakage.

3. The ORM baseline performs substantially worse than the other methods in several settings. For example, on GSM8K with Qwen3-8B, Standard CoT achieves 90.40% accuracy, whereas ORM obtains only 61.36%. On StrategyQA, ORM achieves only 50.43%, which is close to random guessing for a binary classification task. These unexpectedly poor results may indicate problems in the implementation or optimization of ORM. The authors should report the ORM training loss, validation loss, and path-level classification accuracy. The learning rate, number of training epochs, batch size, pooling strategy, and other hyperparameters used to reproduce ORM should also be provided. Additional diagnostic metrics, such as AUROC, F1 score, and the positive-to-negative sample ratio, would also be helpful.

4. The complete method achieves 64.29% accuracy on StrategyQA in Table 3. However, in Table 1, D-ProtoCoT achieves 86.20% and 72.60% on StrategyQA with the two backbone models. The result of 64.29% does not correspond to either backbone setting reported in the main experiments. The authors should clarify which backbone model, dataset split, number of test samples, and random seed were used for Table 3, and explain why the result differs substantially from the corresponding results in Table 1.

5. The positive and negative labels of reasoning paths are determined according to whether the final answer matches the gold answer. At the same time, the complete reasoning path, including the final answer, is used as input to the BERT encoder. Consequently, the encoder may learn shortcuts based on correlations between the question and the answer text while largely ignoring the intermediate reasoning process. The authors should provide additional ablation studies to demonstrate that the reasoning process itself contributes to the reported performance. For example, they could remove the final answer from each reasoning path, replace it with a common placeholder, or introduce a baseline that uses only the question and the final answer as input. If the question–answer-only baseline performs similarly to the full method, this would indicate that the current gains are not primarily attributable to reasoning-process modeling.

6. The authors claim that the step-level InfoNCE objective achieves process-level supervision granularity at the cost of outcome-level annotation. However, many reasoning paths may contain incorrect intermediate steps while still producing the correct final answer. For example, for the problem 7+2+1=10, a model may generate the reasoning “7+2=8; 8+1=10.” Although the intermediate calculations are incorrect, the final answer is correct. Under the current labeling strategy, all steps in this reasoning path would be treated as positive samples. Conversely, correct intermediate steps may be treated as negative samples if the final answer is incorrect. This label noise raises doubts about whether the proposed method genuinely learns step-level reasoning quality. The authors should provide step-level validation or adversarial examples with incorrect reasoning but correct final answers. Otherwise, the claim of process-level supervision should be substantially weakened.

7. The manuscript alternates between “sequence-level representation” and “path-level representation,” while “reasoning chain,” “reasoning path,” and “trajectory” are also used interchangeably without explicitly stating whether they refer to the same concept. Similar terminology inconsistencies appear in other parts of the manuscript. I recommend defining a unified set of terms in the problem formulation section and using them consistently throughout the text, equations, figures, and tables.

8. The authors emphasize that D-ProtoCoT is not an explicit correctness predictor, but instead learns a representation space, and is therefore fundamentally different from other methods. However, from a functional perspective, D-ProtoCoT still trains an auxiliary model using positively and negatively labeled reasoning paths and then uses the learned model to rank or select candidate paths. The main difference lies in the scoring formulation: ORM outputs a predicted correctness probability, whereas D-ProtoCoT uses representation similarity and prototype alignment. Therefore, the two approaches are closely related in terms of supervision and functional purpose, even though their optimization objectives and output forms differ. The manuscript should describe this distinction more carefully and avoid overstating it as a fundamentally different paradigm.

9. From an algorithmic perspective, the proposed method consists of three main components: contrastive fine-tuning of a text encoder, similarity-based weighting of reasoning paths with respect to the question, and nearest-neighbor selection using a weighted path centroid. Each of these components is relatively common in contrastive representation learning and prototype-based modeling. The authors should therefore clarify more precisely where the methodological novelty lies. In particular, it would be helpful to distinguish whether the main contribution is the propagation of outcome-level supervision to step-level representations, the asymmetric use of step-level representations for training and path-level representations for inference, the dynamic prototype construction, or the particular combination of these components.

Overall Recommendation
The manuscript explores reasoning-path selection from a representation-learning perspective and has a certain degree of research value. The reported results also suggest that D-ProtoCoT can outperform Self-Consistency under several settings. However, the current version still contains a number of substantial issues, including an incorrectly described baseline, unclear dataset usage, unexpectedly weak ORM results, inconsistencies between the main and ablation results, possible answer leakage, and insufficient support for the claim of process-level supervision. The novelty of the method also needs to be articulated more clearly.

In its current form, I do not believe the manuscript is ready for acceptance. The authors should carefully address the above concerns, provide additional experiments, and revise the methodological claims accordingly.

Recommendation: Major Revision / Reject in the current form.
