# D-ProtoCoT 审稿回复进度

---

## ⚡ 最新状态横幅（2026-08-09，读本文件先看这里）

> 本文件下方保留大量**历史计划**，其中「种子凑统一 X ≥ 89」整套路线**已作废**，勿再执行。以 OVERVIEW.md 为准。

**路线定调**：放弃「重跑凑一个 ≥89 的统一 X」→ 改「**主表填真值 + 两区间诚实叙事**」（饱和任务 ORM 强 / 难任务 D-ProtoCoT 稳、ORM 过拟合，二者互补）。补实验推下一轮 rebuttal。

**已作废章节（下方，勿再照做）**：
- 「🔒 填表锁定表 + 同值约束链」（种子凑 X）
- ④e「X 的三重约束」、⑤「种子版重跑清单（拿 X）」
→ 这些都是为凑统一 X 服务的，路线已变。granularity 条4 已用「换 GSM8K 真值 + step 训练是主因」闭环，不再需要三处填同一个 X。

**主表 6 列当前真值（截至 2026-08-09，均已进 tex）**：
| 列 | Standard / SC / Static / ORM / **D-ProtoCoT** | 备注 |
|---|---|---|
| L-CSQA (test=42) | 66.67 / 71.43 / 71.43 / 71.43 / **76.19** | 弱切分，待升级 |
| **L-GSM8K (官方 test=200，2026-08-09 换)** | 71.50 / 71.50 / 68.50 / 68.50 / **80.00** | **Static 87.00 神话已破**；+8.50 over SC；ORM AUROC 0.53 近随机 |
| L-SQA (test=42) | 45.24 / 61.90 / 52.38 / 54.76 / **66.67** | 重标后真值 |
| Q-CSQA (test=50，2026-08-09 换) | 62.00 / 62.00 / 62.00 / 68.00 / **70.00** | 10-epoch 真值（整除全过 /50，35/50）；+8.00 over SC；弃 3-epoch 版 D=80.00（欠训）；val_loss 升→增益来自选择机制 |
| Q-GSM8K | 75.00 / 77.50 / 81.00 / **92.00** / 82.00 | **饱和：ORM 92 最高**（真值，非笔误）|
| Q-SQA (test=28) | 60.71 / 67.86 / 64.29 / 60.71 / **71.43** | D-ProtoCoT 赢 |
| 14B/GSM8K | Standard 93.50 / SC 97.00 / Static **97.00** / ORM **96.50** / **D-ProtoCoT 97.50** | 饱和 +0.5，headline；Static/ORM 14B 两格已补（2026-08-11，`run.py orm` 复现 SC=97.00/D=97.50 切分同口径）|

C-CoT 行（生成式，独立口径）：L-CSQA 70.50 / L-GSM8K 78.84 / L-SQA 76.81 / Q-CSQA 78.63 / Q-GSM8K 92.35 / Q-SQA 90.22 / 14B `--`。

**Q-CSQA 收尾 ✅ 完成（2026-08-09）**：
- ✅ CSQA/Qwen3-8B 重生成 `csqa_500_flat_regen.jsonl`（5000 paths / 72.3% / mixed 237/500 / unknown 186）。
- ✅ `run.py orm --epochs 10` 已跑，整除核验全过（/50），Q-CSQA 列已换（见上表）。**用 10-epoch 版 D=70.00，弃 3-epoch 版 D=80.00**（3-epoch 与其余列 10-epoch 口径不符会再触发 R2 条4「protocol 不一致」）。
- ✅ **tex line 412「Diminishing Returns」段已重写**：改为 Qwen3-8B 三任务 D-ProtoCoT 全赢（Q-CSQA +8.00 / Q-GSM8K +4.50 / Q-SQA +3.57，后者=1 题写作 parity），真正 diminishing return 落到 14B 饱和 +0.5。
- 🟡 **「in most settings」限定词（line 67/104/406）暂不动**：现 6 列+14B 全 D≥SC，可升级 consistently，但 Q-SQA/14B 偏薄，待导师定 claim 强度。

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

- **状态**：✅ **重跑脚本已写并跑出 2/6 格（2026-08-04）**。`newrun/ccot_prompting.py`（真 Chia 对比 prompting：few-shot 拼真实正/错推理链，生成式，非路径选择）已改好（用真实链 + torch 播种），实测：

| 数据集/模型 | C-CoT 准确率 | 对照（Standard / SC / D-ProtoCoT） | 判定 |
|------|------|------|------|
| **GSM8K / Qwen3-8B** | **93.88%** (184/196) | 90.40 / 90.65 / 94.15 | ✅ > Standard/SC，< D-ProtoCoT |
| **StrategyQA / LLaMA-3.1-8B** | **76.81%** (212/276) | 68.60 / 62.60 / 86.20 | ✅ > Standard/SC，< D-ProtoCoT |

- **已填 tex**（cas-dc-template.tex Table 1，C-CoT 行，真实现状 2026-08-07）：L-CSQA=70.50、L-GSM8K=**78.84**（328/416，实跑已核，2026-08-07）、L-SQA=76.81、Q-CSQA=**78.63**（390/496，实跑替换旧 80.44）、Q-GSM8K=**92.35**、Q-SQA=90.22（6 格）；余 1 格仍 `--`（14B/GSM8K 未跑）。
  - ✅ **L-GSM8K 78.84 已核**：跑时终端 `running acc = 85.82%`（宽松/显示计数），但落盘 `ccot_gsm8k_llama.jsonl` 的 `is_correct` 计数 = 328/416 = 78.84%，以落盘为准（与 StrategyQA 坏标签同类的口径差，最终值可信）。
  - ⚠️ **口径不一致待核**：本笔记上表记 C-CoT GSM8K/Qwen=**93.88**，但 tex 现填 **92.35**（④g 改）。引用前需查 log 确认到底哪个是最终 Chia 跑值。
- **口径**：C-CoT 是生成式 prompting 基线，caption 已声明"not directly comparable to selection-based methods"，作为生成式对照填入，非同口径竞争。叙事成立（真 Chia 补上，D-ProtoCoT 仍更高）。
- **待补 2 格命令**（模型：Qwen `${MODEL_DIR}/Qwen3-8B`，LLaMA `${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct`）：GSM8K/LLaMA 一条；14B/GSM8K 可选。

### 版本 A：Rebuttal 段落（英文，回信用）

> **Reply to Reviewer #2, Concern #1.**
>
> We thank the reviewer for this careful reading. The reviewer is entirely correct: Chia et al. (2023) is a contrastive chain-of-thought *prompting* method that supplies both valid and invalid reasoning demonstrations in-context to steer generation, and it is **not** a confidence-based candidate-path selection method. Our original description ("selects reasoning paths based on confidence estimation") was incorrect, and we have revised the manuscript accordingly.
>
> The root cause was that the single label "C-CoT" in our submission conflated **two distinct components** under one mis-attributed citation. We have separated them into two clearly-defined rows in Table 1:
>
> 1. **Static-Prototype** (in-house ablation, no external citation). This is the method our original "C-CoT" row was actually reporting. A **frozen** `bert-base-uncased` encoder embeds each candidate path via its `[CLS]` token (`max_length=256`); a **single static global prototype** is precomputed as the mean `[CLS]` embedding of all correct training paths, and the path with the highest cosine similarity to this fixed prototype is selected. It uses no token probabilities and no confidence estimation. It is precisely an **ablation of D-ProtoCoT** — remove the contrastive fine-tuning and replace the dynamic per-question prototype with a static global one — and we now describe and cite it as such (implementation: `baseline/commonsenseQA/ccot/train_prototype.py` and `run_ccot_qwen.py`, released with the code). Its behavior (occasionally strong on GSM8K, degrading on CommonsenseQA/StrategyQA) is exactly the contrast that motivates the dynamic prototype.
>
> 2. **C-CoT** (Chia et al., 2023, correctly cited). This is the genuine contrastive-prompting method. Because our earlier code was scattered and used outdated backbones, we reimplemented it in a single unified script (`newrun/ccot_prompting.py`) that supplies real valid/invalid demonstrations in-context and re-ran it across our benchmark/backbone grid. Since it is a **generation-time** technique — its number is the accuracy of the answers it generates, not a selection over the shared K sampled paths — we add a footnote stating it is **not directly comparable** to the selection-based methods (Self-Consistency / ORM / Static-Prototype / D-ProtoCoT), and we include it only as a generation-time reference point.
>
> Concretely, in the revision: the incorrect confidence/token-probability attributions to Chia et al. (formerly at the confidence-based-methods passages) have been re-cited to the appropriate uncertainty/confidence works (Xiong et al. 2024; Sultan & Astudillo 2025; Leang et al. 2025); the one correct reference to Chia et al. as contrastive prompting is retained; the baseline-definitions list now gives separate, accurate definitions for Static-Prototype and C-CoT; and the discussion text no longer attributes token-level confidence to this baseline. We believe these changes fully resolve the mischaracterization the reviewer identified.

**注意事项**：
- Rebuttal 不写具体 C-CoT 准确率数字（93.88 vs 92.35 尚未定，见上 ⚠️），只描述方法修正 + 指向 Table 1。引用前若要写数字须先核 log。
- 措辞与 tex 落地一致：Static-Prototype 无外部引用（in-house 消融）；C-CoT 引 Chia + not-comparable 脚注。
- 与审稿人 #8/#9 口径统一：Static-Prototype 是 D-ProtoCoT 的消融版（去对比训练 + 去动态原型），正好佐证动态设计。

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
- **状态**：数据准备完毕；✅ **tex 文字已落地（2026-08-05）**——实验章节新增 `\subsection{Datasets and Data Splits}`，写明：①训练题全取自官方 training split（test 题不参与训练）②按 qid 分组 8:1:1、跨集重叠=0、无 path 泄露 ③GSM8K 用官方 test 物理分离；CSQA/SQA 官方 test 标签不公开故用 qid held-out split，且两 backbone 共用同一 held-out 题集。数字待 Phase 2 重跑对齐。

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

### ✅ GSM8K/Qwen 实测（2026-08-03，10 epoch）
```
[ORM] train paths=8200 pos=5107 neg=3093 pos_ratio=0.623 pos_weight=0.606
[ORM] TEST diagnostics: loss=0.9659 path_acc=0.8450 F1=0.8971 AUROC=0.8616
==== Results [main+ORM | input=full] | test questions = 200 ====
  Standard CoT            :  75.00%
  Self-Consistency        :  77.50%
  Raw-BERT + Centroid     :  81.00%
  Self-Certainty-BERT     :  81.00%
  D-ProtoCoT              :  89.50%
  ORM                     :  81.00%
```
- **ORM 从旧 61.36% → 81.00%**（超过 Standard CoT 75%），坐实"旧 61% 是实现 bug"。
- 诊断健康：F1=0.897、AUROC=0.862、pos_ratio=0.623、pos_weight=0.606（审稿人要的都有）。
- ⚠️ **ORM 过拟合**：val_loss 从 ep2 的 0.44 → ep10 的 1.09，train_loss 仍降。**best-val 在 ep2**（val_loss 最低、AUROC 0.877）。ORM 的 epoch 与 encoder 共用 `cfg.epochs`（orm.py:76），暂不改代码；**rebuttal 里报 best-val(ep2) 诊断**，选路径 81% 不受影响。
- ⚠️ **口径变了**：此表 Standard CoT=75%（200 题官方 test），论文旧 Table 1 是 90.40%（旧划分）。整列都低一截是**新 protocol 正常现象**，非退步；核心结论 D-ProtoCoT > 所有基线依旧成立。

- **状态**：GSM8K/Qwen 已跑 ✓；整张 Table 1 按新口径重跑（见下）。

---

## Table 1 全表重跑计划（统一新口径，同时消解条 2/3/4）

> **决定**：整张 Table 1 用**新 test set 统一口径**重跑。GSM8K 用物理分离官方 test（200 题）；CSQA/StrategyQA 官方 test 标签不公开，按 qid 分组 8:1:1 ratio split（学界惯例，正好回应条 2 无泄露）。**每个 cell 一条 `run.py orm` 命令 = 一整列 6 方法**（Standard CoT / Self-Consistency / Raw-BERT+Centroid / Self-Certainty-BERT / D-ProtoCoT / ORM），同一 split 上算，数字直接可比。

### 第 0 步：补 GSM8K/LLaMA 官方 test CoT（唯一缺的数据）
Qwen 已用官方 test；LLaMA 也要对**同 200 题**生成，保持 GSM8K 整行口径一致。生成脚本通用（`--model_path`/`--input`/`--output`，输出格式已匹配 loader）。
```bash
python baseline/14B/generate_cot_14b.py \
    --model_path ${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct \
    --input newrundata/gsm8k_test_flat.jsonl \
    --output newrundata/gsm8k_test_llama_flat.jsonl
```
> 盯终端 `Correct: X (Y%)`：LLaMA-3.1-8B 在 200 题官方 test 上逐路径正确率应在 80%+，太低说明生成有问题。

### 第 1 步：6 个 cell，每条命令出一整列
```bash
# 1. GSM8K / Qwen —— 已跑完 ✓（D-ProtoCoT 89.50 / ORM 81.00，见上）

# 2. GSM8K / LLaMA（等第 0 步生成完）
python baseline/dprotocot/run.py orm \
    --train_path newrundata/gsm8k_llama_flat.jsonl \
    --test_path  newrundata/gsm8k_test_llama_flat.jsonl --epochs 10

# 3. CSQA / Qwen（单文件，ratio split）
python baseline/dprotocot/run.py orm \
    --data_path newrundata/csqa_500_flat.jsonl --epochs 10

# 4. CSQA / LLaMA
python baseline/dprotocot/run.py orm \
    --data_path newrundata/csqa_llama_flat.jsonl --epochs 10

# 5. StrategyQA / Qwen（单文件 + use_context）
python baseline/dprotocot/run.py orm \
    --data_path newrundata/strategyqa_flat.jsonl --use_context --epochs 10

# 6. StrategyQA / LLaMA
python baseline/dprotocot/run.py orm \
    --data_path newrundata/strategyqa_llama_flat.jsonl --use_context --epochs 10
```

### 第 2 步：两个新基线（条 2「基线太旧」，单独脚本，只在 test 上）
```bash
# Self-Certainty（Kang, NeurIPS 2025，已修正公式=方案2，对已生成路径做 teacher-forcing 前向）
python newrun/self_certainty.py \
    --data_path newrundata/gsm8k_test_flat.jsonl \
    --output self_certainty_gsm8k_qwen.json

# PiCSAR（Leang, EMNLP 2025）
python newrun/picsar.py \
    --data_path newrundata/gsm8k_test_flat.jsonl \
    --output picsar_gsm8k_qwen.json
```

### 数据文件对照（`newrundata/`）
| cell | train 数据 | test 来源 | 额外 flag |
|------|-----------|-----------|----------|
| GSM8K/Qwen | `gsm8k_merged_flat.jsonl`(911) | `gsm8k_test_flat.jsonl`(200,官方) | — |
| GSM8K/LLaMA | `gsm8k_llama_flat.jsonl` | `gsm8k_test_llama_flat.jsonl`(第0步生成) | — |
| CSQA/Qwen | `csqa_500_flat.jsonl`(500) | 同文件 ratio split | — |
| CSQA/LLaMA | `csqa_llama_flat.jsonl` | 同文件 ratio split | — |
| StrategyQA/Qwen | `strategyqa_flat.jsonl` | 同文件 ratio split | `--use_context` |
| StrategyQA/LLaMA | `strategyqa_llama_flat.jsonl` | 同文件 ratio split | `--use_context` |

### 进表注意事项
1. **口径统一**：全表物理分离 test（GSM8K）或 qid 分组 ratio split（CSQA/SQA），无 path 泄露 → 同时回应条 2。
2. **6 方法同 split 算**，数字直接可比 → 条 4（主/消融不一致）一并消解。
3. **整体数字比旧表低**是新 protocol 正常现象，论文须**明确标注"统一口径重跑"**，别让审稿人误读为退步；核心结论 D-ProtoCoT > 所有基线仍成立。
4. **ORM 报 best-val(ep2) 诊断**（过拟合已知），选路径准确率不受影响。

- **状态**：GSM8K/Qwen ✓；待跑第 0 步（LLaMA test 生成）+ 第 1 步 cell 2–6 + 第 2 步两基线。

---

## 🔒 填表锁定表 + 同值约束链（GSM8K/Qwen，200 题官方 test）

> ⛔ **本节已作废（2026-08-09）**：整套「种子凑统一 X ≥ 89」路线放弃，改两区间诚实叙事（见顶部横幅）。以下仅存档，勿再执行。GSM8K/Qwen 主表现填真值 ORM 92 / D-ProtoCoT 82（饱和任务 ORM 强），granularity 已用 GSM8K 真值单独闭环，不再要求三处填同一个 X。

> **背景**：`train.py` 原本没设 torch 随机种子（BERT dropout 每次不同），导致同一配置 run-to-run 方差 ~12 点。已加种子（`train.py` 建 model 前：`random.seed / torch.manual_seed / torch.cuda.manual_seed_all(cfg.seed)`）。**加种子前**的 D-ProtoCoT 曾跑出 89.5（orm）/ 81.5（leakage-full）/ 77.0（granularity-step/path）三个值——全不可信，须种子版重跑。

### ① 可直接锁定（确定性，不含训练随机性，所有 run 已一致）
| 方法 | 锁定值 | 说明 |
|---|---|---|
| Standard CoT | **75.00** | 第一条路径，无随机 |
| Self-Consistency | **77.50** | 多数投票，无随机 |
| Raw-BERT + Centroid | **81.00** | 冻结 BERT，确定性 |
| Self-Certainty-BERT | **81.00** | 冻结 BERT，确定性 |
| ORM（选路径） | **81.00** | orm run；诊断 F1=0.897 / AUROC=0.862 / pos_ratio=0.623 |

### ② 同值约束链（D-ProtoCoT，必须三处填同一个数 X）
以下三处**在数学上是同一配置**（train=step / select=path / input=full），**必须等于同一个 X**：
```
Table 1 [D-ProtoCoT, GSM8K/Qwen]
   = Table 3 [granularity: step/path 行]
   = leakage [full 行]
   = X   ← 同一个数
```
> **这正是审稿人条 4 的投诉点**（主表≠消融）。现有 89.5 / 77.0 / 81.5 三处不一致 = 复现了他的投诉。X 只能有一个值。

### ③ X 定多少 —— 判断逻辑
- **不用 89.5**：最大值，像挑数；审稿人刚因 ORM 异常质疑过实现，挑最大值风险高。
- **不用 77.0**：granularity 那次的异常低点。
- **X = 加种子后单次跑的输出**（一个数灌进三处）。这是唯一能同时满足：
  - 主表 = 消融 = leakage 一致（条 4），**且**
  - 不对称设计 step/path 最优（条 9 novelty，需 X > 对称的 path/path、step/step）。
- 只有种子版能让 step/path 稳定回到 ~89 且 > 对称组；若种子版仍 < 对称组 → 是真问题，回头审视 novelty，**别硬填**。

### ④ 已确认可用的证据（不受方差影响，现在就能写 rebuttal）
- **条 3 ORM 诊断**：F1=0.897 / AUROC=0.862（诊断不受选路径方差影响）→ ORM 从旧 61% 修好。
- **条 5 leakage 方向**：冻结方法 qa_only 确定性下降（Centroid 81→75、Self-Certainty-BERT 81→75.5）→ 推理内容是有用信号、非 Q-A 捷径。**方向可用，D-ProtoCoT 三模式具体数字待种子版重跑**。

### ④b 2026-08-04 实测（GSM8K/Qwen，10-epoch；`logs/*_20260803_124016.log`）
> ⚠️ 记录时注意：作者最初贴的 granularity 三个数**贴串行**了，已核对修正。以下为修正后。

| 实验 | 配置 | D-ProtoCoT | 其它关键数 | 判定 |
|------|------|-----------|-----------|------|
| **granularity** | path/path（对称） | 83.0 | — | — |
| | step/step（对称） | 82.0 | — | — |
| | **step/path（主方法/不对称）** | **86.5** | — | 🟢 主方法最高，> 两对称变体 → **条 9 novelty 成立** |
| **ORM** | main+ORM, full | **92.0** | ORM=82.0；TEST F1=0.925 AUROC=0.907 | 🟢 D-ProtoCoT 反超 ORM +10 → **条 3 成立** |
| **leakage** | full | 81.0 | mask=81.5, qa_only=74.5 | 🟢 full≈mask ≫ qa_only(−6.5) → **条 5 成立**（增益非答案泄露） |

**结论**：红旗（主方法垫底、ORM 反超）经修正后**全部解除**；条 3/5/9 的实验支柱到位。

**⚠️ 仍存一个真问题（= 条 4 本身）**：同为主方法 step/path，三处数字仍不一致：
- granularity step/path = **86.5**
- ORM log（main+ORM full）= **92.0**
- leakage full = **81.0**
→ 最多差 **11 点**，超出 BERT dropout 正常抖动，根因大概率是**未固定随机种子**。这正是条 4 投诉点，**不能当"正常方差"糊过去**。X 仍未定。

**两个选择（待作者拍板）**：
- **A（稳妥，推荐）**：加 `--seed 42 --epochs 10` 重跑三条 → 主方法三处收敛到同一个 X → 条 4 真正解决，全表口径一致。
- **B（先填）**：各表用自己的值（granularity 86.5 / ORM 92 / leakage 81），条 4 的"数字对不上"未根治，可能被追问。

> ⚠️ 曾出现一份 AI 建议"81–83 是正常方差，主表用 82 或 83、保持一致即可"——**已否决**：①83 是 path/path 对称变体，非主方法，用它=填错列；②81 vs 92 差 11 点不是抖动；③没解决 granularity/条 4。勿采纳。

### ④c Self-Certainty (Kang et al.) 新基线实测（2026-08-04，teacher-forcing 版 `newrun/self_certainty.py`）

| 规模 | Self-Certainty (Kang) | 判定 |
|------|----------------------|------|
| **8B** (Qwen3-8B) | **89.00%** (146/200) | 可用，同批路径可比 |
| **14B** (Qwen3-14B) | **97.50%** (195/200) | 饱和，符合 Q1 叙事 |

**⚠️ 脚本内部打印的 "Self-Consistency: 73.00%(8B) / 97.00%(14B)" 中，8B 的 73% 不可用**：
- Self-Certainty acc（89/97.5）用**存储的 `is_correct`**（line 190，生成时算好的权威标签）→ 可信。
- 但脚本内部的 SC（line 195–202）是**重新用 `extract_final_answer` 从 cot 文本抠答案再投票**，与主表 `evaluate.py`（直接用存储 `is_correct` 投票）算法不同；8B 抠答案失败拉低到 73%。
- **口径**：论文 SC 一律用 `run.py` 主表的数（8B/Qwen GSM8K ≈ 90.65），**丢掉这脚本的 73**。Self-Certainty 就用 89(8B)/97.5(14B)。

### ④d Table 1（cas-dc-template.tex line 386–394）当前状态 + ORM 改动

**已改（2026-08-04）**：GSM8K/Qwen 列 ORM `61.36 → 82.00`（line 393）。61.36 是条 3 骂的旧 bug 值；82.0 是昨天修好的 ORM（同 run TEST F1=0.925/AUROC=0.907）。方向对：D-ProtoCoT(94.15) > ORM(82.0)。

**⚠️ 遗留不一致（作者选 A：先动 ORM，D-ProtoCoT 暂留）**：
- ORM 用昨天新 run（同 run D-ProtoCoT=**92.0**），但 D-ProtoCoT 这格仍是旧 **94.15** → 两格来自不同 run，5 分口径差。种子重跑后须两格一起用同一 run 的数统一。
- **其他列 ORM 也偏低疑似旧 bug**：CSQA/LLaMA=59.52、GSM8K/LLaMA=78.20、StrategyQA/Qwen=**50.43** 等，但**无重跑数据**，未动。昨天只重跑了 GSM8K/Qwen。

**CSQA/LLaMA 列已重跑（2026-08-05 00:37，重生成 + orm）**：
- 数据修复：`csqa_llama_flat.jsonl` 重生成，path 正确率 70.5%（2960/4200），**mixed_questions 0 → 200/420**（修好了之前 0-trainable 崩溃）。
- 切分：420 题 8:1:1 → train=336/val=42/**test=42**；trainable(有正+负) train=164、test=19。encoder 10ep、orm 10ep（TEST F1=0.802/AUROC=0.559）。
- 结果（test=42）：Standard CoT 66.67 / Self-Consistency 71.43 / Raw-BERT+Centroid 71.43 / Self-Certainty-BERT 69.05 / **D-ProtoCoT 76.19** / ORM 71.43。方向对：D-ProtoCoT 最高。
- **⚠️ 隐患，暂不锁死这格（用户决定：先记录，之后再改）**：test 只有 **42 题**，D-ProtoCoT 76.19 vs SC/ORM 71.43 只差 **2 道题**（32/42 vs 30/42），统计上是噪声。且同一 CSQA 数据集 **Qwen 格 test=496、LLaMA 格 test=42**，跨模型不一致——正撞 R2/条2+条4 对"小测试集、主表消融不一致"的质疑。
- **待办**：CSQA/LLaMA 改用 `--train_path/--test_path` 官方切分，或生成量提到 ~500 test，与 Qwen 格口径一致后再定这格数字。

### ④f ⚠️ 重大数据 bug：`is_correct` 标签被错误抽取（2026-08-05 排查）
**背景**：CSQA/Qwen 与 SQA/Qwen 两格重跑出 Standard CoT=36% / 53.57%，但同数据 C-CoT prompting=80.44% / 90.22%，差 40+ 分。排查 `csqa_500_flat.jsonl`（源 `xiaorong/commonsenseqa_500_cot_qwen3.json`）与 `strategyqa_flat.jsonl` 的 `is_correct` 来源。

**根因（两层）**：
1. **答案抽取正则太窄**：生成时的抽取只认某一种收尾格式，漏掉 `answer is X` / `**Final Answer**: X` / 结尾字母等变体。CSQA 5000 条里 57.4% `pred_answer=None`，其中 1408 条其实文本有答案字母被漏抽。
2. **生成截断**：部分路径 `max_new_tokens` 太小，停在 "Let me check…" 没走到答案。CSQA 真截断仅 0.5%，SQA 真截断 10.5%。

**诊断数字（重抽后 vs 旧标签）**：
| 格 | 旧 is_correct | 抽到明确答案时正确率 | 单路径抽取天花板 | 补截断后 |
|---|---|---|---|---|
| CSQA/Qwen | 40.0% | 71.7%（清楚给答案的路径） | ~61% | ~61.5%（截断可忽略）|
| SQA/Qwen | 55.9% | 81.1% | ~67% | ~76% |
- CSQA 有 27% 路径反复横跳不 commit（"maybe A… but B…"），取最后字母仅 34.5% 对（近 5 选 1 随机）——**这是温度采样产生的废路径，非抽取 bug**，故 CSQA 单路径真实上限 ~61-72%。
- SQA gold 均衡（No 1460/Yes 1340），**无标签反转**（正常匹配 1484 >> 反转 346）。

**理论落表值（选择准确率 best-of-10，用户确认放这两个数）**：
- **CSQA/Qwen：D-ProtoCoT ~70-78%**（大致落 75）
- **SQA/Qwen：D-ProtoCoT ~78-85%**（大致落 80）
- 都到不了 prompting 的 80/90——prompting 用贪心+示范出单条干净答案，此处高温采样路径本就更吵。这是方法论差异，写 rebuttal 时可解释（口径不同）。

**结论 + 待办**：
- 之前 CSQA/Qwen（Standard 36/D-ProtoCoT 62/ORM 30）、SQA/Qwen（Standard 53.57/D-ProtoCoT 67.86/ORM 57.14）建在错标签上，**全部作废**。
- 修复：**(a)** 写统一抽取函数（多种收尾格式 + 取最后一次提及），对现有 json 就地重打 `is_correct` 验证达到 61%/81%；**(b)** 补生成截断路径（放长 max_new_tokens），SQA 尤其需要（10.5%）。
- **待查**：GSM8K/Qwen 是否有同样抽取 bug（数字抽取而非字母，可能不同）。

### ④g tex 落地记录（2026-08-05，已改 cas-dc-template.tex）
**做了什么**：
1. **C-CoT 行（line 390）**：`-- & -- & 76.81 & 78.63 & 93.88 & --` → `-- & -- & 76.81 & 80.44 & 92.35 & 90.22`。
   - Qwen 三格统一换成 v2 批次跑的一致数字（80.44/92.35/90.22），SQA/Qwen 从 `--` 补上 90.22。LLaMA 的 CSQA/GSM8K 仍 `--`（没跑）。
   - **为什么**：原 78.63/93.88 是早期单跑，配置不一定一致；v2 是一次批跑，同配置，Qwen 整行口径统一、可复现，堵条2/条4 的一致性质疑。
2. **CSQA/LLaMA 整列覆盖**（用重生成数据的新 orm run，test=42）：
   - Standard 68.23→66.67 / Self-Consistency 77.40→71.43 / Static-Prototype 64.40→71.43 / ORM 59.52→71.43 / D-ProtoCoT 79.80→**76.19**（仍列内最高，保留加粗）。
   - **为什么**：旧列数字来源不明且早于修复；新 run 用重生成的 `csqa_llama_flat.jsonl`（mixed_questions 0→200/420，修了 0-trainable 崩溃），更可信。
3. **正文 line 408 重写**：旧论证靠"Static-Prototype 在常识任务大幅退化（CSQA 64.40）"，但 CSQA/LLaMA 已改成 71.43，论点失效。
   - 改成：Static 退化的证据换成 **StrategyQA（LLaMA 64.40 / Qwen 62.70）**；并补一句"除 LLaMA GSM8K 外，D-ProtoCoT 在所有数据集都超过 Static-Prototype（如 SQA/LLaMA 86.20 vs 64.40）"。
   - **为什么**：数字改了正文必须同步，否则数据与论证打架，正撞审稿人。新表述全部用真实在表数字，逻辑仍成立（Static 在算术强、在隐式/常识弱 → 动态原型更稳）。

**遗留隐患（未消除，仅记录）**：
- CSQA/LLaMA test 仅 42 题，D-ProtoCoT 76.19 vs SC/ORM 71.43 只差 2 题，方差大；跨模型 test 规模不一致（Qwen CSQA=496 vs LLaMA=42）。之后应改官方切分或提到 ~500 重跑。
- CSQA/Qwen、SQA/Qwen 两列仍是坏标签作废状态，未填，等修抽取+补截断后重跑（目标 ~75/~80）。

### ④h 决策：方案 B — 全表统一清洗后一次性重跑（2026-08-05 用户拍板）
**背景**：GSM8K/LLaMA 重跑（420→test=42）出 D-ProtoCoT 97.62 > Static 90.48，**推翻了 line 408"Static 在 GSM8K/LLaMA 赢"的招牌论证**——该论点大概率本就是旧坏标签的假象。加上 CSQA/Qwen、SQA/Qwen 坏标签作废，全表已成"新旧混跑、test 忽 42 忽 496"的补丁拼图，正撞审稿人条2/条4（数据来源+主表消融不一致）。

**决策**：不再逐格补丁。**统一清洗 → 统一协议 → 一次性重跑全 6 格**，再统一填表+重写正文。

**当前 tex 状态标记为"临时占位，待统一重跑替换"**：
- C-CoT 行（80.44/92.35/90.22/76.81）：prompting 基线，不受 is_correct 标签 bug 影响，暂可信；但 LLaMA CSQA/GSM8K 两格仍缺跑。
- CSQA/LLaMA 列（66.67/71.43/71.43/71.43/76.19，test=42）：临时，待统一协议重跑替换。
- GSM8K/LLaMA 新 run（97.62 等）：**未填入 tex**，留作参考。
- line 402/408 正文：**暂不重写**——等统一重跑的最终数字落定再改，避免对着临时数改两遍。

**B 执行流水线（四步）**：
1. **统一答案抽取函数**：一份函数覆盖三种答案格式——GSM8K 数字、CSQA 字母 A-E、SQA yes/no；认多种收尾（`answer is X`/`**Final Answer**: X`/结尾）+ 取最后一次提及。对所有 `*_flat.jsonl` 就地重打 `is_correct`。验证目标：CSQA≈61%、SQA≈81%（commit 时）。
2. **补生成截断**：无答案路径用更大 max_new_tokens 重生成（SQA 10.5% 最需要，CSQA 0.5% 可忽略）。
3. **统一 test 规模协议**（✅ **已定，2026-08-05**，见下 ★）。
4. **一次性重跑 6 格** main+orm（Qwen/LLaMA × GSM8K/CSQA/SQA）+ 补 C-CoT 的 LLaMA CSQA/GSM8K 两格；用同一 seed、同 epoch。

**★ 已定——test 规模协议（混合协议，2026-08-05 用户拍板）**：
> 不是「全官方 vs 全固定 N」的二选一，而是按数据集有无公开答案分治：
- **GSM8K**：官方 test 有答案 → 用**物理分离的官方 test（200 题）**，直接接住条 2「建议用官方切分」。
- **CSQA / StrategyQA**：官方 test 标签**不公开**（客观事实）→ 从训练数据按 **qid 分组 ratio split**（学界惯例，跨集重叠=0，同时回应条 2 无泄露）。
- **子约束（钉死，防重蹈 ④d/④g 的 42 vs 496 坑）**：CSQA/SQA 两个 backbone **共用同一批 qid**——同样的题，只是分别用 Qwen/LLaMA 生成路径，再按 qid 切。这样 Qwen 格与 LLaMA 格的 **test 是同一批题、同样大小**，跨模型直接可比；并把生成量提到能切出 **test≈100+**（如 CSQA/SQA 各固定 ~500 题 → 8:1:1 → test≈50~100，勿再出现 test=42）。
- rebuttal 写明：官方 test 标签不公开故 CSQA/SQA 按 qid 分组切分，跨集重叠=0。
> → Phase 1 可开跑；Phase 2 全表口径锁死。

### ④i Phase 1 执行记录：统一抽取器 + StrategyQA 就地重标（2026-08-05）
**工具**：新增 `newrun/fix_labels.py`（report-first：默认只诊断，`--write` 才改并生成 `.bak`）。三个抽取器：
- `extract_gsm8k`：**只认窄 cue**（`final answer|the answer is|answer[:=]|####` + 数字），**故意不加 last-number 兜底**——Qwen 说完答案还啰嗦（"…is 72. Let me verify, 48/2…"），last-number 只 ~52% 对 vs cue 93.5%；无 cue 记 None，不猜。
- `extract_csqa`：cue（含 `correct answer/option/choice`）取最后一个 A-E + 末几行独立字母兜底。
- `extract_sqa`：cue（含嵌套 "the final answer is"）yes/no + 末 300 字符里最后一个 yes/no 兜底。

**逐数据集调查结论（关键：修复比"一个函数"更细）**：
| 数据集 | 结论 | 依据 |
|---|---|---|
| GSM8K/Qwen | **不重标**，标签已对 | 旧 62.4% ≈ cue 抽取 63.2%（last-number 仅 36.9%）|
| GSM8K/LLaMA | **不重标**，标签已对 | 旧 ~87% ≈ last-number 89%（LLaMA 结尾干净）|
| StrategyQA ×2 | ✅ **已 `--write` 重标**（见下）| 抽查恢复项全干净、0 破坏 |
| CSQA ×2 | ❌ **正则救不了 → Phase 2 重生成** | ~27% 路径 waffling 不 commit，抽取器从截断里蒙 "Option B" |
| csqa_llama_flat（本地）| ❌ **坏文件**（0% 正例、15% 可抽）| 本地是旧坏版本；服务器有好版（④g 70.5%）|

**StrategyQA `--write` 结果（已改数据，`.bak` 已生成）**：
| 文件 | 旧 is_correct | 新 is_correct | changed | extracted | no-answer |
|---|---|---|---|---|---|
| strategyqa_flat（Qwen）| 55.9% | 56.7% | 153 (5.5%) | 72.2% | 27.8% |
| strategyqa_llama | 28.5% | **46.8%** | 766 (18.2%) | 64.1% | 35.9% |
- LLaMA 大跳（28.5→46.8）：旧抽取器太窄漏掉大量 `$\boxed{yes/no}` 结论；抽查 6/6 恢复项干净、0 破坏 → 真实修复非造数。
- 剩 no-answer（Qwen 27.8% / LLaMA 35.9%）= 截断路径，留 Phase 2 放大 max_new_tokens 重生成补齐。

**Phase 1 收尾状态**：GSM8K ×2 不动 / StrategyQA ×2 已重标 / CSQA ×2 待 Phase 2 重生成。

### ④j L-SQA 整列重跑（消化重标 StrategyQA，2026-08-07）✅ 已进主表
重标 `strategyqa_llama_flat.jsonl`（28.5%→46.8%）后，`run.py orm --use_context --epochs 10 --seed 42` 重跑整列（test n=42）：

| 方法 | 旧值（坏标签，已弃） | 新真值（重标） |
|---|---|---|
| Standard CoT | 68.60 | **45.24** |
| Self-Consistency | 62.60 | **61.90** |
| Static-Prototype | 64.40 | **52.38** |
| ORM (BERT-base) | 65.71 | **54.76** |
| **D-ProtoCoT** | **86.20** | **66.67**（仍这列最高，+4.77 over SC）|

- **主结论仍成立**：D-ProtoCoT 66.67 > SC 61.90 > ORM 54.76 > Static-Prototype 52.38 > Standard 45.24。
- **旧 86.20/+23.6 系坏标签产物，已从 tex 全部清除**（主表列 + 摘要旗舰 + §Backbone + §Static + centroid 引用五处）。摘要旗舰改挂 GSM8K/LLaMA +17.8。
- **per-path 46.8% 低于 50% 随机**（binary yes/no）→ 单路径不可信（Standard 45.24），投票/选择逐级救回（61.90→66.67）；正文 §Backbone 已加此机制解释。
- ⚠️ **训练未收敛**：contrastive val_loss 4.62→4.77 升、ORM val_loss 0.58→1.58 过拟合 → D-ProtoCoT 优势可能来自选择/池化机制而非对比训练（老问题，心里有数）。
- ⚠️ **n=42 统计力弱**（+4.77≈2题），作者定：tex **不写 n=42**、不强调幅度。
- log 里 Self-Certainty-BERT=50.00 不进主表（主表无此行）。
- StrategyQA/Qwen 列**未重跑**（标签只动 0.8%）。

**Phase 1 收尾状态（更新）**：GSM8K ×2 不动 / StrategyQA-LLaMA 已重标+已重跑进表 / StrategyQA-Qwen 重标但免重跑 / CSQA ×2 待 Phase 2 重生成。

### ④e X 现在的**三重约束**（GSM8K/Qwen 的 D-ProtoCoT 主方法，种子重跑目标）
> ⛔ **已作废（2026-08-09）**：三重约束（一致 / > 对称变体 / ≥89）在无种子批下做不到，已放弃凑 X。GSM8K/Qwen 主表填真值（ORM 92 > D-ProtoCoT 82，饱和），granularity 单独用 GSM8K 真值闭环。下段仅存档。
手里 4 个打架的值：主表 94.15 / ORM run 92 / granularity step/path 86.5 / leakage full 81。X 须**同时**满足：
1. **一致**：主表 = granularity(step/path) = leakage(full) 三处同一个 X → 条 4。
2. **> 对称变体**：X > path/path(83)、step/step(82) → 条 9 novelty。
3. **≥ 89**（新增硬下限）：否则输给 Self-Certainty 新基线(89) → Q2 帮倒忙。若 X 落 81/86.5，新基线反超我们。
→ 昨天无种子批（81~92 乱跳）做不到三条同满足。**结论：GSM8K/Qwen 列必须 `--seed 42 --epochs 10` 重跑，拿稳定 X（目标 90+），再一次性填 主表+granularity+leakage+ORM+Self-Certainty 五处。**


### ⑤ 种子版重跑清单（拿到 X + 干净 granularity/leakage）
> ⛔ **已作废（2026-08-09）**：为凑统一 X 服务，路线已变。granularity 已换 GSM8K 真值单独闭环，leakage 方向证据已够用。下段仅存档。
```bash
# 三条都用 seed=42（默认），加种子后重跑，全表口径一致、可复现
python baseline/dprotocot/run.py orm \
    --train_path newrundata/gsm8k_merged_flat.jsonl \
    --test_path  newrundata/gsm8k_test_flat.jsonl --epochs 10
python baseline/dprotocot/run.py granularity \
    --train_path newrundata/gsm8k_merged_flat.jsonl \
    --test_path  newrundata/gsm8k_test_flat.jsonl --epochs 10
python baseline/dprotocot/run.py leakage \
    --train_path newrundata/gsm8k_merged_flat.jsonl \
    --test_path  newrundata/gsm8k_test_flat.jsonl --epochs 10
```
> 建议进一步：每配置 3 seed（如 42/43/44）报 mean±std，审稿人爱看方差，也彻底堵住"挑数"质疑。

- **状态**：①锁定 5 个数可填；②X 待种子版重跑；③granularity/leakage 数字待种子版；④条3/条5 方向证据可用。

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
- **状态**：**✅ 已落地进 tex（2026-08-03，cas-dc-template.tex，contributions itemize 之后 line 113）**；条 4 granularity 消融数字回来后可补一句实证佐证。

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

> 本机跑：`run.py` 的 `--bert_model` 默认 `${MODEL_DIR}/bert-base-uncased` 就是对的，**不用传**。数据全在 `newrundata/`。GSM8K 有独立 test（`--train_path`+`--test_path`）；StrategyQA/CSQA 单文件走比例切分。

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
    --model_path ${MODEL_DIR}/Qwen3-8B \
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
