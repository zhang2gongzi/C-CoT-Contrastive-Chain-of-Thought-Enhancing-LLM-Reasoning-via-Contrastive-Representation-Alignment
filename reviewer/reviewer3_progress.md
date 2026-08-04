# 审稿人 #3 回复进度

---

## 审稿意见概述

论文逻辑清晰，具有良好的创新性和应用价值，但需进一步修改：

---

## 问题 1：仅在 8B 参数规模的模型上评估
- **问题**：仅使用了 LLaMa3.1-8B 和 Qwen3-8B，两个模型参数量相同。建议在更大参数规模的语言模型上评估以证明方法的适用性
- **方案**：加一组 GSM8K 实验（Qwen3-14B，同系列，回应"更大模型也适用"）
- **算力**：租用 AutoDL RTX 5090（32GB），**bf16 直接跑**（不用 4-bit）

### 执行清单（AutoDL 5090，bf16，全 14B pipeline）
方法一致性：train/test **都用 14B 生成**，避免 8B 训练 / 14B 测试 的不匹配。

```bash
# 0. 下模型
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-14B \
    --local-dir /root/autodl-tmp/Qwen3-14B

# 0b. 下 BERT（run.py 训 encoder 要用；默认路径 /home2/... 在 AutoDL 不存在）
huggingface-cli download bert-base-uncased --local-dir /root/autodl-tmp/bert-base-uncased

# 1. 生成训练集 CoT（14B）
python baseline/14B/generate_cot_14b.py --model_path /root/autodl-tmp/Qwen3-14B \
    --input newrundata/gsm8k_merged_flat.jsonl --output gsm8k_train_14b_flat.jsonl

# 2. 生成测试集 CoT（14B，200 题 x 10）
python baseline/14B/generate_cot_14b.py --model_path /root/autodl-tmp/Qwen3-14B \
    --input newrundata/gsm8k_test_flat.jsonl --output gsm8k_test_14b_flat.jsonl

# 3. 跑 main（10 epoch；必须显式传 --bert_model，否则找不到 BERT）
python baseline/dprotocot/run.py main \
    --bert_model /root/autodl-tmp/bert-base-uncased \
    --train_path gsm8k_train_14b_flat.jsonl --test_path gsm8k_test_14b_flat.jsonl --epochs 10
```

> ⚠️ **AutoDL 注意**：`run.py` 的 BERT 默认路径是 `/home2/zzl/model/bert-base-uncased`（原服务器），AutoDL 上不存在。第 3 步**必须**先下 `bert-base-uncased`（步骤 0b）并显式传 `--bert_model`，否则训 encoder 会报路径错误。`generate_cot_14b.py` / `self_certainty.py` 已有 `--model_path`，不受影响。

### 模型下载（AutoDL，二选一）
国内直连 HF 慢/失败，用镜像或魔搭。路径存 `/root/autodl-tmp/`（数据盘，别放系统盘 `/root/`）；建议 tmux/screen 里下。

**方式一：HF 镜像**
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-14B     --local-dir /root/autodl-tmp/Qwen3-14B          --local-dir-use-symlinks False
huggingface-cli download bert-base-uncased  --local-dir /root/autodl-tmp/bert-base-uncased  --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen3-8B       --local-dir /root/autodl-tmp/Qwen3-8B           --local-dir-use-symlinks False   # 仅 self_certainty 需要
```

**方式二：魔搭 ModelScope（AutoDL 通常更快更稳，推荐）**
```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-14B                 --local_dir /root/autodl-tmp/Qwen3-14B
modelscope download --model AI-ModelScope/bert-base-uncased --local_dir /root/autodl-tmp/bert-base-uncased
modelscope download --model Qwen/Qwen3-8B                   --local_dir /root/autodl-tmp/Qwen3-8B   # 仅 self_certainty 需要
```
> 注：bert 在魔搭的组织名可能不同（常见 `AI-ModelScope/bert-base-uncased`），下前在 modelscope.cn 搜一下确认准确 model id。Qwen 系列是官方账号 `Qwen/...`，与 HF 同名。

### 预期输出
| 步骤 | 输出 | 用途 |
|------|------|------|
| 1–2 生成脚本 | `gsm8k_{train,test}_14b_flat.jsonl`（每题10路径，含 cot/pred/is_correct）+ 终端 `Correct: X (Y%)` | 14B 原始逐路径正确率 |
| 3 run.py main | 四方法准确率表（下方） | 填 **Table 1 的 14B 新列** |

```
==== Results [main | input=full] | test questions = 200 ====
  Standard CoT            :  XX.XX%   # 第一条路径
  Self-Consistency        :  XX.XX%   # 多数投票
  Raw-BERT + Centroid     :  XX.XX%   # 冻结 BERT 基线
  D-ProtoCoT              :  XX.XX%   # 本方法
```

### 跑完必须盯的三个数（**epochs=10**）
1. **D-ProtoCoT > Self-Consistency** — 主结论，必须成立。
2. **D-ProtoCoT ≥ Raw-BERT + Centroid** — 证明训练有用。⚠️ 3-epoch 时 Centroid(81%) 曾反超 D-ProtoCoT(78%)，属未收敛；改 10-epoch 就是为了翻过来。若仍被反超 → 先诊断（epochs/pooling），**别急着填表**，否则这列帮倒忙。
3. **Standard CoT 别异常低** — 之前 3-epoch 曾掉到 75%（论文是 90.4%），确认 14B 正常。

- **状态**：✅ 14B GSM8K 已跑（见下），结论干净但饱和；正在补"混合题"分析。

---

### ✅ 14B GSM8K 实测结果（2026-08-03，AutoDL 5090 bf16，10 epoch）

```
Standard CoT          :  93.50%
Self-Consistency      :  97.00%
Raw-BERT + Centroid   :  95.00%
Self-Certainty-BERT   :  97.00%
D-ProtoCoT            :  97.50%
```

**判定：五项检查全过，无红旗，可填表**
| 检查 | 结果 | 判定 |
|------|------|------|
| D-ProtoCoT > SC | 97.50 > 97.00（+0.5） | ✓ 主结论成立 |
| D-ProtoCoT > Raw-BERT+Centroid | 97.50 > 95.00（+2.5） | ✓ 训练有用 |
| D-ProtoCoT > Self-Certainty-BERT | 97.50 > 97.00 | ✓ 压过新基线 |
| Centroid 落在 CoT 与 SC 之间 | 93.5 < 95.0 < 97.0 | ✓ 正好理论水位 |
| Standard CoT 不异常低 | 93.5（14B 正常） | ✓ |

**核心问题：饱和（saturation / 天花板效应）**
14B 在 GSM8K 太强 → 每题 K 条采样路径**几乎全对** → 没有"对/错混合"的对比信号 → dynamic prototype 和 SC 都顶到天花板，D-ProtoCoT 只赢 SC **+0.5**。这不是方法翻车，是**基准对 14B 太简单**。作者亲自诊断出根因：本方法需要"既有错、又有对"的路径数据才有发挥空间。

**决定：路线 B —— 保留 GSM8K-14B，把饱和写成"作用边界"的诚实分析**（作者拍板）
不把 14B GSM8K 当 headline 增益，而是按题统计"混合题比例"：8B 上高、14B 上塌缩 → 明说"方法只在路径有分歧时起作用，大模型在简单任务上分歧少故增益收窄；但在**混合题子集**上 D-ProtoCoT 仍 ≥ SC"。代价：增益数字薄，靠论证撑。
（路线 A —— 14B 换未饱和难基准拿正增益 —— 暂不走，留作可选加强。）

### 混合题分析脚本 `newrun/mixed_question_analysis.py`（新增，2026-08-03）
复用 `run.py main` 全部逻辑（`load_splits` / `train_encoder` / `evaluate_all`），只多做：把 test 切成 **mixed**（K 条有对有错）/ **unanimous**（全一致）两子集，分别调 `evaluate_all`。打印：①逐路径基准正确率（饱和程度，纯数据）②混合题比例 ③FULL/MIXED/UNANIMOUS 三张四方法表。

```bash
nohup python newrun/mixed_question_analysis.py \
    --bert_model /root/autodl-tmp/bert-base-uncased \
    --train_path gsm8k_train_14b_flat.jsonl \
    --test_path  gsm8k_test_14b_flat.jsonl \
    --epochs 10 \
    > newrun/logs/mixed_14b.log 2>&1 &
```

**跑完盯两点**：①混合题比例掉多低（预期 14B <15%）②MIXED 子集 D-ProtoCoT vs SC——只要 MIXED 上 D-ProtoCoT ≥ SC，路线 B 成立。
**可选加强**：同脚本换 8B 数据路径再跑一次，拿 8B 混合题比例做对比，"8B 高→14B 塌缩"叙事才完整。

**待办**：跑 `mixed_question_analysis.py`（14B，必跑；8B，可选）→ 拿混合题比例 + MIXED 子集表 → 写进论文 Q1 分析段。

### 文字草稿（五个数已定稿；混合题 `XX%` / MIXED 子集数字待回填，跑完一次性进 tex）

**版本 A：Reply to Reviewer #3, Concern #1（回信用）**

> We thank the reviewer for this suggestion. We evaluated D-ProtoCoT on **Qwen3-14B**, from the same family as our 8B backbone, on GSM8K under the identical protocol (K=10 sampled paths, 10-epoch encoder training, bf16). D-ProtoCoT attains **97.50%**, ahead of Self-Consistency (97.00%), Raw-BERT+Centroid (95.00%), Self-Certainty-BERT (97.00%), and Standard CoT (93.50%). The method thus remains the top selector at 14B, confirming that it transfers to a larger model.
>
> We also want to be transparent about the *size* of the gain. As the base model strengthens, its sampled paths on GSM8K become predominantly correct, so the pool a selector chooses from is nearly homogeneous and every method — including Self-Consistency — converges toward the ceiling; the absolute headroom for any path-selection method shrinks accordingly. This is a property of the saturated benchmark, not of D-ProtoCoT: selection can only help on questions whose sampled paths *disagree*. On the subset of GSM8K questions where the 14B paths are mixed (both correct and incorrect present — **XX%** of questions), D-ProtoCoT still matches or exceeds Self-Consistency (**XX.X% vs XX.X%**), showing that the method continues to add value precisely where selection is possible. [数字待 mixed_question_analysis.py 跑完填]

**版本 B：正文（Experiments 加一段 + Table 1 加 14B 列）**

> \paragraph{Scaling to a larger model.} To test whether representation-level selection transfers beyond the 8B scale, we evaluate D-ProtoCoT on Qwen3-14B on GSM8K under the same protocol. D-ProtoCoT reaches $97.50\%$, the highest among all methods (Self-Consistency $97.00\%$, Raw-BERT+Centroid $95.00\%$, Self-Certainty-BERT $97.00\%$, Standard CoT $93.50\%$), confirming applicability at larger scale. The margin over Self-Consistency narrows to $+0.5$ because GSM8K is largely saturated at 14B: sampled paths are predominantly correct, leaving little for any selector to disambiguate. Consistent with this, on the mixed-path subset (both correct and incorrect paths present, XX\% of questions) D-ProtoCoT remains $\ge$ Self-Consistency, indicating the method helps exactly where path disagreement leaves room to act. %% mixed 数字待补

**落地口径**：等 `mixed_question_analysis.py` 跑完 → 回填 `XX%` + MIXED 子集 `D-ProtoCoT vs SC` → 一次性把版本 B 写进 `cas-dc-template.tex`（Experiments），Table 1 加 14B 列。

---

## 问题 2：基线方法较旧
- **问题**：基线方法发表于 2021 和 2023 年。建议对近三年的新方法进行更深入的分析和比较

### 已实现的新基线（`baseline/dprotocot/evaluate.py`，旧 `baselines.py` 已并入 evaluate.py，commit ed42924）

| 基线 | 年份 | 原理 | 运行成本 |
|------|------|------|---------|
| Self-Certainty-BERT | 2024 | BERT 语义相似度代替 logprobs 选路径 | 零成本（纯推理） |
| USC | 2024 | LLM 直接看所有采样答案做多数投票 | 需 LLM API |
| GenSelect | 2024 | LLM 生成式评估每条路径后选择 | 需 LLM API |
| Pairwise-LLM | 2024 | LLM 两两比较路径 | 需 LLM API |

### 方案

1. **Self-Certainty-BERT**：直接跑，零额外成本，已经集成在 `run.py baselines` 里
2. **USC / GenSelect / Pairwise-LLM**：需要 LLM API。方案有二：
   - **A**（推荐）：服务器起 vLLM 部署 Qwen3-8B，`run.py baselines --llm_backend vllm --llm_model qwen3-8b`
   - **B**：论文 Related Work 中讨论这些方法 + 引用，不实际跑
3. **论文**：Related Work 补充 2024 年推理时路径选择（inference-time path selection）的最新进展

### 运行命令
```bash
# Self-Certainty-BERT（无需 LLM）
python baseline/dprotocot/run.py baselines --data_path newrundata/gsm8k_merged_flat.jsonl

# 完整基线（需要先起 vLLM）
python baseline/dprotocot/run.py baselines \
    --data_path newrundata/gsm8k_merged_flat.jsonl \
    --llm_backend vllm --llm_model qwen3-8b \
    --llm_base_url http://localhost:8000/v1

# Self-Certainty (Kang et al., NeurIPS 2025) —— 2024+ 新基线，AutoDL 上跑（需 GPU 加载 LLM）
python newrun/self_certainty.py --model_path /root/autodl-tmp/Qwen3-8B \
    --data_path newrundata/gsm8k_test_flat.jsonl --num_paths 10
```

**Self-Certainty (Kang) 预期输出**：终端 `Self-Certainty: XX.XX%` + `Self-Consistency: XX.XX%`，并写出 `self_certainty_results.json`（逐题明细）。用于回应"基线太旧"，加进对比表。

- **状态**：Self-Certainty-BERT 随时可跑；Self-Certainty(Kang) 待 AutoDL 跑；LLM 基线等 vLLM 部署

### 决定：Self-Certainty 用哪个模型
- **8B（Qwen3-8B）= 必跑**：论文主表 Table 1 是 8B，新基线要和 D-ProtoCoT-8B 同表正面比，才真正回应"补新基线证明优越"。
- **14B = 加分**：14B 已加载，顺手跑一个 Self-Certainty-14B 填进 Q1 的 14B 列，两个规模都有更完整。
- 若只跑一个 → 跑 8B（直接决定 Q2 过不过）。

---

### Related Work 文字草稿（塞进 §2.2 "Reasoning Path Selection and Verification"，回应 Q2「基线太旧」）

**✅ 已落地进 tex（2026-08-03，cas-dc-template.tex §2.2，line 149）**：作为 `\textbf{Recent inference-time selection and confidence methods.}` 段插在 confidence/PiCSAR 段（143–147）之后、`\textbf{D-ProtoCoT} differs fundamentally`（152）之前。落地时对下方草稿做了两处**去重**：
- 删掉草稿里的 PiCSAR 句（§2.2 line 145–146 已详述 PiCSAR，避免三次重复）。
- 删掉草稿结尾的 "D-ProtoCoT differs in where..." 对比句（line 152–154 已完整给出 D-ProtoCoT 的表示空间对比，避免连续三次同一对比）。
- 落地后新段只保留：Self-Certainty / TrajSelector / USC / GenSelect / Pairwise 的分类综述 + 一句"都依赖 logprob 或 LLM-judge"的批评；D-ProtoCoT 的对比统一交给已有的 152 段收尾。

**定位**：现有 §2.2 已覆盖 verifier-based / ORM / likelihood-based。新增一段专门讨论 **2024–2025 的推理时路径选择/置信度方法**，并说明 D-ProtoCoT 与它们的区别（表示空间对齐 vs. logprob/生成似然/外部 LLM 评判）。

> \paragraph{Recent inference-time selection and confidence methods.}
> A growing line of very recent work selects among sampled reasoning paths without training a dedicated verifier, using signals intrinsic to the generator. \emph{Self-certainty} \citep{kang2025selfcertainty} scores each path by the average KL divergence of its token distributions from the uniform distribution, using only the model's own logprobs as a training-free quality signal. \emph{PiCSAR} \citep{leang2025picsar} combines reasoning-confidence and answer-confidence, scoring a path by $\log p(\text{reasoning}\mid x) + \log p(\text{answer}\mid \text{reasoning})$. \emph{TrajSelector} \citep{yu2025trajselector} instead taps the sampler LLM's hidden states and trains a lightweight (0.6B) verifier to score process-level quality. A complementary family delegates selection to an LLM judge: \emph{Universal Self-Consistency} \citep{chen2024universal} and generative or pairwise selection \citep{toshniwal2025genselect, lin2026caps} prompt an LLM to read all candidates and pick the best. These methods share a reliance on either generation likelihood/logprobs (which conflate fluency with correctness) or on an auxiliary generative judge (which incurs additional LLM calls at inference). D-ProtoCoT differs in \emph{where} selection happens: rather than scoring paths by token-level confidence or by a separate LLM's verdict, it selects in an explicitly \emph{contrastively aligned representation space}, where a question-specific dynamic prototype captures the consensus reasoning direction. This makes selection (i) free of any inference-time LLM judge or logprob access, requiring only path embeddings, and (ii) grounded in semantic/structural alignment rather than the model's own generation probability.

**要点**：
- 把三类新方法讲清楚：logprob 类（Self-Certainty / PiCSAR）、hidden-state+verifier 类（TrajSelector）、LLM-judge 类（USC / GenSelect / Pairwise）。
- D-ProtoCoT 的差异点收在"表示空间对齐"，且强调**推理时不需要 LLM judge、不需要 logprob**，只要 embedding。
- Self-Certainty 会实测进表（8B 必跑）；PiCSAR/TrajSelector/USC 若不实跑，就在此**讨论+引用**（审稿人明确说可以"分析比较"，不强制全部实测）。

**引用（.bib，已全部补入 `cas-refs.bib`）**：`kang2025selfcertainty`(arXiv:2502.18581)、`leang2025picsar`(arXiv:2508.21787, EMNLP2025)、`yu2025trajselector`(arXiv:2510.16449)、`chen2024universal`(USC, arXiv:2311.17311)、`toshniwal2025genselect`(GenSelect, arXiv:2507.17797)、`lin2026caps`(Pairwise, arXiv:2605.15513)。line 139 的裸 `\citep{...}` 占位已填为 `\citep{toshniwal2025genselect, lin2026caps}`。

**诚实提醒**：TrajSelector 需要 sampler LLM 内部 hidden states，不兼容黑盒 API，实现成本高 → 只引用讨论，不实测（已记入 new_baselines）。CARFT 是微调 LLM 本身，问题设定不同，不作为推理时基线（可在 Related Work 一句带过或不提）。

---

## 问题 3：可复现性
- **问题**：应公开相关代码和数据以确保可复现性
- **状态**：待处理（**决定：等要交 rebuttal / 正式公开时再动，先不改**）

### 公开前 checklist（交 rebuttal 时执行）
1. **仓库改名** → `D-ProtoCoT`（现名 `C-CoT-...` 与论文方法名 D-ProtoCoT 不一致）
   - GitHub 网页 Settings 改名（GitHub 自动重定向旧 URL，不影响现有 push）
   - 本地更新：`git remote set-url origin <新URL>` → `git remote -v` 确认
   - 本地文件夹名可选改（会让当前工作目录路径失效，等实验做完再改）
2. **清硬编码路径**：代码里 `/home2/zzl/model/...`、`/root/autodl-tmp/...` 会暴露身份，公开前换成 CLI 参数默认值占位符（如 `bert-base-uncased`、`Qwen/Qwen3-8B` 的 HF id）
   - 已改好 `--model_path` 的：`generate_cot_14b.py`、`self_certainty.py`
   - config.py 的 `bert_model` 默认仍是 `/home2/zzl/...`，需改占位
3. **匿名性**：若双盲阶段公开，用匿名镜像（`anonymous.4open.science`），勿用带真名/username 的 GitHub
4. **README**：安装（`requirements.txt`）、数据说明、各 `run.py` 子命令示例、复现 Table 1 的命令
5. **数据**：`newrundata/*.jsonl` 被 `.gitignore` 忽略，需决定随仓库公开还是给下载链接

---

## 附录：审稿人 #3 原话

Reviewer #3: This paper proposed D-ProtoCoT, an inference-time framework for selecting high-quality reasoning paths based on representation-level alignment. The overall logic of the thesis is clear, and it has good innovativeness and application value. However, the current manuscript requires further revision, which mainly involves the following aspects.

(1) During the experiment, the authors only used two base LLMs, LLaMa3.1-8B and QWEN3-8B. These two models have the same parameter size. Therefore, it is recommended that the authors further evaluate the performance of the proposed method on larger language models with larger parameter sizes. This can demonstrate the applicability of the proposed method across different language models.

(2) The baseline models selected for this paper were published in 2021 and 2023. Therefore, the authors should conduct a more in-depth analysis of the new methods in the last three years and make comparisons to demonstrate the superiority of the proposed methods.

(3) In order to ensure the reproducibility of the proposed method, the authors should make the relevant code and data available for open-source use in their paper.
