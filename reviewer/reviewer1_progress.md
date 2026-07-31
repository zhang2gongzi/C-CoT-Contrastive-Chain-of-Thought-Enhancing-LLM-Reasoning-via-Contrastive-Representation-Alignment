# 审稿人 #1 回复进度

---

## 审稿意见概述

D-ProtoCoT 的动机明确（PRM 需要昂贵的人工标注），但有以下担忧：

---

## 问题 1：步级标签噪声
- **问题**：监督仅来自最终答案匹配，路径级标签被广播到每一个步骤。包含有缺陷步骤但碰巧得出正确答案的路径会被视为正样本。这是 ORM 的典型失败模式，但本文却声称对局部逻辑错误具有敏感性。
- **状态**：待处理（只能文本软化，不能完全解决）
- **分析**：这条批评本质正确——我们的监督是 outcome-level（最终答案匹配），"碰巧答对但中间步骤有误"的路径确实会被当成正样本，存在标签噪声。这与 Reviewer #6/#8 是同一问题。不能靠实验完全消除，只能：
  1. **软化过强表述**：删掉/弱化"对局部逻辑错误敏感 (sensitivity to localized logical errors)"这类断言，改成"倾向于选择与问题语义对齐、整体连贯的路径"。
  2. **正面论证为何仍然有效**：InfoNCE 是 step-level 训练，把一条正样本路径的所有 step 拉近问题、负样本的 step 推远。即使个别 step 有噪声，聚合到 path/prototype 层面时噪声被平均稀释；而系统性错误的路径其多数 step 都会偏离，仍会被区分。
  3. **用数据佐证**：`analyze_reviewer1_q3.py` 的 **M4 AUC**（alignment 预测路径正确性）> 0.5 就说明，即便存在标签噪声，学到的表示仍能有效区分对/错路径——噪声没有摧毁信号。
- **计划**：论文正文改 claim（措辞软化）+ 引用 M4 AUC 作为"噪声下仍有效"的证据。无需新实验。

---

## 问题 2：对齐 ≈ 推理质量 仅被断言
- **问题**：建议在图 2 中添加训练后的对比可视化图，并补充定量指标（如相似度预测路径正确性的 AUC）
- **状态**：脚本已就绪，待服务器上跑（需 GPU）
- **分析**：审稿人要两样东西，缺一不可：
  - **(a) 定量指标 AUC**：alignment cosine 预测路径 correctness 的 AUC。
  - **(b) 训练后的 Figure 2 对照图**：现有 Figure 2 (`fig:tsne`, tex 第409行) 是**冻结 BERT** 对一个 StrategyQA 例子 100 条路径的 t-SNE，绿=对/红=错，结论是"混在一起，表面相似度不够"。审稿人要一张**训练后**的同款图，展示对比对齐后正确/错误路径被分开。
- **已做的计划/脚本**：
  1. **(a) AUC** → 复用 `newrun/analyze_reviewer1_q3.py` 的 **M4**（`M4_align_correctness_auc`，Mann-Whitney U AUC，无需 sklearn）。不用额外脚本。
  2. **(b) 对照图** → 新建 `newrun/viz_reviewer1_q2.py`：读同一份 `cot100.csv`，用**训练前(未训练 encoder)** vs **训练后(D-ProtoCoT encoder)** 并排画 t-SNE，两个 panel 同架构（path-level mean-pool），唯一差别是对比训练。额外报告**可分性数值**：余弦空间下 leave-one-out 1-NN 准确率(before/after)，对比多数类基线。
- **运行命令**（服务器）：
  ```bash
  python newrun/viz_reviewer1_q2.py \
      --train_path newrundata/strategyqa_flat.jsonl --use_context \
      --csv cot100.csv --epochs 10 \
      --out_png newrun/tsne_cot100_after.png
  ```
- **期望结果**：训练前 1-NN 准确率 ≈ 多数类（混在一起）；训练后明显升高（绿/红分开）。图 + M4 AUC 一起，把"alignment≈质量"从断言变成证据。
- **论文改动**：新增一张 after-alignment 图（或把 fig:tsne 改成 before/after 双 panel），正文加 AUC 数值 + 1-NN 可分性提升。

---

## 问题 3：相似度选择可能惩罚深度推理
- **问题**：按与问题的相似度选择路径，可能会惩罚那些远离问题原始措辞、引入新想法或进行非显而易见逻辑跳跃的路径，同时奖励只是重复问题的浅层路径
- **状态**：脚本已就绪，待服务器上跑（需 GPU）；最适合纯文本反驳 + 数据加强
- **分析**：这是概念性质疑，可以纯文本反驳，配数据更有力。反驳要点：
  1. D-ProtoCoT **不是**直接和 question 比相似度选路径，而是先把所有路径在**学习后的语义空间**里做相似度加权聚合得到 dynamic prototype，再选与 prototype 最对齐的路径。
  2. encoder 经过对比学习，度量的是**语义/推理质量**，不是字面措辞匹配。
- **已做的计划/脚本**：`newrun/analyze_reviewer1_q3.py`，一次算出三个反驳指标（+ 附带问题2的 AUC）：
  - **M1 词面重叠 vs 被选中**：token Jaccard、q_coverage(|P∩Q|/|Q|)。若选中路径重叠**不高于**未选中 → 不是靠"重复问题"选。
  - **M2 推理深度 vs 被选中**：#steps、#tokens、novel_ratio(|P\Q|/|P|，新词比)。若选中路径**不更短/不更浅** → 不惩罚深度推理。
  - **M3 语义≠字面**：Pearson(align_cosine, lexical_jaccard)。相关性**低** → 学到的是语义空间不是措辞。
  - **M4 AUC**：align 预测正确性（顺带解决问题2的(a)）。
- **运行命令**（服务器）：
  ```bash
  python newrun/analyze_reviewer1_q3.py \
      --train_path newrundata/gsm8k_merged_flat.jsonl \
      --test_path  newrundata/gsm8k_test_flat.jsonl \
      --epochs 10 --output newrun/reviewer1_q3_gsm8k.json
  ```
- **期望结果**：M1 选中 ≤ 未选中；M2 选中 ≥ 未选中；M3 低相关；M4 > 0.5。
- **论证链**：prototype 是路径在学习后语义空间的加权聚合 → 选中路径词面重叠不高(M1)、深度不低(M2)、align 与字面弱相关(M3)、且 align 能预测正确性(M4) → 相似度追踪的是推理质量而非表面措辞。
- **论文改动**：Related Work / Method 或 rebuttal 中加一段，用 M1–M3 反驳"惩罚深度推理"。

---

## 总结：三条意见的解决路径
| 意见 | 能否纯文本 | 需要的实验/图 | 脚本 | 状态 |
|------|-----------|--------------|------|------|
| #1 步级标签噪声 | 部分（软化 claim） | 无（引用 M4 AUC 佐证） | analyze_reviewer1_q3.py | 待跑 |
| #2 对齐≈质量仅断言 | 否 | (a) AUC + (b) 训练后 Figure 2 | analyze(M4) + viz_reviewer1_q2.py | 待跑 |
| #3 相似度惩罚深度 | 是（数据加强） | M1/M2/M3 分析 | analyze_reviewer1_q3.py | 待跑 |

两个脚本都在 `newrun/`，都在服务器上跑（要用 GPU 训 encoder）。跑完把 JSON/终端数字 + 图发给 Claude，写进 rebuttal 和论文。

---

## 附录：审稿人 #1 原话

Reviewer #1: This paper proposes D-ProtoCoT, an inference-time framework for selecting chain-of-thought reasoning paths. Instead of fine-tuning the language model, it trains a lightweight auxiliary encoder with a step-level InfoNCE objective (supervised by gold-answer matching) to build a representation space in which correct paths align with their question. At inference, it aggregates the sampled paths into a similarity-weighted dynamic prototype and selects the path best aligned with it.

Motivation is clear which PRM requires expensive human label is well-known problem but I do have some concerns.

1. Supervision comes only from final-answer matching, with path-level labels broadcast to every step. How do you ensure the selected path's intermediate steps are correct, e.g., a path with flawed steps that reaches the right answer by coincidence (and is thus treated as positive)? This is ORM's failure mode, yet the paper claims sensitivity to localized logical errors.

2. The premise "alignment ≈ reasoning quality" is asserted but not demonstrated. I recommend adding a post-training counterpart to Figure 2 (representation space after contrastive alignment), plus a quantitative measure (e.g., AUC of similarity predicting path correctness), to make the claim convincing.

3. My main concern is conceptual. D-ProtoCoT picks the path that is most similar to the question, and the prototype weights favor paths that stay close to it. But good reasoning often moves away from the wording of the question, bringing in new ideas not in the prompt or making non-obvious jumps. Selecting by similarity may punish these deep but less-similar paths, while rewarding shallow ones that just repeat the question. Does choosing paths by similarity to the question hurt the model's deeper thinking, favoring paths that stay on topic over paths that actually reason better?
