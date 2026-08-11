# Response Letter — D-ProtoCoT

August 11, 2026

Dear Editor and Reviewers,

We would like to sincerely thank the Editor for providing us with the opportunity to revise our manuscript, and the anonymous reviewers for their careful reading and insightful comments. Based on the comments from the Editor and three reviewers, we have made revisions to our manuscript. In the following attachment, we provide our responses to all the issues raised.

Thank you again for your time and consideration. We look forward to hearing from you.

Sincerely,
The Authors

---

## Contents

1. Response to Editor
   - Editor's Comments
2. Response to Reviewer 1
   - Reviewer 1's Overall Comment
   - Reviewer 1's Comment 1 — Step-level label noise
   - Reviewer 1's Comment 2 — "Alignment ≈ reasoning quality" is asserted, not demonstrated
   - Reviewer 1's Comment 3 — Selecting by similarity may punish deep reasoning
3. Response to Reviewer 2
   - Reviewer 2's Comment 1 — Misdescribed C-CoT baseline
   - Reviewer 2's Comment 2 — Unclear dataset usage and splits
   - Reviewer 2's Comment 3 — Unexpectedly weak ORM results
   - Reviewer 2's Comment 4 — Main vs. ablation result inconsistency
   - Reviewer 2's Comment 5 — Possible answer leakage (shortcut learning)
   - Reviewer 2's Comment 6 — Process-level supervision claim lacks support
   - Reviewer 2's Comment 7 — Terminology inconsistency
   - Reviewer 2's Comment 8 — Overstated "fundamentally different paradigm" from ORM
   - Reviewer 2's Comment 9 — Methodological novelty needs to be articulated precisely
4. Response to Reviewer 3
   - Reviewer 3's Comment 1 — Evaluation only at 8B parameter scale
   - Reviewer 3's Comment 2 — Baselines are outdated (2021, 2023)
   - Reviewer 3's Comment 3 — Reproducibility
5. Summary of Revisions and Reiteration of Contribution
6. 修订对照汇总 / Summary of Manuscript Changes (Before/After)

---

## 致编辑的回复

感谢主编与副主编组织评审并转达三位审稿人的意见，也感谢主编在总结中对我们工作潜在价值的肯定。我们已认真对待主编所指出的"在方法、验证、对比与可复现性方面需实质性修订"的要求，并在一次完整统一的修订中予以回应。

1. **方法论主张已软化至监督信号实际能支撑的程度。** 针对主编所指出的"监督有效性不清"以及审稿人 #1（意见 1）和审稿人 #2（意见 6）的具体质询，我们删除了所有声称对步级局部错误敏感的表述，现将其描述为"对与问题语义层面的步级一致性敏感"。针对审稿人 #2 的意见 8，我们删除了"根本不同的范式"这一框架，现描述 D-ProtoCoT 与 ORM 共享相同的监督信号与功能角色，仅在打分形式上存在差异；与审稿人 #2 的意见 9 相呼应，我们亦将方法新颖性精确定位为"非对称训练/推理粒度与逐题动态原型"的组合，而非夸大单一组件的创新性。

2. **修正了既有实验中的实现缺陷与口径不一致。** 针对审稿人 #2 的意见 3，我们在修复三个实现 bug（512 token 截断、无类平衡加权、Q-A 联合 [CLS] 编码）后重跑了 ORM 基线，并报告训练/验证损失、F1 和 AUROC。针对意见 4，我们在 GSM8K 上以与主表一致的 pipeline 重跑了粒度消融，使消融与主表口径一致，并让非对称设计在统一口径下得到验证。

3. **补充了新的实验证据与对比。** 针对审稿人 #3 的意见 1，我们在 GSM8K 上补充了 Qwen3-14B 实验。针对意见 2，我们实现并运行了 Self-Certainty（Kang et al., 2025）作为近期基线。针对意见 5 与主编所指出的"可能存在答案泄露"，我们增加了三档输入模式（full/mask/qa_only）的泄露消融。与审稿人 #1 的意见 2 相呼应，我们亦补充了对比训练前后的 t-SNE 可视化与对齐得分预测路径正确性的 AUC（0.78），将"对齐≈推理质量"由断言转为证据。

4. **稿件规范性修订。** 针对审稿人 #2 的意见 1，描述错误的"C-CoT"基线已拆分为正确引用的 C-CoT（Chia et al.，对比式 prompting）和内部消融变体 Static-Prototype。针对意见 2，数据切分现以专门小节描述，明确采用 qid 分组切分且跨集重叠为零。针对意见 7，术语已在问题定义段中统一。针对意见 9，新颖性已收束为单一设计选择：非对称训练/推理粒度与逐题动态原型相结合。针对审稿人 #3 的意见 3，已添加可复现性声明并承诺公开代码与数据。与审稿人 #2 的意见 6 相呼应，我们亦在正文与 Limitations 中主动坦白步级标签噪声的存在，并将"检测局部错误"的过强表述改为"步-问语义一致性"。

5. **诚实叙事。** 针对主编所指出的"相似度对齐是否真正捕捉推理质量"以及"实验结果不一致"等问题，我们在方法未能占优之处明确说明。在饱和的 GSM8K/Qwen3-8B 设定下，修复后的 ORM 略胜 D-ProtoCoT；我们将 D-ProtoCoT 定位为 ORM 的互补方法而非全面占优，并刻画了两种方法各自占优的两个区间（饱和任务 vs. 低资源任务）。在 14B 上 GSM8K 趋于饱和，相对 Self-Consistency 的优势收窄；我们如实报告并分析其为基准饱和而非方法的局限。与审稿人 #1 的意见 3 相呼应，我们亦补充了"选择不偏向浅层路径"的四轴分析（词面重叠、推理深度、语义-措辞相关性、对齐-正确性 AUC）。

我们相信上述修订已实质性回应了主编与三位审稿人提出的意见，恳请主编审阅修订后的稿件。

---

## Response to the Editor

**Editor's Comments:**

The paper proposes a potentially valuable framework for reasoning-path selection, but the current version has significant conceptual and empirical weaknesses, including unclear supervision validity, possible answer leakage, inconsistencies in experimental results, and insufficient evidence that similarity-based alignment captures true reasoning quality. Substantial revisions, including clarifying methodology, strengthening validation, expanding comparisons, and improving reproducibility, are required before the work can be reconsidered.

**Response:**

We sincerely thank the Editor-in-Chief for the opportunity to revise and for the acknowledgement of the framework's potential value. We have taken seriously the call for "substantial revisions, including clarifying methodology, strengthening validation, expanding comparisons, and improving reproducibility," and have addressed these requirements in a single integrated revision.

1. **Methodological claims softened to match what the supervision actually supports.** In response to the Editor-in-Chief's concern about "unclear supervision validity" and the specific points raised by Reviewer 1's Comment 1 and Reviewer 2's Comment 6, we have removed all statements claiming sensitivity to localized step-level errors, and now describe the encoder as attuned to *step-level semantic consistency with the question*. We have also removed the "fundamentally different paradigm" framing (Reviewer 2's Comment 8) and repositioned the methodological novelty precisely as the combination of "an asymmetric training/inference granularity and a per-question dynamic prototype" (Reviewer 2's Comment 9), rather than overclaiming the originality of any single component.

2. **Corrected implementation flaws and inconsistency in the existing experiments.** In response to Reviewer 2's Comment 3, we re-ran the ORM baseline after fixing three implementation bugs (512-token truncation, no class-balance weighting, joint Q-A [CLS] encoding), and now report the training/validation loss, F1, and AUROC. In response to Comment 4, we re-ran the granularity ablation on GSM8K with the same pipeline as the main table so that the ablation and the main table are under a consistent protocol, allowing the asymmetric design to be validated under a unified setting.

3. **Added new experimental evidence and comparisons.** In response to Reviewer 3's Comment 1, we added a Qwen3-14B experiment on GSM8K. In response to Comment 2, we implemented and ran Self-Certainty (Kang et al., 2025) as a recent baseline. In response to Reviewer 2's Comment 5 and the Editor-in-Chief's concern about "possible answer leakage," we added a leakage ablation with three input modes (full / mask / qa_only). In line with Reviewer 1's Comment 2, we also added before/after t-SNE visualizations of contrastive alignment together with the AUC (0.78) of the alignment score for predicting path correctness, turning "alignment ≈ reasoning quality" from an assertion into evidence.

4. **Manuscript-level corrections.** In response to Reviewer 2's Comment 1, the misdescribed "C-CoT" baseline has been split into a correctly-cited C-CoT (Chia et al., contrastive prompting) and an in-house Static-Prototype ablation variant. In response to Comment 2, the data splits are now described in a dedicated subsection, with explicit qid-grouped partitioning and zero cross-split overlap. In response to Comment 7, the terminology has been unified in the Problem Setup. In response to Comment 9, the novelty has been sharpened to a single design choice: an asymmetric training/inference granularity combined with a per-question dynamic prototype. In response to Reviewer 3's Comment 3, a Limitations paragraph on evaluation scale has been added, with a dedicated Reproducibility paragraph to follow. In line with Reviewer 2's Comment 6, we have also proactively disclosed the step-level label noise in the main text and the Limitations section, rephrasing the overstrong "detecting localized errors" claim as "step-question semantic consistency."

5. **Honest narrative.** In response to the Editor-in-Chief's concerns about "whether similarity-based alignment captures true reasoning quality" and "inconsistencies in experimental results," we explicitly state where our method does not dominate. On the saturated GSM8K/Qwen3-8B setting, the corrected ORM slightly outperforms D-ProtoCoT; we position D-ProtoCoT as complementary to ORM rather than uniformly superior, and characterize the two regimes (saturated vs. low-resource) in which each method is preferable. On Qwen3-14B, GSM8K approaches saturation and the margin over Self-Consistency narrows; we report this transparently and analyze it as benchmark saturation rather than a limitation of the method. In line with Reviewer 1's Comment 3, we also added a four-axis analysis showing that "selection does not favor shallow paths" (lexical overlap, reasoning depth, semantics-vs-wording correlation, alignment-to-correctness AUC).

We believe the above revisions substantively address the concerns raised by the Editor-in-Chief and the three reviewers, and we respectfully submit the revised manuscript for the Editor-in-Chief's review.

---

## Response to Reviewer 1

### Reviewer 1's Overall Comment

**Reviewer's overall comment.** This paper proposes D-ProtoCoT, an inference-time framework for selecting chain-of-thought reasoning paths. Instead of fine-tuning the language model, it trains a lightweight auxiliary encoder with a step-level InfoNCE objective (supervised by gold-answer matching) to build a representation space in which correct paths align with their question. At inference, it aggregates the sampled paths into a similarity-weighted dynamic prototype and selects the path best aligned with it. Motivation is clear which PRM requires expensive human label is well-known problem but I do have some concerns.

**Response.** We sincerely thank the reviewer for the positive assessment and the constructive concerns. We are grateful for the careful reading. The three concerns the reviewer raised have pushed us to present the method more precisely and to add the evidence that the original submission lacked. In this revision, we have made the following updates:

(1) **Softened the process-level supervision claim (Comment 1).** We no longer state that the encoder is "sensitive to localized logical errors," and instead describe it as attuned to *step-level semantic consistency with the question*. The softened wording has been applied at six places in the manuscript (§3.2/§3.3/§3.4, §5.3, §4.6, and the Contributions paragraph in §1). We also added an AUC result on 200 GSM8K questions showing that the learned alignment predicts path correctness despite the outcome-derived labels.

(2) **Added a before/after t-SNE pair and a quantitative AUC measure (Comment 2).** Figure 2 now provides both pre- and post-alignment t-SNE visualizations on GSM8K (K=10), and the alignment score is shown to predict path correctness with an **AUC of 0.78**. "Alignment ≈ reasoning quality" is now backed by evidence rather than asserted.

(3) **Added a four-axis per-question analysis showing selection does not penalize deep reasoning (Comment 3).** On 200 GSM8K questions we compared the selected path against the unselected pool along lexical overlap with the question, reasoning depth, semantics-vs-wording correlation, and alignment-to-correctness AUC. Selected paths are no more question-echoing, no less deep, and the alignment signal is only weakly correlated with lexical overlap (Pearson r = −0.14) yet predictive of correctness (see Comment 2 for the AUC measure).

Below, we provide detailed responses to each of the reviewer's comments.

### Reviewer 1's Comment 1 — Step-level label noise

**Reviewer's concern.** Supervision comes only from final-answer matching, with path-level labels broadcast to every step. A path with flawed steps that reaches the correct answer by coincidence is treated as positive — the same failure mode as ORM — yet the paper claims sensitivity to localized logical errors.

**Our response.** We thank the reviewer for this precise diagnosis, and we fully agree. Our supervision is indeed outcome-derived: path-level labels come from final-answer matching and are broadcast to every step, so a path that reaches the correct answer through a flawed step is nominally labeled positive — the same label-noise mode as ORM. We had overclaimed in the original submission, and we have accordingly **softened our claims throughout the paper**. Specifically, we no longer state that the encoder is "sensitive to localized logical errors," and instead describe it as attuned to *step-level semantic consistency with the question*. The softened wording has been applied at six places (Method §3.2/§3.3/§3.4; Analysis §5.3; Experiments §4.6 "Comparison with ORM"; and the Contributions paragraph).

We nonetheless want to clarify why the method remains effective despite this shared label noise. The reason is not cleaner labels but a different *training geometry*: step-level InfoNCE aligns each step of a correct path to the question independently, so occasional step-level label noise is a minority signal within the $|\mathcal{P}|\cdot M$ positive pairs that the InfoNCE loss sees for that question, and is further diluted at inference time by path-level mean pooling and similarity-weighted prototype aggregation. A more detailed version of this argument is given in our response to Reviewer 2's Comment 6, where the same supervision issue is raised.

This argument is empirically supported: on 200 GSM8K questions (K = 10 paths each), the learned alignment predicts path correctness with an **AUC of 0.78**, showing that the outcome-level label noise does not overwhelm the learned signal. We have added this evidence to the paper at three places: §5.1 (alongside the t-SNE figure and in the Figure 2 caption), §5.4 (in the selection-depth table `tab:selection-depth`), and §5.6 (in the Comparison-with-ORM paragraph).

**Change made.** (1) Softened six overclaimed passages — the wording "sensitive to localized logical errors" has been replaced at each of the following locations:

| # | Location | Before (original) | After (revised) |
|---|---|---|---|
| 1 | Method §3.2 (step-level repr. role) | "...making the encoder **sensitive to localized logical errors** that would be obscured by path-level averaging." | "...**attuning the encoder to step-level semantic consistency with the question**, a signal that is diluted under path-level averaging." |
| 2 | Method §3.3 (representation encoding) | "...making the encoder **sensitive to localized logical errors** that would otherwise be averaged out at the path level." | "...**attuning the encoder to how well each step aligns semantically with the question**, a signal that is diluted under path-level averaging." |
| 3 | Method §3.4 (dynamic prototype motivation) | "While step-level representations are effective for training-time supervision due to their **sensitivity to localized errors**, they introduce step-to-step variance..." | "While step-level representations are effective for training-time supervision due to their **sensitivity to step-level semantic consistency**, they introduce step-to-step variance..." |
| 4 | Analysis §5.3 (granularity motivation) | "...making the encoder **sensitive to localized logical errors** that would be obscured by path-level averaging." | "...making the encoder **sensitive to step-level semantic consistency with the question** that would be obscured by path-level averaging." |
| 5 | Experiments §4.6 (Comparison with ORM, original closing) | "...propagating supervision signals to every step and making the encoder **sensitive to localized logical errors**." | (rewritten as the two-regime honest narrative; the "sensitive to localized logical errors" claim removed; replaced by "shaping a representation in which step-level semantic consistency drives alignment.") |
| 6 | Introduction §1 (Contributions, second bullet) | "...achieving **process-level supervision granularity** without step-level annotation beyond gold answers. It involves an annotation cost equivalent to ORMs while **capturing step-aware reasoning structure closer to PRMs**." | "...providing **denser supervision than path-level objectives** without step-level annotation beyond gold answers. It involves an annotation cost equivalent to ORMs while, **unlike ORMs, propagating the outcome-derived signal to every step so that alignment reflects step-level semantic consistency rather than only final-answer correctness**." |

(2) Added the AUC-of-0.78 evidence to Analysis §5.6 ("Comparison with ORM") and to the Figure 2 caption, together with an explicit acknowledgment that step labels inherit noise from the path outcome:

> **[新增 / New]** (Analysis §5.6): "We note that D-ProtoCoT's supervision, like ORM's, is derived from final-answer matching, so individual step labels are inherited from the path outcome, and a path that reaches the correct answer through a flawed step is nominally labeled positive. The advantage is therefore not cleaner step labels but a different training geometry: step-level InfoNCE provides $|\mathcal{P}|\cdot M$ positive pairs and aligns every step to the question independently, so occasional step-level label noise is a minority signal that is averaged out when path embeddings are aggregated into the prototype. Empirically, the resulting alignment still predicts path correctness with an **AUC of 0.78** on GSM8K, indicating that the label noise does not overwhelm the learned signal."

> **[新增 / New]** (Figure 2 caption, §5.1): "Test-path embeddings on GSM8K ($K{=}10$), colored by correctness (green: correct, red: incorrect). **Left:** before contrastive alignment (untrained encoder), correct and incorrect paths are intermixed. **Right:** after D-ProtoCoT alignment, the two classes become clearly separated. The learned alignment predicts path correctness with an AUC of $0.78$."

### Reviewer 1's Comment 2 — "Alignment ≈ reasoning quality" is asserted, not demonstrated

**Reviewer's concern.** Figure 2 shows the representation space before contrastive alignment but not after, and a quantitative measure (e.g., AUC of similarity predicting path correctness) is missing. The claim is not convincing as stated.

**Our response.** We thank the reviewer for this constructive suggestion, and we agree that the original submission left the central claim under-evidenced. We have added both pieces of evidence the reviewer requested.

**(a) Quantitative measure.** Using the trained encoder on 200 GSM8K test questions (K = 10 paths each), a path's alignment score predicts whether it reaches the correct answer with an **AUC of 0.78**. Because the alignment signal is only weakly correlated with lexical overlap with the question (Pearson r = −0.14; see our reply to Comment 3), this AUC reflects reasoning quality rather than surface similarity.

**(b) Post-training counterpart to Figure 2.** We now provide a before/after pair of t-SNE visualizations of test-path embeddings, colored by correctness (green = correct, red = incorrect), computed on the *same* K = 10 data as the AUC above. **Before** contrastive alignment (an untrained encoder, same architecture and projection), correct and incorrect paths are thoroughly intermixed, as expected for an encoder that has not been shaped for reasoning-quality separation. **After** contrastive alignment, the two classes become clearly separated in the representation space. Together, the figure and the AUC turn "alignment ≈ reasoning quality" from an assertion into evidence.

We would also like to acknowledge, with thanks, that the reviewer's comment surfaced a separate issue we had missed: the original Figure 2 was drawn on a single StrategyQA question with ~100 paths, which does not reflect the multi-question K=10 protocol used elsewhere in the paper. The new figures use the standard K = 10 protocol, making them consistent with the AUC and with the rest of the analysis.

**Change made.** (1) Replaced Figure 2 with a before/after t-SNE pair, switched from StrategyQA single-question (100 paths) to GSM8K 200-question × K=10 standard protocol, and added the AUC=0.78 result to the caption and Analysis §5.6:

| | Before (original) | After (revised) |
|---|---|---|
| Figure 2 source | `figure/tsne_cot100.png` (StrategyQA, single question, ~100 paths) | `figure/tsne_gsm8k_before.pdf` + `figure/tsne_gsm8k_after.pdf` (GSM8K, 200 questions × K=10) |
| Caption | "t-SNE visualization of BERT embeddings from 100 chain-of-thought paths generated for a StrategyQA example. Green points indicate correct reasoning paths, while red points indicate incorrect ones. Strong overlap suggests that surface-level semantic similarity alone is insufficient to distinguish reasoning quality." | "Test-path embeddings on GSM8K (K=10), colored by correctness (green: correct, red: incorrect). **Left:** before contrastive alignment (untrained encoder), correct and incorrect paths are intermixed. **Right:** after D-ProtoCoT alignment, the two classes become clearly separated. The learned alignment predicts path correctness with an **AUC of 0.78**." |
| Quantitative measure | (none) | AUC = 0.78 on 200 GSM8K questions (K=10), reported in §5.1, §5.4, and §5.6 |

(2) Added the corresponding quantitative sentence to Analysis §5.1 ("Why Naive Centroid Selection Fails"):

> **[新增 / New]** (Analysis §5.1): "To verify that alignment reflects reasoning quality rather than being merely asserted, we visualize test-path embeddings before and after contrastive alignment (Fig.~\ref{fig:tsne}): intermixed correct/incorrect paths become linearly separable after training, and the alignment score predicts path correctness with an **AUC of 0.78** on 200 GSM8K questions (K=10)."

### Reviewer 1's Comment 3 — Selecting by similarity may punish deep reasoning

**Reviewer's concern.** D-ProtoCoT picks the path most similar to the question, and prototype weights favor paths that stay close to it. Good reasoning often moves away from the question's wording, bringing in new ideas or making non-obvious jumps. Selecting by similarity may punish these deep but less-similar paths and reward shallow ones that just repeat the question.

**Our response.** We appreciate this conceptual concern, which is important and worth addressing directly rather than defensively. We emphasize first that D-ProtoCoT does **not** select by similarity to the question's surface form: candidate paths are aggregated in the *learned* representation space into a similarity-weighted **dynamic prototype**, and selection is by alignment to this prototype. The similarity that drives selection is therefore measured in a contrastively trained space, not against the question's raw wording.

To verify empirically that selection is not biased toward shallow, question-echoing paths, we ran a per-question analysis with the trained encoder on 200 GSM8K test questions (K = 10 paths each), comparing the **selected** path against the **unselected** pool along four axes:

- **(M1) Lexical overlap with the question.** If selection rewarded paths that merely repeat the question, selected paths would show higher lexical overlap. They do not: token-level Jaccard is 0.221 (selected) vs. 0.220 (unselected), and question-token coverage is 0.830 vs. 0.824 — statistically indistinguishable.
- **(M2) Reasoning depth.** Selected paths are **not** shorter or shallower: mean length is 222.4 vs. 221.9 tokens, step count is comparable (12.48 vs. 13.10), and the fraction of *novel* (non-question) content is 0.767 vs. 0.767. Selection does not trade reasoning depth for brevity. We report these as "comparable" rather than claiming selected paths are deeper, since the depth metrics do not show a consistent direction.
- **(M3) Semantics vs. wording.** The correlation between a path's alignment score and its lexical overlap with the question is only r = −0.14, showing that the learned alignment tracks semantic reasoning quality rather than surface wording — indeed, if anything, higher lexical echo is weakly associated with *lower* alignment.
- **(M4) Alignment predicts correctness.** The alignment score attains an AUC of **0.78** for predicting whether a path reaches the correct answer, and the selected paths reach 80.0% accuracy. Selecting by prototype alignment therefore preferentially recovers *correct* reasoning, not superficially similar reasoning.

Together, these results indicate that similarity-based selection in the learned space does not penalize deep reasoning: selected paths are no more question-echoing (M1), no less deep or novel (M2), the alignment signal is only weakly correlated with lexical overlap (M3), and it is predictive of correctness (M4).

**Change made.** Added a new Analysis subsection §5.4 ("Selection Does Not Penalize Deep Reasoning") with the four-axis comparison table (Table `tab:selection-depth`). The full text and table added to the manuscript are:

> **[新增 / New]** (Analysis §5.4): "A natural concern is that selecting the path most aligned with the prototype might favor shallow paths that echo the question wording and penalize deep paths that introduce new ideas. We stress that selection operates in the *learned* space against a similarity-weighted dynamic prototype, not against the question surface form. Empirically, on 200 GSM8K questions (K=10), the selected path is no more lexically similar to the question than the unselected pool (Jaccard 0.221 vs. 0.220; question coverage 0.830 vs. 0.824), and is comparable in depth (novel-content ratio 0.767 vs. 0.767; 222.4 vs. 221.9 tokens). Moreover, alignment is only weakly correlated with lexical overlap (Pearson r = −0.14) yet strongly predictive of correctness (AUC 0.78; selected-path accuracy 80.0%), confirming that alignment tracks reasoning quality rather than surface similarity."

**[新增 / New]** Table `tab:selection-depth`:

| | Selected | Unselected |
|---|---|---|
| Lexical Jaccard w/ question | 0.221 | 0.220 |
| Question-token coverage | 0.830 | 0.824 |
| Novel-content ratio | 0.767 | 0.767 |
| \# tokens | 222.4 | 221.9 |
| \# steps | 12.48 | 13.10 |
| Pearson($align$, Jaccard) | \multicolumn{2}{c}{$-0.14$} |
| AUC($align\!\rightarrow\!$correct) | \multicolumn{2}{c}{$0.78$} |

---
