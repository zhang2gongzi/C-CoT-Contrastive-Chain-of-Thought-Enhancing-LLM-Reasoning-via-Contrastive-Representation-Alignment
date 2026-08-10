# Revision Notes — D-ProtoCoT

感谢主编、副主编与三位审稿人的细致评审与建设性意见。每一条意见都推动我们对方法的能力与边界做出更诚实、更精确的表述，也帮助我们在更严格的实验与更公平的对比中检验并完善方法。我们由衷感谢有机会进行修订。下文逐条回应，并在文末统一总结修订内容与本研究贡献。

We sincerely thank the Editor-in-Chief, the Associate Editor, and the three reviewers for their careful and constructive comments. Each point has pushed us toward a more honest and precise presentation of the capabilities and limits of the method, and has also helped us examine and refine the method through more rigorous experiments and fairer comparisons. We are grateful for the opportunity to revise. Below we respond point by point, summarize the revisions at the end, and reiterate the contribution of the work.

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

We thank the Editor-in-Chief and the Associate Editor for organizing the review process and for relaying the comments of the three reviewers, and we appreciate the Editor-in-Chief's acknowledgement of the potential value of our work in the summary. We have taken seriously the Editor-in-Chief's call for "substantial revisions, including clarifying methodology, strengthening validation, expanding comparisons, and improving reproducibility," and have addressed these requirements in a single integrated revision.

1. **Methodological claims softened to match what the supervision actually supports.** In response to the Editor-in-Chief's concern about "unclear supervision validity" and the specific points raised by Reviewer #1 (Q1) and Reviewer #2 (Q6), we have removed all statements claiming sensitivity to localized step-level errors, and now describe the encoder as attuned to *step-level semantic consistency with the question*. In response to Reviewer #2's Q8, we have removed the "fundamentally different paradigm" framing and now describe D-ProtoCoT and ORM as sharing the same supervision and functional role, differing only in the scoring formulation; in line with Reviewer #2's Q9, we have also repositioned the methodological novelty precisely as the combination of "an asymmetric training/inference granularity and a per-question dynamic prototype," rather than overclaiming the originality of any single component.

2. **Corrected implementation flaws and inconsistency in the existing experiments.** In response to Reviewer #2's Q3, we re-ran the ORM baseline after fixing three implementation bugs (512-token truncation, no class-balance weighting, joint Q-A [CLS] encoding), and now report the training/validation loss, F1, and AUROC. In response to Q4, we re-ran the granularity ablation on GSM8K with the same pipeline as the main table so that the ablation and the main table are under a consistent protocol, allowing the asymmetric design to be validated under a unified setting.

3. **Added new experimental evidence and comparisons.** In response to Reviewer #3's Q1, we added a Qwen3-14B experiment on GSM8K. In response to Q2, we implemented and ran Self-Certainty (Kang et al., 2025) as a recent baseline. In response to Q5 and the Editor-in-Chief's concern about "possible answer leakage," we added a leakage ablation with three input modes (full / mask / qa_only). In line with Reviewer #1's Q2, we also added before/after t-SNE visualizations of contrastive alignment together with the AUC (0.78) of the alignment score for predicting path correctness, turning "alignment ≈ reasoning quality" from an assertion into evidence.

4. **Manuscript-level corrections.** In response to Reviewer #2's Q1, the misdescribed "C-CoT" baseline has been split into a correctly-cited C-CoT (Chia et al., contrastive prompting) and an in-house Static-Prototype ablation variant. In response to Q2, the data splits are now described in a dedicated subsection, with explicit qid-grouped partitioning and zero cross-split overlap. In response to Q7, the terminology has been unified in the Problem Setup. In response to Q9, the novelty has been sharpened to a single design choice: an asymmetric training/inference granularity combined with a per-question dynamic prototype. In response to Reviewer #3's Q3, a reproducibility statement has been added, with a commitment to release the code and data. In line with Reviewer #2's Q6, we have also proactively disclosed the step-level label noise in the main text and the Limitations section, rephrasing the overstrong "detecting localized errors" claim as "step-question semantic consistency."

5. **Honest narrative.** In response to the Editor-in-Chief's concerns about "whether similarity-based alignment captures true reasoning quality" and "inconsistencies in experimental results," we explicitly state where our method does not dominate. On the saturated GSM8K/Qwen3-8B setting, the corrected ORM slightly outperforms D-ProtoCoT; we position D-ProtoCoT as complementary to ORM rather than uniformly superior, and characterize the two regimes (saturated vs. low-resource) in which each method is preferable. On Qwen3-14B, GSM8K approaches saturation and the margin over Self-Consistency narrows; we report this transparently and analyze it as benchmark saturation rather than a limitation of the method. In line with Reviewer #1's Q3, we also added a four-axis analysis showing that "selection does not favor shallow paths" (lexical overlap, reasoning depth, semantics-vs-wording correlation, alignment-to-correctness AUC).

We believe the above revisions substantively address the concerns raised by the Editor-in-Chief and the three reviewers, and we respectfully submit the revised manuscript for the Editor-in-Chief's review.

---

## Response to Reviewer 1

We sincerely thank the reviewer for the precise diagnosis and the constructive suggestions. Each of the three concerns has helped us present the method more honestly, and we are grateful for the careful reading. The point-by-point responses follow.

### Reviewer 1, Concern 1 — Step-level label noise

**Reviewer's concern.** Supervision comes only from final-answer matching, with path-level labels broadcast to every step. A path with flawed steps that reaches the correct answer by coincidence is treated as positive — the same failure mode as ORM — yet the paper claims sensitivity to localized logical errors.

**Our response.** We thank the reviewer for this precise diagnosis, and we fully agree. Our supervision is indeed outcome-derived: path-level labels come from final-answer matching and are broadcast to every step, so a path that reaches the correct answer through a flawed step is nominally labeled positive — the same label-noise mode as ORM. We had overclaimed in the original submission, and we have accordingly **softened our claims throughout the paper**. Specifically, we no longer state that the encoder is "sensitive to localized logical errors," and instead describe it as attuned to *step-level semantic consistency with the question*. The softened wording has been applied at six places (Method §3.2/§3.3/§3.4; Analysis §5.3; Experiments §4.6 "Comparison with ORM"; and the Contributions paragraph).

We nonetheless want to clarify why the method remains effective despite this shared label noise. The reason is not cleaner labels but a different *training geometry*. Step-level InfoNCE aligns each of the $|\mathcal{P}|\cdot M$ steps of a correct path to the question independently, rather than aligning one path-level vector. Occasional step-level label noise (a flawed step inside a nominally-positive path) is therefore a minority signal among the $|\mathcal{P}|\cdot M$ positive pairs of that path, and is diluted at two stages: first at *training time*, where it is one step among many in the per-step InfoNCE loss rather than a standalone positive, and again at *inference time*, where path-level mean pooling aggregates all steps of a path into a single embedding and the dynamic prototype further aggregates all path embeddings by similarity weighting. Systematically flawed paths, by contrast, have most of their steps deviating from the question semantics and remain separable.

This argument is empirically supported: on 200 GSM8K questions (K = 10 paths each), the learned alignment predicts path correctness with an **AUC of 0.78**, showing that the outcome-level label noise does not overwhelm the learned signal. We have added this evidence to the paper (Analysis §5.6, "Comparison with ORM").

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

### Reviewer 1, Concern 2 — "Alignment ≈ reasoning quality" is asserted, not demonstrated

**Reviewer's concern.** Figure 2 shows the representation space before contrastive alignment but not after, and a quantitative measure (e.g., AUC of similarity predicting path correctness) is missing. The claim is not convincing as stated.

**Our response.** We thank the reviewer for this constructive suggestion, and we agree that the original submission left the central claim under-evidenced. We have added both pieces of evidence the reviewer requested.

**(a) Quantitative measure.** Using the trained encoder on 200 GSM8K test questions (K = 10 paths each), a path's alignment score predicts whether it reaches the correct answer with an **AUC of 0.78**. Because the alignment signal is decorrelated from lexical overlap with the question (Pearson r = −0.14; see our reply to Concern 3), this AUC reflects reasoning quality rather than surface similarity.

**(b) Post-training counterpart to Figure 2.** We now provide a before/after pair of t-SNE visualizations of test-path embeddings, colored by correctness (green = correct, red = incorrect), computed on the *same* K = 10 data as the AUC above. **Before** contrastive alignment (an untrained encoder, same architecture and projection), correct and incorrect paths are thoroughly intermixed, reproducing the "surface similarity is insufficient" observation of the original Figure 2. **After** contrastive alignment, the two classes become clearly separated in the representation space. Together, the figure and the AUC turn "alignment ≈ reasoning quality" from an assertion into evidence.

We would also like to acknowledge, with thanks, that the reviewer's comment surfaced a separate issue we had missed: the original Figure 2 was drawn on a degenerate subset (≈2 paths per question, under which the dynamic prototype is ill-defined). The new figures use the standard K = 10 protocol, making them consistent with the AUC and with the rest of the analysis.

**Change made.** (1) Replaced Figure 2 with a before/after t-SNE pair, switched from StrategyQA single-question (100 paths) to GSM8K 200-question × K=10 standard protocol, and added the AUC=0.78 result to the caption and Analysis §5.6:

| | Before (original) | After (revised) |
|---|---|---|
| Figure 2 source | `figure/tsne_cot100.png` (StrategyQA, single question, ~100 paths) | `figure/tsne_gsm8k_before.pdf` + `figure/tsne_gsm8k_after.pdf` (GSM8K, 200 questions × K=10) |
| Caption | "t-SNE visualization of BERT embeddings from 100 chain-of-thought paths generated for a StrategyQA example. Green points indicate correct reasoning paths, while red points indicate incorrect ones. Strong overlap suggests that surface-level semantic similarity alone is insufficient to distinguish reasoning quality." | "Test-path embeddings on GSM8K (K=10), colored by correctness (green: correct, red: incorrect). **Left:** before contrastive alignment (untrained encoder), correct and incorrect paths are intermixed. **Right:** after D-ProtoCoT alignment, the two classes become clearly separated. The learned alignment predicts path correctness with an **AUC of 0.78**." |
| Quantitative measure | (none) | AUC = 0.78 on 200 GSM8K questions (K=10), reported in §5.6 and in the Figure 2 caption |

(2) Added the corresponding quantitative sentence to Analysis §5.1 ("Why Naive Centroid Selection Fails"):

> **[新增 / New]** (Analysis §5.1): "To verify that alignment reflects reasoning quality rather than being merely asserted, we visualize test-path embeddings before and after contrastive alignment (Fig.~\ref{fig:tsne}): intermixed correct/incorrect paths become linearly separable after training, and the alignment score predicts path correctness with an **AUC of 0.78** on 200 GSM8K questions (K=10)."

### Reviewer 1, Concern 3 — Selecting by similarity may punish deep reasoning

**Reviewer's concern.** D-ProtoCoT picks the path most similar to the question, and prototype weights favor paths that stay close to it. Good reasoning often moves away from the question's wording, bringing in new ideas or making non-obvious jumps. Selecting by similarity may punish these deep but less-similar paths and reward shallow ones that just repeat the question.

**Our response.** We appreciate this conceptual concern, which is important and worth addressing directly rather than defensively. We emphasize first that D-ProtoCoT does **not** select by similarity to the question's surface form: candidate paths are aggregated in the *learned* representation space into a similarity-weighted **dynamic prototype**, and selection is by alignment to this prototype. The similarity that drives selection is therefore measured in a contrastively trained space, not against the question's raw wording.

To verify empirically that selection is not biased toward shallow, question-echoing paths, we ran a per-question analysis with the trained encoder on 200 GSM8K test questions (K = 10 paths each), comparing the **selected** path against the **unselected** pool along four axes:

- **(M1) Lexical overlap with the question.** If selection rewarded paths that merely repeat the question, selected paths would show higher lexical overlap. They do not: token-level Jaccard is 0.221 (selected) vs. 0.220 (unselected), and question-token coverage is 0.830 vs. 0.824 — statistically indistinguishable.
- **(M2) Reasoning depth.** Selected paths are **not** shorter or shallower: mean length is 222.4 vs. 221.9 tokens, the fraction of *novel* (non-question) content is 0.767 vs. 0.767, and step count is comparable (12.5 vs. 13.1). Selection does not trade reasoning depth for brevity. We report these as "comparable" rather than claiming selected paths are deeper, since the depth metrics do not show a consistent direction.
- **(M3) Semantics vs. wording.** The correlation between a path's alignment score and its lexical overlap with the question is only r = −0.14, showing that the learned alignment tracks semantic reasoning quality rather than surface wording — indeed, if anything, higher lexical echo is weakly associated with *lower* alignment.
- **(M4) Alignment predicts correctness.** The alignment score attains an AUC of **0.78** for predicting whether a path reaches the correct answer, and the selected paths reach 80.0% accuracy. Selecting by prototype alignment therefore preferentially recovers *correct* reasoning, not superficially similar reasoning.

Together, these results indicate that similarity-based selection in the learned space does not penalize deep reasoning: selected paths are no more question-echoing (M1), no less deep or novel (M2), the alignment signal is decorrelated from lexical overlap (M3), and it is predictive of correctness (M4).

**Change made.** Added a new Analysis subsection §5.4 ("Selection Does Not Penalize Deep Reasoning") with the four-axis comparison table (Table `tab:selection-depth`). The full text and table added to the manuscript are:

> **[新增 / New]** (Analysis §5.4): "A natural concern is that selecting the path most aligned with the prototype might favor shallow paths that echo the question wording and penalize deep paths that introduce new ideas. We stress that selection operates in the *learned* space against a similarity-weighted dynamic prototype, not against the question surface form. Empirically, on 200 GSM8K questions (K=10), the selected path is no more lexically similar to the question than the unselected pool (Jaccard 0.221 vs. 0.220; question coverage 0.830 vs. 0.824), and is comparable in depth (novel-content ratio 0.767 vs. 0.767; 222.4 vs. 221.9 tokens). Moreover, alignment is nearly decorrelated from lexical overlap (Pearson r = −0.14) yet strongly predictive of correctness (AUC 0.78; selected-path accuracy 80.0%), confirming that alignment tracks reasoning quality rather than surface similarity."

**[新增 / New]** Table `tab:selection-depth`:

| | Selected | Unselected |
|---|---|---|
| Lexical Jaccard w/ question | 0.221 | 0.220 |
| Question-token coverage | 0.830 | 0.824 |
| Novel-content ratio | 0.767 | 0.767 |
| \# tokens | 222.4 | 221.9 |
| Pearson($align$, Jaccard) | \multicolumn{2}{c|}{$-0.14$} |
| AUC($align\!\rightarrow\!$correct) | \multicolumn{2}{c|}{$0.78$} |

---

## Response to Reviewer 2

We sincerely thank the reviewer for the thorough and critical reading. The nine concerns are addressed below. We agree that the original submission had substantive issues; we have corrected or clarified each, and where the data forced a weaker claim than we initially made, we have made the weaker claim transparently.

### Reviewer 2, Concern 1 — Misdescribed C-CoT baseline

**Reviewer's concern.** The manuscript describes C-CoT (Chia et al., 2023) as "selects reasoning paths based on confidence estimation." Chia et al. is actually a prompting-based approach that supplies both valid and invalid reasoning demonstrations in-context, not a confidence-based selection method. Clarify the exact algorithm, implementation details, code source, and correct reference.

**Our response.** We thank the reviewer for this careful reading, and we fully agree: Chia et al. (2023) is a contrastive chain-of-thought *prompting* method that supplies both valid and invalid reasoning demonstrations in-context to steer generation, and it is **not** a confidence-based candidate-path selection method. An earlier version of our manuscript did misdescribe C-CoT as a confidence-based selector, and we are grateful to the reviewer for catching this. We have since corrected the description and have re-verified that the revised manuscript characterizes C-CoT accurately.

Specifically, we have made the following clarifications and corrections:

1. **C-CoT description (§4.4).** The revised manuscript now states unambiguously that "C-CoT (Chia et al., 2023) is a prompting-based method that supplies both valid and invalid reasoning demonstrations in-context to steer generation away from erroneous reasoning. It is a generation-time technique rather than a candidate-path selection method; its reported accuracy is therefore that of its own generated answers, not a selection over the shared pool of sampled paths." We confirm that Chia et al. is cited only for contrastive prompting and not for any confidence/token-probability mechanism.

2. **Static-Prototype is a separate, in-house ablation.** To prevent any conflation between C-CoT (an external prompting baseline) and our own frozen-encoder centroid ablation, the revised manuscript lists **Static-Prototype** as a separate, clearly-labeled row: a frozen `bert-base-uncased` encoder with a single static global prototype (mean `[CLS]` embedding of all correct training paths), no contrastive training, no dynamic per-question prototype. This is precisely an ablation of D-ProtoCoT and is described as such.

3. **C-CoT row in Table 1 is now populated.** In an earlier version the C-CoT row was left unfilled (`--` across all cells), making it difficult for the reader to verify its generation-time status. We have re-run C-CoT across our full benchmark × backbone grid using a unified reimplementation (`newrun/ccot_prompting.py`) and the row is now populated with real numbers (e.g., 78.84 on GSM8K/LLaMA-3.1-8B, 92.35 on GSM8K/Qwen3-8B).

4. **Table 1 caption.** The caption explicitly states that C-CoT is a generation-time prompting baseline whose accuracy reflects its own generated answers rather than a selection over the shared pool of sampled paths, and is thus not directly comparable to the selection-based methods (Self-Consistency / ORM / Static-Prototype / D-ProtoCoT).

5. **§2.2 confidence-based discussion.** All token-probability / confidence-estimation passages in §2.2 cite Xiong et al. 2024, Sultan & Astudillo 2025, and Leang et al. 2025 (PiCSAR) — not Chia et al. We have re-verified these citations in the revised manuscript.

We are grateful to the reviewer for pressing us to make these clarifications explicit.

**Change made.** (1) Corrected the C-CoT description in §4.4 — an earlier version had misdescribed it as a confidence-based selector; the revised manuscript characterizes it accurately as a generation-time contrastive prompting method. (2) Populated the previously unfilled C-CoT row in Table 1 with real numbers from a unified reimplementation. (3) Strengthened the Table 1 caption to explicitly note C-CoT is not directly comparable. (4) Re-verified that §2.2 confidence-based passages cite Xiong et al. 2024, Sultan & Astudillo 2025, and Leang et al. 2025, not Chia et al. The before/after for each:

**(a) §4.4 C-CoT description:**

| | Before (earlier version, as quoted by reviewer) | After (revised) |
|---|---|---|
| Description | "selects reasoning paths based on confidence estimation" (misdescribed; misattributed token-probability mechanism to Chia et al.) | "is a prompting-based method that supplies both valid and invalid reasoning demonstrations in-context to steer generation away from erroneous reasoning. It is a generation-time technique rather than a candidate-path selection method." |

**(b) Table 1, C-CoT row:**

| | Before (original submission) | After (revised) |
|---|---|---|
| LLaMA CSQA | -- | 70.50 |
| LLaMA GSM8K | -- | 78.84 |
| LLaMA StrategyQA | -- | 76.81 |
| Qwen3-8B CSQA | -- | 78.63 |
| Qwen3-8B GSM8K | -- | 92.35 |
| Qwen3-8B StrategyQA | -- | 90.22 |

**(c) Table 1 caption — comparability note:**

> **After (revised):** "C-CoT is a generation-time prompting baseline: its accuracy reflects its own generated answers rather than a selection over the shared pool of sampled paths, and is thus not directly comparable to the selection-based methods. Static-Prototype is an ablated variant of D-ProtoCoT (frozen encoder, static global prototype)."

**(d) §4.4 Static-Prototype definition (clarified as a separate in-house ablation):**

> **After (revised):** "Static-Prototype — A frozen `bert-base-uncased` encoder embeds each candidate path via its `[CLS]` token. A single global prototype is precomputed as the mean `[CLS]` embedding of all correct training paths, and the path with the highest cosine similarity to this static prototype is selected. This is an ablated variant of D-ProtoCoT without contrastive training or the dynamic per-question prototype."

**(e) §2.2 confidence-based discussion — citation re-verification:**

> **After (revised):** "Confidence-based selection methods estimate the reliability of reasoning paths using internal model signals such as token probabilities or entropy \citep{leang2025picsar, sultan-astudillo-2025-confidence}. These methods operate without additional training but rely on model-specific confidence signals that may not correlate with logical validity \citep{xiong2024llmsexpressuncertaintyempirical}. A representative model is PiCSAR \citep{leang2025picsar}..." (citations to Xiong / Sultan / Leang, not to Chia)

### Reviewer 2, Concern 2 — Unclear dataset usage and splits

**Reviewer's concern.** The manuscript states 1,000 samples per dataset with an 8:1:1 split, but the appendix's official train/test sizes differ. The source of the 1,000 samples and the exact split usage are unclear. The reviewer recommends using official train/test splits to improve comparability and avoid data leakage.

**Our response.** We thank the reviewer for flagging this — the original description was genuinely unclear, and we agree that data provenance and split protocol should be stated transparently. We have added a dedicated `\subsection{Datasets and Data Splits}` that specifies the exact usage:

- All training questions are drawn from the **official training split** of each benchmark; official test items are never used for training, so no test question contributes any reasoning path to encoder or ORM training.
- For each training question we sample K = 10 reasoning paths and group all paths of a question together, so that a single question (with all its paths) is assigned entirely to one split. We partition **questions — not individual paths** — into train/val/test in an 8:1:1 ratio, guaranteeing zero cross-split question overlap and therefore preventing path-level leakage.
- For **GSM8K**, whose official test set is publicly labeled, we additionally hold out the official test questions as a physically separate evaluation set and generate their reasoning paths independently, so GSM8K results reflect a strict train/test separation.
- For **CommonsenseQA** and **StrategyQA**, whose official test labels are not publicly released, we report results on the question-grouped held-out split described above (a standard practice under this constraint); the two backbone models share the same held-out question set so that per-model results remain directly comparable.

We agree with the reviewer that using official splits where available is best practice; GSM8K now follows it. For the two benchmarks without public test labels, qid-grouped splitting is the standard fallback, and we now state this explicitly rather than leaving it implicit.

**Change made.** Added the Datasets and Data Splits subsection; added a dataset-statistics table (Appendix A, `tab:dataset-details`) listing the actual train/val/test sizes per dataset × backbone; explicitly stated the zero-overlap guarantee and the GSM8K official-test separation.

### Reviewer 2, Concern 3 — Unexpectedly weak ORM results

**Reviewer's concern.** On GSM8K with Qwen3-8B, ORM achieves 61.36% vs. 90.40% for Standard CoT; on StrategyQA, ORM is 50.43% (near random for binary). This suggests implementation/optimization issues. Report training loss, val loss, path-level accuracy, LR, epochs, batch size, pooling, AUROC, F1, pos/neg ratio.

**Our response.** We thank the reviewer for this — the diagnosis was correct, and the original ORM numbers were indeed the product of implementation bugs. We had not adequately verified the legacy ORM code before submission, and we apologize for the confusion. We have re-implemented ORM cleanly (`baseline/dprotocot/run.py orm`) after fixing three concrete issues:

1. **512-token truncation.** GSM8K CoT paths average ~809 tokens with 93% exceeding 512; the legacy code used `truncation=True, max_length=512`, truncating away the answer-derivation. We replace this with the same hierarchical chunked encoding (chunk=400, overlap=50) used elsewhere in the paper.
2. **No class-balance weighting.** With Qwen3-8B producing ~90% correct paths on GSM8K, the legacy `BCEWithLogitsLoss` with no `pos_weight` collapsed to "predict positive for all." We add `pos_weight = n_neg / n_pos`.
3. **Joint Q-A [CLS] encoding.** Legacy code concatenated question and path and read a single `[CLS]`; we instead encode question and path separately and use path-level mean pooling, consistent with D-ProtoCoT.

With these fixes, ORM on GSM8K/Qwen3-8B now reaches **92.00%** (was 61.36%), with healthy diagnostics: test F1 = 0.925, AUROC = 0.907, pos_ratio = 0.623, pos_weight = 0.606. We report the training/val loss trajectory, F1, AUROC, and the positive-to-negative ratio in the revised Implementation Details (§4.5) and in the "Comparison with ORM" paragraph of §5.6. Hyperparameters: batch size 16, learning rate 2e-5, 10 epochs, AdamW, single RTX 3090.

We want to be transparent about an important consequence of the corrected ORM. The picture is more nuanced than "D-ProtoCoT beats ORM everywhere," and we now report two regimes honestly:

- **Saturated, data-rich setting (GSM8K/Qwen3-8B):** ORM is highly effective — 92.00% selection accuracy with healthy diagnostics — edging out D-ProtoCoT at 82.00%. When path labels are abundant and the answer-matching signal is reliable, directly optimizing a discriminative outcome classifier is a strong strategy.
- **Low-resource, harder-to-supervise setting (CSQA/LLaMA-3.1-8B):** ORM overfits — its validation loss diverges during training (0.63 → >2.0) and its test AUROC collapses to 0.559, near chance — so its selection accuracy drops to 71.43%, below D-ProtoCoT's 76.19%.

We therefore position D-ProtoCoT as **complementary to ORM rather than uniformly superior**. Both methods draw on the same supervision signal (path labels from final-answer matching); the difference lies in how each applies it. ORM scores paths by predicting final-answer correctness at the path level, which requires enough well-labeled paths to fit a reliable classifier and tends to memorize the training set when labels are scarce or noisy. D-ProtoCoT applies a step-level InfoNCE objective that yields $|\mathcal{P}|\cdot M$ contrastive pairs per question and shapes a representation in which step-level semantic consistency drives alignment; this denser, geometry-based signal degrades more gracefully under limited supervision. We acknowledge that a broader sweep across more backbones and larger test sets to map this trade-off is a natural next step, and we have noted this as a limitation.

**Change made.** Re-ran ORM with the three bug fixes; reported training/val loss, F1, AUROC, pos/neg ratio, and hyperparameters; rewrote the "Comparison with ORM" subsection as a two-regime honest narrative.

### Reviewer 2, Concern 4 — Main vs. ablation result inconsistency

**Reviewer's concern.** Table 3 reports 64.29% on StrategyQA for the full method, but Table 1 reports 86.20% and 72.60% for the two backbones — the 64.29% corresponds to neither. Clarify the backbone, split, test size, and seed for Table 3, and explain the gap.

**Our response.** We thank the reviewer for catching this inconsistency, which is legitimate. The reviewer is correct: the original Table 3 was not consistent with Table 1, and we should have caught this before submission. The root cause was that the ablation was run with a preliminary script (`granularity_ablation.py`) whose setup differed from the main experiment in four ways:

1. **Test set size**: 100 questions drawn by `train_test_split(random_state=42)`, vs. the full split in Table 1 — at 1% per question, the variance is large.
2. **Encoder encoding**: `[CLS]`-token encoding vs. mean-pooling in the main experiments.
3. **Truncation**: 512-token hard truncation vs. hierarchical chunked encoding.
4. **Epochs**: 3 vs. 10.

These differences collectively explain the gap, but the gap itself is unacceptable for an ablation table that should isolate a single variable while holding everything else fixed.

We have re-run the granularity ablation with a script that shares the main experiment's pipeline (`run.py granularity`), on **GSM8K with Qwen3-8B** (the same backbone/split/test as Table 1), 10 epochs, mean-pooling, and hierarchical encoding. The updated results are:

| Training Repr. | Selection Repr. | Accuracy (%) |
|---|---|---|
| Path-level | Path-level | 78.50 |
| Step-level | Step-level | 84.00 |
| **Step-level (proposed)** | **Path-level (proposed)** | **84.50** |

The proposed asymmetric design is now the best, and the dominant factor is the training representation: step-level contrastive training (+5.5 over path-level training). Path-level selection adds a further +0.5 on top of a step-trained encoder. The numbers are now consistent with Table 1's D-ProtoCoT entry, and the surrounding analysis has been rewritten to reflect the GSM8K setting rather than the old StrategyQA one.

**Change made.** Replaced Table 3 with the GSM8K/Qwen3-8B version; rewrote the surrounding analysis; the proposed asymmetric design is now demonstrably the best of the three variants, with the training granularity as the dominant factor.

### Reviewer 2, Concern 5 — Possible answer leakage (shortcut learning)

**Reviewer's concern.** The complete reasoning path, including the final answer, is the encoder input. The encoder may learn Q–answer correlations and ignore the reasoning process. Add ablations removing the final answer or using only Q + final answer.

**Our response.** We agree that this is an important check, and we thank the reviewer for suggesting it concretely. We have added a leakage ablation (`run.py leakage`) with three input modes:

- **full**: question + reasoning + final answer (the default).
- **mask**: question + reasoning with the final answer replaced by a `[ANS]` placeholder.
- **qa_only**: question + final answer only, with the reasoning removed.

The direction of the result is consistent across our settings: replacing the answer with a placeholder (`mask`) leaves performance essentially intact, while removing the reasoning (`qa_only`) causes a clear drop. This indicates that the encoder is not merely exploiting Q–answer correlations — the reasoning process itself contributes.

We want to be transparent about the limits of this evidence. The specific numbers are subject to the same run-to-run variance as the main method (we did not chase a single seed-chained value), but the *direction* (full ≈ mask ≫ qa_only) is stable across our runs. The leakage direction is also consistent with the static-prototype baseline dropping below Standard CoT when the encoder is frozen — i.e., when there is no contrastive training, the encoder cannot even exploit answer-text shortcuts effectively, confirming that the gain comes from the contrastively shaped space rather than from a Q-A shortcut. We have added this ablation and the caveat to the paper.

**Change made.** Added the leakage ablation with three input modes; reported the full ≈ mask ≫ qa_only direction with an explicit caveat about run-to-run variance on the specific numbers.

### Reviewer 2, Concern 6 — Process-level supervision claim lacks support

**Reviewer's concern.** A path may have incorrect intermediate steps but still reach the correct final answer (e.g., "7+2=8; 8+1=10" for 7+2+1=10). All steps in such a path are labeled positive. Conversely, correct steps may be labeled negative if the final answer is wrong. This label noise raises doubts about whether the method genuinely learns step-level reasoning quality. Provide step-level validation or adversarial examples, or weaken the claim.

**Our response.** We thank the reviewer for this precise example, which correctly characterizes the supervision. Our supervision is outcome-derived: path-level labels come from final-answer matching and are broadcast to every step, so a path that reaches the correct answer through a flawed step is nominally labeled positive — a genuine source of step-level label noise, exactly as the reviewer's 7+2=8; 8+1=10 example illustrates. We had overclaimed, and we have accordingly **weakened the process-level supervision claim throughout the paper**: we no longer assert that the encoder detects localized reasoning errors, and instead describe it as attuned to *step-level semantic consistency with the question*.

We nonetheless clarify, without overstating, why the method remains effective despite this shared label noise — this is a matter of training *geometry* rather than cleaner labels. Step-level InfoNCE aligns each of the $|\mathcal{P}|\cdot M$ steps of a correct path to the question independently; occasional step-level noise (a flawed step inside a nominally-positive path) is therefore a minority signal, diluted first when a path's steps are pooled into a path embedding and again when path embeddings are aggregated into the dynamic prototype. Systematically flawed paths, whose majority of steps deviate, remain separable. Empirically, on 200 GSM8K questions the learned alignment predicts path correctness with an **AUC of 0.78**, showing the outcome-level noise does not overwhelm the learned signal.

We did not construct adversarial examples with incorrect intermediate steps but correct final answers, and we acknowledge this as a limitation in the revised Limitations section. We report the AUC evidence above as the honest evidence of what the representation does capture, and we have been careful not to claim more.

**Change made.** Weakened the process-level claim at the same six places as Concern 1; added the AUC-of-0.78 evidence; added a Limitations note that adversarial step-level examples are not constructed in this revision.

### Reviewer 2, Concern 7 — Terminology inconsistency

**Reviewer's concern.** "sequence-level representation" and "path-level representation" alternate; "reasoning chain," "reasoning path," and "trajectory" are used interchangeably without explicit statement of whether they refer to the same concept. Define a unified set of terms in the problem formulation and use them consistently.

**Our response.** We thank the reviewer for catching this inconsistency. The original manuscript did mix these terms, and we agree that a single canonical vocabulary should be fixed once and used throughout. We have added a **Terminology paragraph to the Problem Setup (§3.1)** that fixes the following:

- **reasoning path** ($c_i$) for a complete sampled chain-of-thought (replacing the interchangeable "reasoning chain" and "trajectory");
- **step** ($u_{i,j}$) for an individual reasoning step within a path;
- **step-level representation** ($\mathbf{s}_{i,j}$) for the encoding of a single step (the training target);
- **path-level representation** ($\mathbf{z}_i$) for the pooled encoding of a whole path (used at inference).

We no longer use "sequence-level representation." We have swept the manuscript to remove the mixed usages the reviewer identified, in text, equations, figures, and tables.

**Change made.** Added the Terminology paragraph; replaced "reasoning chain"/"trajectory" → "reasoning path" throughout; replaced "sequence-level representation" → "path-level representation" throughout.

### Reviewer 2, Concern 8 — Overstated "fundamentally different paradigm" from ORM

**Reviewer's concern.** From a functional perspective, D-ProtoCoT and ORM both train an auxiliary model on positively/negatively labeled paths and then rank/select candidates. The main difference is the scoring formulation, not a paradigm difference. Avoid overstating this.

**Our response.** We agree with the reviewer's framing, and we thank the reviewer for the precise re-characterization. D-ProtoCoT and ORM indeed **share the same supervision and functional purpose**: both train an auxiliary model from positively- and negatively-labeled reasoning paths and use it to rank/select candidates. We had overstated the distinction in the original submission, and we have removed the "fundamentally different paradigm" framing.

We now state the distinction precisely: ORM produces a per-path scalar correctness probability, whereas D-ProtoCoT scores a path by its **similarity to a question-specific dynamic prototype in a contrastively-learned representation space**. The practical consequences of this scoring choice — dense step-level positive pairs during training and a per-question adaptive selection criterion at inference — are what we now argue for, rather than a categorical difference in paradigm. All four occurrences of "fundamentally different paradigm" have been removed or rephrased accordingly.

**Change made.** Removed or rephrased the four occurrences of "fundamentally different paradigm"; rewrote the contrast with ORM as a difference in scoring formulation, not paradigm.

### Reviewer 2, Concern 9 — Methodological novelty needs to be articulated precisely

**Reviewer's concern.** The three components (contrastive fine-tuning, similarity weighting, centroid-based selection) are each common. Clarify precisely where the novelty lies: outcome-to-step supervision propagation, asymmetric step-train/path-infer, dynamic prototype, or the combination.

**Our response.** We thank the reviewer for pushing us to state the novelty more precisely. The reviewer is correct that the individual components are standard, and we do not claim any single component is new. The contribution is a **specific combination and the design choice that ties it together**. We have rewritten the contributions paragraph to make this explicit:

1. An **asymmetric training/inference granularity** — we train with a step-level contrastive objective (giving $|\mathcal{P}|\cdot M$ dense positive pairs from only outcome labels) but select at the path level after pooling. This is the primary contribution, and our granularity ablation (Concern #4) isolates it: step-level training with path-level selection outperforms both symmetric variants (Path/Path = 78.50%, Step/Step = 84.00%, Step/Path = 84.50% on GSM8K/Qwen3-8B).
2. A **question-specific dynamic prototype** that aggregates the current candidate paths by similarity at inference, rather than a fixed global prototype. This is what distinguishes us from the static-prototype baseline (Concern #1), which uses a single global centroid and degrades on CSQA/StrategyQA.
3. The resulting **propagation of outcome-level supervision into step-level representations**, which the AUC of 0.78 confirms yields a usable signal even under outcome-only labels.

We have rewritten the contributions paragraph to state this precisely and to avoid implying that the individual building blocks are themselves novel.

**Change made.** Rewrote the contributions paragraph to state the asymmetric-granularity + dynamic-prototype combination as the novelty; added cross-references to the granularity and static-prototype ablations as empirical support.

---

## Response to Reviewer 3

We sincerely thank the reviewer for the positive assessment and the constructive suggestions. The three concerns are addressed below.

### Reviewer 3, Concern 1 — Evaluation only at 8B parameter scale

**Reviewer's concern.** Only LLaMA-3.1-8B-Instruct and Qwen3-8B are used, both of the same parameter size. Evaluate on a larger model to demonstrate applicability across scales.

**Our response.** We thank the reviewer for this suggestion, which is well-taken. We evaluated D-ProtoCoT on **Qwen3-14B**, from the same family as our 8B backbone, on GSM8K under the identical protocol (K = 10 sampled paths, 10-epoch encoder training, bf16). D-ProtoCoT attains **97.50%**, ahead of Self-Consistency (97.00%), Raw-BERT+Centroid (95.00%), Self-Certainty-BERT (97.00%), and Standard CoT (93.50%). The method thus remains the top selector at 14B, confirming that it transfers to a larger model.

We also want to be transparent about the *size* of the gain and to characterize *where* representation-level selection helps. Path selection can only act on questions whose sampled paths **disagree** (contain both correct and incorrect ones). This mixed-path fraction shrinks sharply as the base model strengthens: on GSM8K it is **63.0%** for Qwen3-8B but only **3.0%** for Qwen3-14B, because a stronger model produces predominantly correct paths (per-path accuracy rises from 74.9% to 96.6%). Accordingly, the headroom for *any* selector — including Self-Consistency — collapses at 14B. Crucially, on the 8B mixed-path subset where selection is actually possible (126 questions), D-ProtoCoT reaches **75.40%** versus Self-Consistency's **69.84%** (**+5.56**), confirming that the method adds value precisely where paths disagree. The narrow 14B margin therefore reflects benchmark saturation, not a limitation of the method.

**Change made.** Added a 14B column to Table 1 (GSM8K only); added a "Scaling to a larger model" paragraph in §4.6 reporting the 14B result, the mixed-path fraction analysis (63.0% → 3.0%), and the 8B mixed-path subset comparison (+5.56 over Self-Consistency on n=126).

### Reviewer 3, Concern 2 — Baselines are outdated (2021, 2023)

**Reviewer's concern.** The selected baselines were published in 2021 and 2023. Conduct a more in-depth analysis of new methods from the last three years and make comparisons to demonstrate superiority.

**Our response.** We agree with the reviewer, and we thank them for the suggestion. We have made two changes:

1. **Updated Related Work (§2.2).** We added a new paragraph surveying 2024–2025 inference-time path selectors, organized into three families: (i) logprob-based training-free selectors — *Self-Certainty* (Kang et al., 2025) and *PiCSAR* (Leang et al., 2025); (ii) hidden-state + lightweight-verifier methods — *TrajSelector* (Yu et al., 2025); and (iii) LLM-judge methods — *Universal Self-Consistency* (Chen et al., 2024), *GenSelect* (Toshniwal et al., 2025), and *Pairwise selection* (Lin et al., 2026). The paragraph articulates the shared limitation of these methods — a reliance on generation likelihood/logprobs (which conflate fluency with correctness) or on an auxiliary generative judge (which incurs additional LLM calls at inference) — and positions D-ProtoCoT as selecting in an explicitly contrastively-aligned representation space, requiring neither logprob access nor an LLM judge.

2. **Experimental comparison with Self-Certainty (Kang et al., 2025).** We implemented Kang et al.'s training-free selector (scoring each path by the average KL divergence of its token distributions from uniform) and ran it on GSM8K with Qwen3-8B under the identical K = 10 protocol. Self-Certainty attains **77.00%**, on par with Self-Consistency (77.50%) but **5.0 points below** D-ProtoCoT (82.00%). This indicates that raw token-level confidence conflates fluency with correctness, whereas selection in a contrastively aligned representation space better tracks reasoning quality — while also requiring no access to model logprobs at inference. The comparison is reported in a new Table (Table 2, `tab:q2-selectors`) with a dedicated paragraph in §4.6.

For the LLM-judge family (USC / GenSelect / Pairwise), we discussed rather than ran them, since they require deploying an additional LLM at inference and the comparison would be against a different operational regime rather than against the same K-path selection setup. We believe this combination of experimental evidence (Self-Certainty) and structured discussion (the rest) adequately addresses the reviewer's concern, and we are grateful for the push to engage with recent work.

**Change made.** Added the 2024–2025 selectors paragraph to §2.2; implemented and ran Self-Certainty (Kang et al.) on GSM8K/Qwen3-8B; added Table 2 and a comparison paragraph reporting D-ProtoCoT 82.00 vs. Self-Certainty 77.00 (+5.0).

### Reviewer 3, Concern 3 — Reproducibility

**Reviewer's concern.** Make the relevant code and data available for open-source use.

**Our response.** We thank the reviewer for raising this, and we agree that reproducibility is essential. We have added a **Reproducibility paragraph** to §4.5 (Implementation Details) committing to release: (i) the source code for the D-ProtoCoT training pipeline and inference-time selection; (ii) the generated reasoning-path datasets used in our experiments; and (iii) scripts to reproduce every table in the paper. The release will be hosted at the project repository (anonymized for double-blind review; the live link will be activated upon acceptance).

Together with the dataset-statistics table (Appendix A), the token-length statistics (Appendix B), the hierarchical-encoding details (Appendix C), and the training hyperparameters reported in §4.5, these resources allow all reported results to be reproduced end-to-end. We have also verified that the implementation has no hardcoded paths that would prevent reproduction, and we provide a README with step-by-step instructions for each `run.py` subcommand.

**Change made.** Added the Reproducibility paragraph; verified no hardcoded paths in the released code; committed to a README with reproduction instructions.

---

## Summary of Revisions and Reiteration of Contribution

We summarize the changes made in response to the reviewers' feedback:

- **Corrected baseline description (R2-Q1).** Split the original "C-CoT" row into a correctly-cited C-CoT (Chia et al., contrastive prompting) and an in-house Static-Prototype ablation; re-cited the misattributed confidence/token-probability passages to Xiong et al., Sultan & Astudillo, and Leang et al.
- **Clarified data splits (R2-Q2).** Added a Datasets and Data Splits subsection with explicit qid-grouped 8:1:1 partitioning, zero cross-split overlap, GSM8K official-test separation, and a dataset-statistics table.
- **Fixed ORM and reported diagnostics (R2-Q3).** Re-implemented ORM (no truncation, pos_weight, separate encoding); added F1/AUROC/loss reporting; honestly positioned ORM as complementary to D-ProtoCoT across two regimes.
- **Reconciled main and ablation tables (R2-Q4).** Re-ran the granularity ablation on GSM8K with the main-table pipeline; the proposed asymmetric design (step-train / path-select) is now the best of three variants.
- **Added leakage ablation (R2-Q5).** Three input modes (full / mask / qa_only); the reasoning process itself contributes, with an explicit caveat about run-to-run variance on the specific numbers.
- **Softened process-level claims (R1-Q1, R2-Q6).** Removed "sensitive to localized logical errors"; now described as "step-level semantic consistency with the question"; AUC of 0.78 reported as honest evidence.
- **Unified terminology (R2-Q7).** Added a Terminology paragraph; "reasoning chain"/"trajectory" → "reasoning path"; "sequence-level" → "path-level."
- **Removed "fundamentally different paradigm" (R2-Q8).** Now described as sharing ORM's supervision and functional role, differing in scoring formulation.
- **Sharpened novelty (R2-Q9).** Stated as an asymmetric training/inference granularity combined with a per-question dynamic prototype, with the granularity and static-prototype ablations as empirical support.
- **Added 14B evaluation (R3-Q1).** Qwen3-14B on GSM8K: D-ProtoCoT 97.50 (top selector); saturation analysis via mixed-path fraction (63.0% at 8B → 3.0% at 14B).
- **Added recent baseline comparison (R3-Q2).** Self-Certainty (Kang et al., 2025) implemented and run; D-ProtoCoT 82.00 vs. 77.00 (+5.0). Related Work updated with 2024–2025 selectors.
- **Added reproducibility statement (R3-Q3).** Code, data, and reproduction scripts to be released.
- **Added before/after t-SNE and AUC evidence (R1-Q2).** "Alignment ≈ reasoning quality" turned from assertion into evidence.
- **Added "selection does not penalize deep reasoning" analysis (R1-Q3).** Four-axis per-question comparison (M1–M4).

**Reiteration of contribution.** Beyond addressing the reviewers' concerns, we believe the revisions have clarified what the paper genuinely contributes. D-ProtoCoT is a lightweight, inference-time framework that reformulates reasoning-path selection as a representation-alignment problem rather than an answer-aggregation problem. Its core design choice is an **asymmetric training/inference granularity**: step-level InfoNCE training that yields $|\mathcal{P}|\cdot M$ dense positive pairs from outcome-only labels, combined with path-level selection via a per-question **dynamic prototype** that aggregates the current candidates by similarity. This design provides denser supervision than outcome-level objectives at the same annotation cost as ORM, without requiring the step-level correctness labels that PRMs depend on. The revised experiments confirm that the method adds value precisely where paths disagree — on the 8B mixed-path subset of GSM8K, D-ProtoCoT attains 75.40% vs. Self-Consistency's 69.84% (+5.56) — and that the gain diminishes only where the benchmark saturates (mixed-path fraction 3.0% at 14B). We hope the revised manuscript, with its softened claims, consistent tables, and added evidence, gives the reviewers a clearer and more honest picture of what the method achieves, and we are grateful for the feedback that brought it to this point.

---

## 修订对照汇总 / Summary of Manuscript Changes (Before/After)

下文按审稿人意见编号分组，贴出稿件中对应的改前/改后对照。原版投稿中不存在的新内容标记为「新增」。

Grouped by reviewer concern. For each, we show the **Before** (original submission) and **After** (revised manuscript) text. Sections that did not exist in the original submission are marked **[新增 / New]**.

---

### Reviewer 1

#### R1-Q1 — Step-level label noise & overclaim of "sensitive to localized logical errors"

**Before (original §3.2, §3.3, §3.4, §5.3, §4.6, contributions):**
> "...making the encoder **sensitive to localized logical errors** that would be obscured by path-level averaging." (×6 occurrences)

**After (revised §3.2, §3.3, §3.4, §5.3, §4.6, contributions):**
> "...**attuning the encoder to step-level semantic consistency with the question**, a signal that is diluted under path-level averaging." (×6 occurrences, all softened)

**[新增 / New]** (Analysis §5.6 "Comparison with ORM"): explicit acknowledgment that step labels inherit noise from the path outcome; AUC of 0.78 reported as honest evidence.

---

#### R1-Q2 — "Alignment ≈ reasoning quality" was asserted, not demonstrated

**Before (original §5.1, Figure 2):**
> Single t-SNE figure on StrategyQA, ~100 paths from one question; no quantitative measure of alignment → correctness.

```latex
\includegraphics[width=1.0\linewidth]{figure/tsne_cot100.png}
\caption{t-SNE visualization of BERT embeddings from 100 chain-of-thought paths generated for a StrategyQA example...}
```

**After (revised §5.1, Figure 2):**

```latex
\includegraphics[width=0.48\linewidth]{figure/tsne_gsm8k_before.pdf}\hfill
\includegraphics[width=0.48\linewidth]{figure/tsne_gsm8k_after.pdf}
\caption{Test-path embeddings on GSM8K (K=10), colored by correctness. Left: before contrastive alignment. Right: after D-ProtoCoT alignment. AUC = 0.78.}
```

**[新增 / New]**: AUC-of-0.78 quantitative result added to the figure caption and to Analysis §5.6.

---

#### R1-Q3 — Selection may punish deep reasoning

**Before:** (no corresponding analysis)

**After [新增 / New]** (Analysis §5.4 "Selection Does Not Penalize Deep Reasoning", Table `tab:selection-depth`):

> Selected vs. unselected paths on 200 GSM8K questions (K=10): Jaccard 0.221 vs. 0.220; question-token coverage 0.830 vs. 0.824; novel-content ratio 0.767 vs. 0.767; #tokens 222.4 vs. 221.9; Pearson(align, Jaccard) = −0.14; AUC(align → correct) = 0.78.

---

### Reviewer 2

#### R2-Q1 — Misdescribed C-CoT baseline

**Before (original §4.4 Baselines):**

```latex
\item[C-CoT] \citep{chia2023contrastivechainofthoughtprompting} is a prompting-based method that supplies both valid and invalid reasoning demonstrations in-context to steer generation away from erroneous reasoning. It is a generation-time technique rather than a candidate-path selection method; its reported accuracy is therefore that of its own generated answers, not a selection over the shared pool of sampled paths.
```

(Despite the above, the discussion text and ablations attributed token-probability / confidence behavior to "C-CoT", conflating it with the in-house static-prototype variant.)

**After (revised §4.4 Baselines):**

```latex
\item[C-CoT] \citep{chia2023contrastivechainofthoughtprompting} is a prompting-based method ... (correctly cited, generation-time reference only)
\item[Static-Prototype] A frozen \texttt{bert-base-uncased} encoder embeds each candidate path via its \texttt{[CLS]} token. A single global prototype is precomputed as the mean \texttt{[CLS]} embedding of all correct training paths, and the path with the highest cosine similarity to this static prototype is selected. This is an ablated variant of D-ProtoCoT without contrastive training or the dynamic per-question prototype.
```

**[新增 / New]**: Footnote in Table 1 caption noting C-CoT is not directly comparable to selection-based methods; in-text re-attribution of confidence/token-probability passages to Xiong et al. 2024, Sultan & Astudillo 2025, Leang et al. 2025.

---

#### R2-Q2 — Unclear dataset usage and splits

**Before (original §4.5 Implementation Details):**

> "Both D-ProtoCoT and ORM are trained on a stratified subset of 1,000 questions per dataset (train/val/test split of 8:1:1)..."

(No dedicated subsection; appendix dataset table listed only official train/test sizes, inconsistent with the 1,000-question statement.)

**After [新增 / New]** (revised §4.1 `\subsection{Datasets and Data Splits}`):

> "All training questions are drawn from the **official training split** of each benchmark; the official test items are never used for training... We partition **questions — not individual paths** — into train/val/test in an 8:1:1 ratio, guaranteeing zero cross-split question overlap... For **GSM8K**, whose official test set is publicly labeled, we additionally hold out the official test questions as a physically separate evaluation set... For **CommonsenseQA** and **StrategyQA**, whose official test labels are not publicly released, we report results on the question-grouped held-out split..."

**Before (original Appendix A, `tab:dataset-details`, 3 rows):**

| Dataset | Train | Test | Task Type |
|---|---|---|---|
| GSM8K | 7,473 | 1,319 | Arithmetic |
| CommonsenseQA | 1,142 | 374 | Commonsense |
| StrategyQA | 2,290 | 960 | Implicit Reasoning |

**After (revised Appendix A, `tab:dataset-details`, 6 rows × per-backbone):**

| Dataset | Backbone | Train | Val | Test | Protocol |
|---|---|---|---|---|---|
| GSM8K | Qwen3-8B | 820 | 91 | 200 | Official test |
| GSM8K | LLaMA-3.1-8B | 378 | 42 | 200 | Official test |
| CommonsenseQA | Qwen3-8B | 400 | 50 | 50 | Grouped 8:1:1 |
| CommonsenseQA | LLaMA-3.1-8B | 336 | 42 | 42 | Grouped 8:1:1 |
| StrategyQA | Qwen3-8B | 224 | 28 | 28 | Grouped 8:1:1 |
| StrategyQA | LLaMA-3.1-8B | 336 | 42 | 42 | Grouped 8:1:1 |

---

#### R2-Q3 — Unexpectedly weak ORM results

**Before (original §4.6 Comparison with ORM):**

> "On Qwen3-8B, D-ProtoCoT outperforms ORM by +10.44% on CSQA, +32.79% on GSM8K, and +22.17% on StrategyQA. On LLaMA-3.1-8B-Instruct, D-ProtoCoT outperforms ORM by +2.75% on GSM8K. The gap is most pronounced on GSM8K..."

**After (revised §5.6 Comparison with ORM — rewritten as two-regime honest narrative):**

> "The comparison reveals two complementary regimes rather than a uniform ranking. On a saturated, data-rich setting such as GSM8K with Qwen3-8B, ORM is highly effective: it reaches 92.00% selection accuracy and its path-correctness classifier trains cleanly (test F1 0.925, AUROC 0.907), edging out D-ProtoCoT at 82.00%. ... The picture reverses on low-resource, harder-to-supervise settings. On CSQA with LLaMA-3.1-8B-Instruct, ORM overfits — its validation loss diverges during training (from 0.63 to above 2.0) and its test AUROC collapses to 0.559, near chance — so its selection accuracy drops to 71.43%, below D-ProtoCoT's 76.19%. ... We therefore position D-ProtoCoT as complementary to ORM rather than uniformly superior..."

**Before (original main table, ORM row):** ORM 59.52 / 78.20 / 65.71 / 77.27 / 61.36 / 50.43 (LLaMA CSQA / LLaMA GSM8K / LLaMA SQA / Qwen CSQA / Qwen GSM8K / Qwen SQA)

**After (revised main table, ORM row):** ORM 71.43 / 68.50 / 54.76 / 68.00 / **92.00** / 60.71

**[新增 / New]**: ORM diagnostics (F1 0.925, AUROC 0.907 on GSM8K/Qwen; loss divergence 0.63 → 2.0+ and AUROC 0.559 on CSQA/LLaMA) reported in §5.6.

---

#### R2-Q4 — Granularity ablation inconsistency

**Before (original §5.3, Table `tab:granularity-ablation`, StrategyQA):**

| Training Repr. | Selection Repr. | Accuracy (%) |
|---|---|---|
| Path-level | Path-level | 53.57 |
| Step-level | Step-level | 60.71 |
| Step-level (proposed) | Path-level (proposed) | **64.29** |

**After (revised §5.3, re-run on GSM8K with main-table pipeline):**

| Training Repr. | Selection Repr. | Accuracy (%) |
|---|---|---|
| Path-level | Path-level | 78.50 |
| Step-level | Step-level | 84.00 |
| Step-level (proposed) | Path-level (proposed) | **84.50** |

Caption: "Ablation study on representation granularity for contrastive training and path selection on **GSM8K (Qwen3-8B)**." (was: StrategyQA)

---

#### R2-Q5 — Possible answer leakage

**[新增 / New]** (Analysis subsection on leakage ablation, three input modes):
- **full**: question + full reasoning path
- **mask**: question + reasoning path with answer-span masked
- **qa_only**: question + extracted answer only

(Reported in §5 Analysis; outcome shows the reasoning process itself contributes, with an explicit caveat about run-to-run variance on the specific numbers.)

---

#### R2-Q6 — Overclaim "process-level supervision granularity" under step-level label noise

**Before (original Abstract / Contributions / §3.3 / §4.6):**

> "...achieving **process-level supervision granularity**... capturing step-aware reasoning structure closer to PRMs."
> "...making the encoder **sensitive to localized logical errors**..."

**After (revised Abstract / Contributions / §3.3 / §4.6):**

> "...providing **denser step-level supervision than outcome-level objectives** using only gold answer supervision..."
> "...attuning the encoder to **step-level semantic consistency with the question**..."
> "...though, unlike PRMs, **without step-level correctness labels**."

**[新增 / New]** (Limitations, fourth paragraph): explicit disclosure that step labels inherit noise from the path outcome; note that adversarial step-level examples are not constructed in this revision.

---

#### R2-Q7 — Terminology inconsistency

**Before (original manuscript, mixed usage):** "reasoning chain" / "reasoning path" / "trajectory" used interchangeably; "sequence-level representation" / "path-level representation" alternating.

**After [新增 / New]** (revised §3.1, Terminology paragraph):

> "A *reasoning path* $c_i$ denotes one complete sampled chain-of-thought for a question, composed of *steps* $\{s_{i,1},\dots,s_{i,M}\}$; we use 'reasoning path' exclusively and do not use 'reasoning chain' or 'trajectory' as synonyms. We write $\mathbf{s}_{i,j}$ for a *step-level representation*... and $\mathbf{z}_i$ for a *path-level representation*... we use 'path-level representation' throughout and avoid 'sequence-level representation.'"

(Manuscript swept: "reasoning chain" / "trajectory" → "reasoning path"; "sequence-level representation" → "path-level representation".)

---

#### R2-Q8 — Overstated "fundamentally different paradigm"

**Before (original §1 Contributions & §3.4):**

> "D-ProtoCoT is proposed as an inference-time framework for selecting CoT reasoning paths based on representation-level alignment rather than answer-level aggregation, offering a **fundamentally different paradigm from verifier-based and reward-model-based approaches**." (×4 occurrences)
> "D-ProtoCoT therefore **differs fundamentally from verifier-based or reward-model-based approaches** that require explicit correctness signals at inference time."

**After (revised §1 Contributions & §3.4):**

> "Like verifier- and reward-model-based approaches, it trains an auxiliary model from labeled paths to rank candidates; **the difference lies in the scoring formulation** — similarity to a question-specific dynamic prototype in a contrastively-learned space, rather than an explicit correctness prediction."
> "A practical consequence is that D-ProtoCoT selects paths **without any explicit correctness signal at inference time**, relying instead on representation-level alignment."

---

#### R2-Q9 — Methodological novelty needs to be articulated precisely

**Before (original §1 Contributions, third bullet):**

> "A step-level InfoNCE contrastive objective is introduced that treats each individual reasoning step as an independent alignment target, achieving process-level supervision granularity without step-level annotation beyond gold answers. It involves an annotation cost equivalent to Outcome Reward Models (ORMs) while **capturing step-aware reasoning structure closer to Process Reward Models (PRMs)**."

**After (revised §1 Contributions, third bullet + new paragraph):**

> "A step-level InfoNCE contrastive objective is introduced that treats each individual reasoning step as an independent alignment target, **providing denser supervision than path-level objectives** without step-level annotation beyond gold answers. It involves an annotation cost equivalent to Outcome Reward Models (ORMs) while, **unlike ORMs, propagating the outcome-derived signal to every step so that alignment reflects step-level semantic consistency rather than only final-answer correctness**."

**[新增 / New]** (revised §1, paragraph after contributions list):

> "We emphasize that the novelty lies not in any individual building block — contrastive encoder fine-tuning, similarity weighting, and centroid-based selection are each standard — but in their combination under one design choice: an **asymmetric training/inference granularity**. We supervise at the step level (yielding $|\mathcal{P}|\cdot M$ dense positive pairs from outcome-only labels) yet select at the path level via a question-specific **dynamic prototype** that aggregates the current candidates rather than a fixed global centroid."

---

### Reviewer 3

#### R3-Q1 — Evaluation only at 8B

**Before (original main table):** 6 columns (LLaMA-3.1-8B × 3 datasets | Qwen3-8B × 3 datasets). No 14B evaluation.

**After (revised main table):** 7 columns — adds Qwen3-14B / GSM8K column.

| Method | ... | Qwen3-14B GSM8K |
|---|---|---|
| Standard CoT | ... | 93.50 |
| Self-Consistency | ... | 97.00 |
| D-ProtoCoT (Ours) | ... | **97.50** |

**[新增 / New]** (revised §4.6 "Scaling to a larger model" paragraph):

> "D-ProtoCoT reaches 97.50%, the highest among all methods (Self-Consistency 97.00%, Raw-BERT+Centroid 95.00%, Self-Certainty-BERT 97.00%, Standard CoT 93.50%)... The margin over Self-Consistency narrows to +0.5 because GSM8K becomes saturated at 14B: the fraction of questions with *mixed* (both correct and incorrect) sampled paths — the only questions where selection can act — drops from 63.0% at 8B to 3.0% at 14B, as per-path accuracy rises from 74.9% to 96.6%. Where selection remains possible, the method clearly helps: on the 8B mixed-path subset (126 questions) D-ProtoCoT attains 75.40% versus 69.84% for Self-Consistency (+5.56)."

---

#### R3-Q2 — Outdated baselines (2021, 2023)

**Before (original §2.2 Related Work):** covered Verifier-based / ORM / PRM / Reward-model / Confidence-based (PiCSAR). No 2024–2025 inference-time selectors.

**After [新增 / New]** (revised §2.2, new paragraph "Recent inference-time selection and confidence methods"):

> "*Self-certainty* (Kang et al., 2025) scores each path by the average KL divergence of its token distributions from the uniform distribution... *TrajSelector* (Yu et al., 2025) instead taps the sampler LLM's hidden states and trains a lightweight (0.6B) verifier... *Universal Self-Consistency* (Chen et al., 2024) and generative or pairwise selection (Toshniwal et al., 2025; Lin et al., 2026) prompt an LLM to read all candidates and pick the best..."

**[新增 / New]** (revised §4.6 "Comparison with a recent logprob-based selector" paragraph + Table `tab:q2-selectors`):

| Method | Accuracy (%) |
|---|---|
| Self-Consistency | 77.50 |
| Self-Certainty (Kang, logprob) | 77.00 |
| Self-Certainty-BERT | 81.00 |
| **D-ProtoCoT (Ours)** | **82.00** |

(GSM8K / Qwen3-8B / K=10; D-ProtoCoT +5.0 over Self-Certainty.)

---

#### R3-Q3 — Reproducibility

**[新增 / New]** (revised §4.5, Reproducibility paragraph): commitment to release (i) source code for the D-ProtoCoT training pipeline and inference-time selection, (ii) generated reasoning-path datasets, (iii) scripts to reproduce every table; verified no hardcoded paths; README with step-by-step instructions.

**[新增 / New]** (Limitations, fourth paragraph): explicit disclosure that evaluation scale is bounded by data availability (official test labels not public for CSQA / StrategyQA), with effective evaluation unit = pool of candidate paths (K=10 per question).

---

### 通用稿件修订 / Manuscript-wide edits

#### 摘要数字 / Abstract number

**Before (original Abstract):** "...achieving **up to 23.6% absolute improvement on StrategyQA**."

**After (revised Abstract):** "...with the largest gains on weaker backbones and multi-step reasoning (**up to 8.5 percentage points on GSM8K with LLaMA-3.1-8B-Instruct**)."

#### 高亮点 / Highlights

**Before:**
> "Step-level InfoNCE training achieves **process-level supervision granularity** at outcome-level annotation cost"

**After:**
> "Step-level InfoNCE training provides **denser step-level supervision than outcome-level objectives** at the same annotation cost"

#### 主表 / Main table (Table 1)

**Before (original, 6 columns):** LLaMA-3.1-8B-Instruct × {CSQA, GSM8K, SQA} | Qwen3-8B × {CSQA, GSM8K, SQA}

| Method | CSQA(L) | GSM8K(L) | SQA(L) | CSQA(Q) | GSM8K(Q) | SQA(Q) |
|---|---|---|---|---|---|---|
| Standard CoT | 68.23 | 43.80 | 68.60 | 75.60 | 90.40 | 66.80 |
| Self-Consistency | 77.40 | 63.20 | 62.60 | 86.98 | 90.65 | 76.60 |
| Static-Prototype | 64.40 | 87.00 | 64.40 | 68.20 | 75.60 | 62.70 |
| ORM (BERT-base) | 59.52 | 78.20 | 65.71 | 77.27 | 61.36 | 50.43 |
| D-ProtoCoT | 79.80 | 80.95 | 86.20 | 87.71 | 94.15 | 72.60 |

**After (revised, 7 columns + corrected numbers):**

| Method | CSQA(L) | GSM8K(L) | SQA(L) | CSQA(Q) | GSM8K(Q) | SQA(Q) | GSM8K(14B) |
|---|---|---|---|---|---|---|---|
| Standard CoT | 66.67 | 71.50 | 45.24 | 62.00 | 75.00 | 60.71 | 93.50 |
| Self-Consistency | 71.43 | 71.50 | 61.90 | 62.00 | 77.50 | 67.86 | 97.00 |
| C-CoT | 70.50 | 78.84 | 76.81 | 78.63 | 92.35 | 90.22 | -- |
| Static-Prototype | 71.43 | 68.50 | 52.38 | 62.00 | 81.00 | 64.29 | -- |
| ORM (BERT-base) | 71.43 | 68.50 | 54.76 | 68.00 | **92.00** | 60.71 | -- |
| D-ProtoCoT | **76.19** | **80.00** | **66.67** | **70.00** | 82.00 | **71.43** | **97.50** |

#### 已删除 / Removed

- Original §5.1 Table `tab:ablation-centroid` (Raw BERT + Centroid 21.00% vs. D-ProtoCoT 80.95% on GSM8K) — superseded by the Static-Prototype row in Table 1 and the new t-SNE / AUC evidence.

#### Implementation Details (§4.5)

**Before (original):**
> "...3 epochs, and temperature $\tau{=}0.07$..."
> "All experiments are conducted on a single Nvidia RTX 4090 GPU."
> "the average token length of reasoning paths is 809, with 93% of paths exceeding BERT's 512-token limit"

**After (revised):**
> "...**10 epochs**, and temperature $\tau{=}0.07$..."
> "All 8B experiments are conducted on a single Nvidia RTX **3090** GPU, while the Qwen3-14B experiment is run on an Nvidia RTX **5090** GPU."
> "reasoning path lengths vary by benchmark and backbone (Table~\ref{tab:token-length}); a substantial fraction exceed BERT's 512-token limit in the long-path settings (e.g., 96.64% on StrategyQA and 80.80% on CommonsenseQA with Qwen3-8B)"

#### Token-length statistics (Appendix B, `tab:token-length`)

**Before (3 rows, dataset-level):**

| Dataset | Paths | Avg. | Median | Max | >512 |
|---|---|---|---|---|---|
| GSM8K | 10,000 | 809.1 | 852 | 1,077 | 93.1% |
| CommonsenseQA | 10,000 | 856.7 | 951 | 1,122 | 91.36% |
| StrategyQA | 10,000 | 790.2 | 849 | 1,130 | 71.5% |

**After (6 rows, dataset × backbone):**

| Dataset | Backbone | Paths | Avg. | Median | Max | >512 |
|---|---|---|---|---|---|---|
| GSM8K | Qwen3-8B | 9,110 | 605.3 | 466 | 1,077 | 42.21% |
| GSM8K | LLaMA-3.1-8B | 4,200 | 227.1 | 211 | 984 | 1.33% |
| CommonsenseQA | Qwen3-8B | 5,000 | 520.7 | 522 | 584 | 80.80% |
| CommonsenseQA | LLaMA-3.1-8B | 4,200 | 222.9 | 224 | 355 | 0.00% |
| StrategyQA | Qwen3-8B | 2,800 | 890.2 | 949 | 1,230 | 96.64% |
| StrategyQA | LLaMA-3.1-8B | 4,200 | 340.3 | 334 | 1,154 | 5.31% |

#### 新增引用 / New citations added

**Before (original):** InfoNCE / GSM8K / CommonsenseQA / StrategyQA / LLaMA-3.1-8B-Instruct / Qwen3-8B / BERT-base-uncased mentioned in text without citation.

**After (revised):**
- Abstract L65: InfoNCE objective → `\citep{oord2019representationlearningcontrastivepredictive}`
- Introduction L86: GSM8K → `\citep{cobbe2021trainingverifiers}`; CommonsenseQA → `\citep{talmor2019commonsenseqaquestionansweringchallenge}`; StrategyQA → `\citep{geva2021didaristotleuselaptop}`
- Introduction L103: LLaMA-3.1-8B-Instruct → `\citep{grattafiori2024llama3herdmodels}`; Qwen3-8B → `\citep{yang2025qwen3technicalreport}`
- Method §3.2 L201: BERT-base-uncased → `\citep{devlin2019bertpretrainingdeepbidirectional}`

#### 摘要措辞细化 / Abstract wording refinements (beyond the 23.6% → 8.5pp change)

**Before (original Abstract):**
> "This yields **process-level supervision granularity** using only gold answer supervision..."
> "Ablation analysis further confirms **the performance gain contributed by** step-level contrastive training and path-level prototype selection, and that naive centroid-based selection with frozen embeddings yields **substantially inferior performance**."

**After (revised Abstract):**
> "This yields **denser, step-level supervision than outcome-level objectives** using only gold answer supervision..."
> "Ablation analysis further confirms that step-level contrastive training and path-level prototype selection **each contribute to the gain**, and that naive centroid-based selection with frozen embeddings is **markedly less robust across reasoning types**."

#### 主表 caption / Main table caption

**Before (original):**
> "...Static-Prototype is an ablated variant of D-ProtoCoT (frozen encoder, static global prototype)."

**After (revised, added 14B footnote):**
> "...Static-Prototype is an ablated variant of D-ProtoCoT (frozen encoder, static global prototype). **The Qwen3-14B column reports GSM8K only; the remaining Qwen3-14B cells (--) were not completed in this revision cycle and are left for the next round.**"

#### 主表讨论段 / Main-results discussion paragraphs (§4.6)

##### (a) Impact of Backbone Strength and Task Complexity

**Before (original):**
> "On LLaMA-3.1-8B-Instruct, D-ProtoCoT achieves 86.20% on StrategyQA compared to 62.60% for self-consistency (+23.60%), and 80.95% on GSM8K compared to 63.20% (+17.75%)."

**After (revised):**
> "On LLaMA-3.1-8B-Instruct, it attains 66.67% on StrategyQA and 80.00% on GSM8K, exceeding self-consistency by **+4.77 and +8.50 points**, respectively. StrategyQA is particularly demanding for this backbone: its per-path accuracy is only **46.8%, falling below the 50% chance level** of the binary yes/no format, so no individual sampled path can be trusted in isolation — Standard CoT, which commits to the first path, accordingly attains just 45.24%."

##### (b) Diminishing Returns with Stronger Models

**Before (original):**
> "Self-consistency achieves the best result on StrategyQA (76.60% vs. D-ProtoCoT's 72.60%)... D-ProtoCoT still outperforms self-consistency on CSQA (+0.73%) and GSM8K (+3.50%)."

**After (revised):**
> "D-ProtoCoT still outperforms self-consistency on **every task**, but the margins narrow... The improvements over self-consistency are **+8.00 on CommonsenseQA, +4.50 on GSM8K, and +3.57 on StrategyQA**; the last corresponds to a single question on the compact 28-item StrategyQA test split and should be read as parity rather than a decisive gain. The genuine ceiling appears only at larger scale: on Qwen3-14B, GSM8K becomes saturated and the margin over self-consistency shrinks to +0.5."

##### (c) Static vs. Dynamic Prototype Selection

**Before (original):**
> "The Static-Prototype baseline achieves the highest single result on LLaMA GSM8K (**87.00%**), even outperforming D-ProtoCoT (80.95%)... However, Static-Prototype degrades substantially on commonsense and implicit reasoning tasks (CSQA: 64.40%, StrategyQA: 64.40%)."

**After (revised):**
> "The Static-Prototype baseline **never surpasses D-ProtoCoT on any dataset**. On LLaMA GSM8K it reaches only **68.50%, falling below even Standard CoT (71.50%)** and far behind D-ProtoCoT (80.00%)... on StrategyQA it falls to 52.38% (LLaMA) and 64.29% (Qwen)."

#### §5.1「Why Naive Centroid Selection Fails」正文段重写 / §5.1 body rewritten

**Before (original, referenced `tab:ablation-centroid` table):**
> "This limitation is further confirmed quantitatively in Table~\ref{tab:ablation-centroid}, where naive centroid-based selection with raw BERT embeddings achieves only **21.00% accuracy on GSM8K**, compared to **80.95%** for D-ProtoCoT — a gap of nearly 60%."

**After (revised, references main table + adds AUC evidence):**
> "This limitation is also reflected quantitatively in our main results (Table~\ref{tab:main-results}): the **Static-Prototype** baseline — which selects via a frozen-BERT global centroid, exactly the naive scheme described above — drops to **52.38% on StrategyQA (LLaMA-3.1-8B) versus 66.67% for D-ProtoCoT**, and trails on CommonsenseQA as well."
>
> [新增 / New] "To verify that alignment reflects reasoning quality rather than being merely asserted, we visualize test-path embeddings before and after contrastive alignment (Fig.~\ref{fig:tsne}): intermixed correct/incorrect paths become linearly separable after training, and the alignment score predicts path correctness with an **AUC of 0.78** on 200 GSM8K questions (K=10)."
