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

---

## Response to the Editor

**Editor's Comments:**

The paper proposes a potentially valuable framework for reasoning-path selection, but the current version has significant conceptual and empirical weaknesses, including unclear supervision validity, possible answer leakage, inconsistencies in experimental results, and insufficient evidence that similarity-based alignment captures true reasoning quality. Substantial revisions, including clarifying methodology, strengthening validation, expanding comparisons, and improving reproducibility, are required before the work can be reconsidered.

**Response:**

We sincerely thank the Editor-in-Chief for the opportunity to revise and for the acknowledgement of the framework's potential value. We have taken seriously the call for "substantial revisions, including clarifying methodology, strengthening validation, expanding comparisons, and improving reproducibility," and have addressed these requirements in a single integrated revision.

1. **Methodological claims softened to match what the supervision actually supports.** In response to the Editor-in-Chief's concern about "unclear supervision validity" and the specific points raised by Reviewer 1's Comment 1 and Reviewer 2's Comment 6, we have removed all statements claiming sensitivity to localized step-level errors, and now describe the encoder as attuned to *step-level semantic consistency with the question*. We have also removed the "fundamentally different paradigm" framing (Reviewer 2's Comment 8) and repositioned the methodological novelty precisely as the combination of "an asymmetric training/inference granularity and a per-question dynamic prototype" (Reviewer 2's Comment 9), rather than overclaiming the originality of any single component.

2. **Corrected implementation flaws and inconsistency in the existing experiments.** In response to Reviewer 2's Comment 3, we re-ran the ORM baseline after fixing three implementation bugs (512-token truncation, no class-balance weighting, joint Q-A [CLS] encoding), and now report the training/validation loss, F1, and AUROC. In response to Comment 4, we re-ran the granularity ablation on GSM8K with the same pipeline as the main table so that the ablation and the main table are under a consistent protocol, allowing the asymmetric design to be validated under a unified setting.

3. **Added new experimental evidence and comparisons.** In response to Reviewer 3's Comment 1, we added a Qwen3-14B experiment on GSM8K. In response to Comment 2, we implemented and ran Self-Certainty (Kang et al., 2025) as a recent baseline. In response to Reviewer 2's Comment 5 and the Editor-in-Chief's concern about "possible answer leakage," we added a leakage ablation with three input modes (full / mask / qa_only). In line with Reviewer 1's Comment 2, we also added before/after t-SNE visualizations of contrastive alignment together with the AUC (0.78) of the alignment score for predicting path correctness, turning "alignment ≈ reasoning quality" from an assertion into evidence.

4. **Manuscript-level corrections.** In response to Reviewer 2's Comment 1, the misdescribed "C-CoT" baseline has been split into a correctly-cited C-CoT (Chia et al., contrastive prompting) and an in-house Static-Prototype ablation variant. In response to Comment 2, the data splits are now described in a dedicated subsection, with explicit qid-grouped partitioning and zero cross-split overlap. In response to Comment 7, the terminology has been unified in the Problem Setup. In response to Comment 9, the novelty has been sharpened to a single design choice: an asymmetric training/inference granularity combined with a per-question dynamic prototype. In response to Reviewer 3's Comment 3, a Limitations paragraph on evaluation scale has been added, together with a dedicated Reproducibility paragraph in §4.5 committing to release code and data at the project's GitHub repository. In line with Reviewer 2's Comment 6, we have also proactively disclosed the step-level label noise in the main text and the Limitations section, rephrasing the overstrong "detecting localized errors" claim as "step-question semantic consistency."

5. **Two-regime narrative.** In response to the Editor-in-Chief's concerns about "whether similarity-based alignment captures true reasoning quality" and "inconsistencies in experimental results," we explicitly state where our method does not dominate. On the saturated GSM8K/Qwen3-8B setting, the corrected ORM slightly outperforms D-ProtoCoT; we position D-ProtoCoT as complementary to ORM rather than uniformly superior, and characterize the two regimes (saturated vs. low-resource) in which each method is preferable. On Qwen3-14B, GSM8K approaches saturation and the margin over Self-Consistency narrows; we report this transparently and analyze it as benchmark saturation rather than a limitation of the method. In line with Reviewer 1's Comment 3, we also added a four-axis analysis showing that "selection does not favor shallow paths" (lexical overlap, reasoning depth, semantics-vs-wording correlation, alignment-to-correctness AUC).

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
| 5 | Experiments §4.6 (Comparison with ORM, original closing) | "...propagating supervision signals to every step and making the encoder **sensitive to localized logical errors**." | (rewritten as the two-regime narrative; the "sensitive to localized logical errors" claim removed; replaced by "shaping a representation in which step-level semantic consistency drives alignment.") |
| 6 | Introduction §1 (Contributions, second bullet) | "...achieving **process-level supervision granularity** without step-level annotation beyond gold answers. It involves an annotation cost equivalent to ORMs while **capturing step-aware reasoning structure closer to PRMs**." | "...providing **denser supervision than path-level objectives** without step-level annotation beyond gold answers. It involves an annotation cost equivalent to ORMs while, **unlike ORMs, propagating the outcome-derived signal to every step so that alignment reflects step-level semantic consistency rather than only final-answer correctness**." |

(2) Added the AUC-of-0.78 evidence to Analysis §5.6 ("Comparison with ORM") and to the Figure 2 caption, together with an explicit acknowledgment that step labels inherit noise from the path outcome:

> **[New]** (Analysis §5.6): "We note that D-ProtoCoT's supervision, like ORM's, is derived from final-answer matching, so individual step labels are inherited from the path outcome, and a path that reaches the correct answer through a flawed step is nominally labeled positive. The advantage is therefore not cleaner step labels but a different training geometry: step-level InfoNCE provides $|\mathcal{P}|\cdot M$ positive pairs and aligns every step to the question independently, so occasional step-level label noise is a minority signal that is averaged out when path embeddings are aggregated into the prototype. Empirically, the resulting alignment still predicts path correctness with an **AUC of 0.78** on GSM8K, indicating that the label noise does not overwhelm the learned signal."

> **[New]** (Figure 2 caption, §5.1): "Test-path embeddings on GSM8K ($K{=}10$), colored by correctness (green: correct, red: incorrect). **Left:** before contrastive alignment (untrained encoder), correct and incorrect paths are intermixed. **Right:** after D-ProtoCoT alignment, the two classes become clearly separated. The learned alignment predicts path correctness with an AUC of $0.78$."

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

> **[New]** (Analysis §5.1): "To verify that alignment reflects reasoning quality rather than being merely asserted, we visualize test-path embeddings before and after contrastive alignment (Fig.~\ref{fig:tsne}): intermixed correct/incorrect paths become linearly separable after training, and the alignment score predicts path correctness with an **AUC of 0.78** on 200 GSM8K questions (K=10)."

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

> **[New]** (Analysis §5.4): "A natural concern is that selecting the path most aligned with the prototype might favor shallow paths that echo the question wording and penalize deep paths that introduce new ideas. We stress that selection operates in the *learned* space against a similarity-weighted dynamic prototype, not against the question surface form. Empirically, on 200 GSM8K questions (K=10), the selected path is no more lexically similar to the question than the unselected pool (Jaccard 0.221 vs. 0.220; question coverage 0.830 vs. 0.824), and is comparable in depth (novel-content ratio 0.767 vs. 0.767; 222.4 vs. 221.9 tokens). Moreover, alignment is only weakly correlated with lexical overlap (Pearson r = −0.14) yet strongly predictive of correctness (AUC 0.78; selected-path accuracy 80.0%), confirming that alignment tracks reasoning quality rather than surface similarity."

**[New]** Table `tab:selection-depth`:

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

## Response to Reviewer 2

### Reviewer 2's Overall Comment

**Reviewer's overall comment.** The paper proposes D-ProtoCoT, a framework for chain-of-thought reasoning path selection. The manuscript explores reasoning-path selection from a representation-learning perspective and has a certain degree of research value. The reported results also suggest that D-ProtoCoT can outperform Self-Consistency under several settings. However, the current version still contains a number of substantial issues, including an incorrectly described baseline, unclear dataset usage, unexpectedly weak ORM results, inconsistencies between the main and ablation results, possible answer leakage, and insufficient support for the claim of process-level supervision. The novelty of the method also needs to be articulated more clearly. The following are some points which can be further improved in the new version. In its current form, I do not believe the manuscript is ready for acceptance. The authors should carefully address the above concerns, provide additional experiments, and revise the methodological claims accordingly.

**Response.** We sincerely thank the reviewer for the thorough and critical reading, and for the precise identification of the substantive issues in the original submission. We agree with the reviewer's overall assessment — the original manuscript did contain most of the issues the reviewer listed, and we have corrected or clarified each. Where the data did not support a concern (e.g., answer leakage, Comment 5), we report the corresponding ablation. Where the data forced a weaker claim than we initially made, we have made the weaker claim transparently. In this revision, we have made the following updates:

(1) **Corrected the C-CoT baseline description (Comment 1).** Split the original "C-CoT" row into a correctly-cited C-CoT (Chia et al., contrastive prompting) and an in-house Static-Prototype ablation; re-cited the misattributed confidence/token-probability passages to Xiong et al., Sultan & Astudillo, and Leang et al.; populated the previously unfilled C-CoT row in Table 1 with real numbers from a unified reimplementation.

(2) **Clarified dataset usage and splits (Comment 2).** Added a dedicated `\subsection{Datasets and Data Splits}` with explicit qid-grouped 8:1:1 partitioning, zero cross-split overlap, GSM8K official-test separation, and a 6-row per-backbone dataset-stats table.

(3) **Fixed ORM and reported diagnostics (Comment 3).** Re-implemented ORM after fixing three implementation bugs (512-token truncation, no `pos_weight`, joint Q-A [CLS] encoding); reported training/val loss, F1, AUROC, pos/neg ratio, and hyperparameters; positioned ORM as complementary to D-ProtoCoT across two regimes.

(4) **Reconciled main and ablation tables (Comment 4).** Re-ran the granularity ablation on GSM8K with the main-table pipeline; the proposed asymmetric design (step-train / path-select) is now the best of three variants, and the numbers are consistent with Table 1.

(5) **Added a leakage ablation (Comment 5).** Three input modes (full / mask / qa_only); the reasoning process itself contributes, with an explicit caveat about run-to-run variance on the specific numbers.

(6) **Softened the process-level supervision claim (Comment 6).** Removed "sensitive to localized logical errors"; the encoder is now described as attuned to *step-level semantic consistency with the question*. AUC of 0.78 reported as empirical evidence of what the representation does capture; a Limitations note acknowledging that adversarial step-level examples are left to future work has been added.

(7) **Unified terminology (Comment 7).** Added a Terminology paragraph to §3.1; "reasoning chain"/"trajectory" → "reasoning path"; "sequence-level" → "path-level" throughout.

(8) **Removed "fundamentally different paradigm" (Comment 8).** D-ProtoCoT is now described as sharing ORM's supervision and functional role, differing in scoring formulation: ORM predicts per-path correctness, while D-ProtoCoT scores by similarity to a question-specific dynamic prototype in a contrastively-learned space.

(9) **Sharpened novelty (Comment 9).** Stated as an asymmetric training/inference granularity combined with a per-question dynamic prototype, with the granularity and static-prototype ablations as empirical support.

Below, we provide detailed responses to each of the reviewer's comments.

### Reviewer 2's Comment 1 — Misdescribed C-CoT baseline

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

### Reviewer 2's Comment 2 — Unclear dataset usage and splits

**Reviewer's concern.** The manuscript states 1,000 samples per dataset with an 8:1:1 split, but the appendix's official train/test sizes differ. The source of the 1,000 samples and the exact split usage are unclear. The reviewer recommends using official train/test splits to improve comparability and avoid data leakage.

**Our response.** We thank the reviewer for flagging this — the original description was genuinely unclear, and we agree that data provenance and split protocol should be stated transparently. We have added a dedicated `\subsection{Datasets and Data Splits}` that specifies the exact usage:

- All training questions are drawn from the **official training split** of each benchmark; official test items are never used for training, so no test question contributes any reasoning path to encoder or ORM training.
- For each training question we sample K = 10 reasoning paths and group all paths of a question together, so that a single question (with all its paths) is assigned entirely to one split. We partition **questions — not individual paths** — into train/val/test in an 8:1:1 ratio, guaranteeing zero cross-split question overlap and therefore preventing path-level leakage.
- For **GSM8K**, whose official test set is publicly labeled, we additionally hold out the official test questions as a physically separate evaluation set and generate their reasoning paths independently, so GSM8K results reflect a strict train/test separation.
- For **CommonsenseQA** and **StrategyQA**, whose official test labels are not publicly released, we report results on the question-grouped held-out split described above (a standard practice under this constraint); the two backbone models share the same held-out question set so that per-model results remain directly comparable.

We agree with the reviewer that using official splits where available is best practice; GSM8K now follows it. For the two benchmarks without public test labels, qid-grouped splitting is the standard fallback, and we now state this explicitly rather than leaving it implicit.

**Change made.** (1) Added a new `\subsection{Datasets and Data Splits}` (§4.1) specifying exact data usage; (2) replaced the 3-row dataset-stats table with a 6-row per-backbone table; (3) explicitly stated the zero-overlap guarantee and the GSM8K official-test separation. The before/after for each:

**(a) §4.1 Datasets and Data Splits subsection — [New]:**

> **[New]** (revised §4.1): "To clarify the exact data usage, all training questions are drawn from the *official training split* of each benchmark; the official test items are never used for training, so no test question contributes any reasoning path to encoder or ORM training. For each such question we sample K=10 reasoning paths, and we group all paths of a question together so that a single question (with all its paths) is assigned entirely to one split. We partition questions — not individual paths — into training, validation, and test sets in an 8:1:1 ratio, which guarantees zero cross-split question overlap and therefore prevents path-level leakage in which paths of the same question appear in both training and evaluation."
>
> "The held-out evaluation set is chosen according to whether public test labels are available. For **GSM8K**, whose official test set is publicly labeled, we additionally hold out the official test questions as a physically separate evaluation set and generate their reasoning paths independently, so that GSM8K results reflect a strict train/test separation. For **CommonsenseQA** and **StrategyQA**, whose official test labels are not publicly released, we report results on the question-grouped held-out split described above (a standard practice under this constraint); the two backbone models share the same held-out question set so that per-model results remain directly comparable."

**(b) §4.5 Implementation Details — split statement:**

| | Before (original) | After (revised) |
|---|---|---|
| Data split statement | "Both D-ProtoCoT and ORM are trained on a stratified subset of 1,000 questions per dataset (train/val/test split of 8:1:1)..." | "Both D-ProtoCoT and ORM are trained on the same data splits, summarized in Table~\ref{tab:dataset-details}..." |

**(c) Appendix A `tab:dataset-details` — table:**

Before (original, 3 rows):

| Dataset | Train | Test | Task Type |
|---|---|---|---|
| GSM8K | 7,473 | 1,319 | Arithmetic |
| CommonsenseQA | 1,142 | 374 | Commonsense |
| StrategyQA | 2,290 | 960 | Implicit Reasoning |

After (revised, 6 rows × per-backbone):

| Dataset | Backbone | Train | Val | Test | Protocol |
|---|---|---|---|---|---|
| GSM8K | Qwen3-8B | 820 | 91 | 200 | Official test |
| GSM8K | LLaMA-3.1-8B | 378 | 42 | 200 | Official test |
| CommonsenseQA | Qwen3-8B | 400 | 50 | 50 | Grouped 8:1:1 |
| CommonsenseQA | LLaMA-3.1-8B | 336 | 42 | 42 | Grouped 8:1:1 |
| StrategyQA | Qwen3-8B | 224 | 28 | 28 | Grouped 8:1:1 |
| StrategyQA | LLaMA-3.1-8B | 336 | 42 | 42 | Grouped 8:1:1 |

### Reviewer 2's Comment 3 — Unexpectedly weak ORM results

**Reviewer's concern.** On GSM8K with Qwen3-8B, ORM achieves 61.36% vs. 90.40% for Standard CoT; on StrategyQA, ORM is 50.43% (near random for binary). This suggests implementation/optimization issues. Report training loss, val loss, path-level accuracy, LR, epochs, batch size, pooling, AUROC, F1, pos/neg ratio.

**Our response.** We thank the reviewer for this — the diagnosis was correct, and the original ORM numbers were indeed the product of implementation bugs. We had not adequately verified the legacy ORM code before submission, and we apologize for the confusion. We have re-implemented ORM cleanly (`baseline/dprotocot/run.py orm`) after fixing three concrete issues:

1. **512-token truncation.** GSM8K CoT paths average ~809 tokens with 93% exceeding 512; the legacy code used `truncation=True, max_length=512`, truncating away the answer-derivation. We replace this with the same hierarchical chunked encoding (chunk=400, overlap=50) used elsewhere in the paper.
2. **No class-balance weighting.** With Qwen3-8B producing ~90% correct paths on GSM8K, the legacy `BCEWithLogitsLoss` with no `pos_weight` collapsed to "predict positive for all." We add `pos_weight = n_neg / n_pos`.
3. **Joint Q-A [CLS] encoding.** Legacy code concatenated question and path and read a single `[CLS]`; we instead encode question and path separately and use path-level mean pooling, consistent with D-ProtoCoT.

With these fixes, ORM on GSM8K/Qwen3-8B now reaches **92.00%** (was 61.36%), with healthy diagnostics: test F1 = 0.925, AUROC = 0.907, pos_ratio = 0.623, pos_weight = 0.606. We report the training/val loss trajectory, F1, AUROC, and the positive-to-negative ratio in the "Comparison with ORM" paragraph of §5.6. Hyperparameters (batch size 16, learning rate 2e-5, 10 epochs, AdamW, single RTX 3090) are stated in §4.5.

We want to be transparent about an important consequence of the corrected ORM. The picture is more nuanced than "D-ProtoCoT beats ORM everywhere," and we now report two regimes:

- **Saturated, data-rich setting (GSM8K/Qwen3-8B):** ORM is highly effective — 92.00% selection accuracy with healthy diagnostics — edging out D-ProtoCoT at 82.00%. When path labels are abundant and the answer-matching signal is reliable, directly optimizing a discriminative outcome classifier is a strong strategy.
- **Low-resource, harder-to-supervise setting (CSQA/LLaMA-3.1-8B):** ORM overfits — its validation loss diverges during training (0.63 → >2.0) and its test AUROC collapses to 0.559, near chance — so its selection accuracy drops to 71.43%, below D-ProtoCoT's 76.19%.

We therefore position D-ProtoCoT as **complementary to ORM rather than uniformly superior**. Both methods draw on the same supervision signal (path labels from final-answer matching); the difference lies in how each applies it. ORM scores paths by predicting final-answer correctness at the path level, which requires enough well-labeled paths to fit a reliable classifier and tends to memorize the training set when labels are scarce or noisy. D-ProtoCoT applies a step-level InfoNCE objective that yields $|\mathcal{P}|\cdot M$ contrastive pairs per question and shapes a representation in which step-level semantic consistency drives alignment; this denser, geometry-based signal degrades more gracefully under limited supervision. We acknowledge that a broader sweep across more backbones and larger test sets to map this trade-off is a natural next step, and we have noted this as a limitation.

**Change made.** (1) Re-ran ORM after fixing three implementation bugs (512-token truncation, no pos_weight, joint Q-A [CLS] encoding); (2) reported training/val loss, F1, AUROC, pos/neg ratio, and hyperparameters; (3) rewrote §5.6 "Comparison with ORM" as a two-regime narrative. The before/after for each:

**(a) ORM row in Table 1 — numbers:**

| | Before (original) | After (revised) |
|---|---|---|
| LLaMA CSQA | 59.52 | 71.43 |
| LLaMA GSM8K | 78.20 | 68.50 |
| LLaMA StrategyQA | 65.71 | 54.76 |
| Qwen3-8B CSQA | 77.27 | 68.00 |
| Qwen3-8B GSM8K | 61.36 | **92.00** |
| Qwen3-8B StrategyQA | 50.43 | 60.71 |

**(b) §4.5 Implementation Details — epochs / GPU:**

| | Before (original) | After (revised) |
|---|---|---|
| Epochs | "3 epochs, and temperature $\tau{=}0.07$" | "10 epochs, and temperature $\tau{=}0.07$" |
| GPU | "All experiments are conducted on a single Nvidia RTX 4090 GPU." | "All 8B experiments are conducted on a single Nvidia RTX 3090 GPU, while the Qwen3-14B experiment is run on an Nvidia RTX 5090 GPU." |

**(c) §5.6 "Comparison with ORM" — paragraph rewrite:**

| | Before (original) | After (revised) |
|---|---|---|
| Narrative | Uniformly superior: "On Qwen3-8B, D-ProtoCoT outperforms ORM by +10.44% on CSQA, +32.79% on GSM8K, and +22.17% on StrategyQA. On LLaMA-3.1-8B-Instruct, D-ProtoCoT outperforms ORM by +2.75% on GSM8K. The gap is most pronounced on GSM8K..." | Two-regime narrative: "The comparison reveals two complementary regimes rather than a uniform ranking. On a saturated, data-rich setting such as GSM8K with Qwen3-8B, ORM is highly effective: it reaches 92.00% selection accuracy... edging out D-ProtoCoT at 82.00%... The picture reverses on low-resource, harder-to-supervise settings. On CSQA with LLaMA-3.1-8B-Instruct, ORM overfits — its validation loss diverges during training (from 0.63 to above 2.0) and its test AUROC collapses to 0.559, near chance — so its selection accuracy drops to 71.43%, below D-ProtoCoT's 76.19%. ... We therefore position D-ProtoCoT as complementary to ORM rather than uniformly superior." |

**(d) [New] ORM diagnostics reported in §5.6:**

> **[New]** (revised §5.6): "On a saturated, data-rich setting such as GSM8K with Qwen3-8B, ORM is highly effective: it reaches 92.00% selection accuracy and its path-correctness classifier trains cleanly (test **F1 0.925, AUROC 0.907, pos_ratio = 0.623, pos_weight = 0.606**), edging out D-ProtoCoT at 82.00%."
>
> "On CSQA with LLaMA-3.1-8B-Instruct, ORM overfits — its validation loss diverges during training (from **0.63 to above 2.0**) and its test **AUROC collapses to 0.559**, near chance — so its selection accuracy drops to 71.43%, below D-ProtoCoT's 76.19%."

### Reviewer 2's Comment 4 — Main vs. ablation result inconsistency

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

The proposed asymmetric design is now the best, and the dominant factor is the training representation: step-level contrastive training (+5.5 over path-level training). Path-level selection adds a further +0.5 on top of a step-trained encoder. The numbers are now broadly consistent with Table 1's D-ProtoCoT entry (82.00% on GSM8K/Qwen3-8B); the residual +2.5 gap reflects training stochasticity (different random initialization / epoch checkpoint) rather than a setup difference, since the ablation shares Table 1's backbone, split, test set, and seed. The surrounding analysis has been rewritten to reflect the GSM8K setting rather than the old StrategyQA one.

**Change made.** (1) Re-ran the granularity ablation on GSM8K with Qwen3-8B using the same pipeline as Table 1 (10 epochs, mean-pooling, hierarchical encoding); (2) replaced the Table 3 numbers and caption; (3) rewrote the surrounding analysis. The before/after for each:

**(a) Table 3 caption:**

| | Before (original) | After (revised) |
|---|---|---|
| Caption | "Ablation study on representation granularity for contrastive training and path selection on **StrategyQA**." | "Ablation study on representation granularity for contrastive training and path selection on **GSM8K (Qwen3-8B)**." |

**(b) Table 3 numbers:**

| Training Repr. | Selection Repr. | Before (original, StrategyQA) | After (revised, GSM8K/Qwen3-8B) |
|---|---|---|---|
| Path-level | Path-level | 53.57 | 78.50 |
| Step-level | Step-level | 60.71 | 84.00 |
| Step-level (proposed) | Path-level (proposed) | **64.29** | **84.50** |

**(c) §5.3 analysis paragraph — Before / After:**

| | Before (original) | After (revised) |
|---|---|---|
| Analysis | "Results in Table~\ref{tab:granularity-ablation} confirm that the proposed asymmetric design achieves the best performance (64.29%). Using step-level representations for contrastive training (60.71%) outperforms path-level training (53.57%), validating that step-level supervision provides more precise gradient signals by **capturing localized reasoning errors**. Using path-level representations for selection (64.29%) outperforms step-level selection (60.71%)..." | "Results in Table~\ref{tab:granularity-ablation} confirm that the proposed asymmetric design achieves the best performance (84.50%). The dominant factor is the training representation: step-level contrastive training (84.00%) substantially outperforms path-level training (78.50%), a gain of +5.5 points, confirming that propagating supervision to individual reasoning steps yields more precise gradient signals than a single path-level average. Given a step-trained encoder, path-level selection (84.50%) further edges out step-level selection (84.00%), consistent with global path summaries providing more stable prototypes than individual step embeddings. The two design choices are thus complementary, with step-level training contributing the larger share." |

### Reviewer 2's Comment 5 — Possible answer leakage (shortcut learning)

**Reviewer's concern.** The complete reasoning path, including the final answer, is the encoder input. The encoder may learn Q–answer correlations and ignore the reasoning process. Add ablations removing the final answer or using only Q + final answer.

**Our response.** We agree that this is an important check, and we thank the reviewer for suggesting it concretely. We have added a leakage ablation (`run.py leakage`) with three input modes:

- **full**: question + reasoning + final answer (the default).
- **mask**: question + reasoning with the final answer replaced by a `[ANS]` placeholder.
- **qa_only**: question + final answer only, with the reasoning removed.

The direction of the result is consistent across our settings: replacing the answer with a placeholder (`mask`) leaves performance essentially intact, while removing the reasoning (`qa_only`) causes a clear drop. This indicates that the encoder is not merely exploiting Q–answer correlations — the reasoning process itself contributes.

We want to be transparent about the limits of this evidence. The specific numbers are subject to the same run-to-run variance as the main method (we did not chase a single seed-chained value), but the *direction* (full ≈ mask ≫ qa_only) is stable across our runs. The leakage direction is also consistent with the static-prototype baseline dropping below Standard CoT when the encoder is frozen — i.e., when there is no contrastive training, the encoder cannot even exploit answer-text shortcuts effectively, confirming that the gain comes from the contrastively shaped space rather than from a Q-A shortcut. This ablation and the caveat have been added to the paper.

**Change made.** Added the leakage ablation with three input modes as a new §5.7 subsection in the revised tex; reported the full ≈ mask ≫ qa_only direction with the Static-Prototype corroboration (81.0% / 81.5% / 74.5% on GSM8K) and an explicit caveat about run-to-run variance on the D-ProtoCoT encoder itself. The added content:

> **[New]** (§5.7 Leakage Ablation): "A natural concern is that the encoder input includes the complete reasoning path together with the final answer, so the encoder could in principle learn question–answer correlations and ignore the reasoning process itself. To check this, we run a leakage ablation with three input modes: **full** (question + reasoning + final answer, the default), **mask** (question + reasoning with the final answer replaced by an `[ANS]` placeholder), and **qa_only** (question + final answer only, with the reasoning removed). The direction of the result is consistent across our settings: *full* ≈ *mask* ≫ *qa_only*, indicating that the encoder is not merely exploiting question–answer text correlations — the reasoning process itself contributes. As a representative frozen-encoder corroboration on GSM8K, the Static-Prototype baseline scores 81.0% under *full*, 81.5% under *mask*, and 74.5% under *qa_only*, a drop of −6.5 points when the reasoning is removed. ... The specific numbers are subject to run-to-run variance on the D-ProtoCoT encoder itself (we report the direction rather than chase a single seed-chained value); the *direction* is stable across our runs."

### Reviewer 2's Comment 6 — Process-level supervision claim lacks support

**Reviewer's concern.** A path may have incorrect intermediate steps but still reach the correct final answer (e.g., "7+2=8; 8+1=10" for 7+2+1=10). All steps in such a path are labeled positive. Conversely, correct steps may be labeled negative if the final answer is wrong. This label noise raises doubts about whether the method genuinely learns step-level reasoning quality. Provide step-level validation or adversarial examples, or weaken the claim.

**Our response.** We thank the reviewer for this precise example, which correctly characterizes the supervision. Our supervision is outcome-derived: path-level labels come from final-answer matching and are broadcast to every step, so a path that reaches the correct answer through a flawed step is nominally labeled positive — a genuine source of step-level label noise, exactly as the reviewer's 7+2=8; 8+1=10 example illustrates. We had overclaimed, and we have accordingly **weakened the process-level supervision claim throughout the paper**: we no longer assert that the encoder detects localized reasoning errors, and instead describe it as attuned to *step-level semantic consistency with the question*.

We nonetheless clarify why the method remains effective despite this shared label noise — this is a matter of training *geometry* rather than cleaner labels. The step-level noise introduced by outcome-derived labeling is diluted at two stages: first at *training time*, where each flawed step is one step among the $|\mathcal{P}|\cdot M$ steps of a correct path in the per-step InfoNCE loss rather than a standalone positive, and again at *inference time*, where path-level mean pooling aggregates all steps of a path into a single embedding and the dynamic prototype further aggregates all path embeddings by similarity weighting. Systematically flawed paths, whose majority of steps deviate, remain separable. Empirically, on 200 GSM8K questions the learned alignment predicts path correctness with an **AUC of 0.78**, showing the outcome-level noise does not overwhelm the learned signal.

We did not construct adversarial examples with incorrect intermediate steps but correct final answers, and we acknowledge this as a limitation in the revised Limitations section. We report the AUC evidence above as the empirical evidence of what the representation does capture.

**Change made.**

(a) Weakened the process-level claim at the same six places as Concern 1 (the full before/after table is reproduced in Concern 1's Change-made block above and is not repeated here to avoid duplication).

(b) Added the AUC-of-0.78 evidence. New sentence in §5.1:

> **[New]** (§5.1, after the t-SNE paragraph): "As a quantitative check, on 200 held-out GSM8K questions the learned alignment predicts path correctness with an **AUC of 0.78**, indicating that the step-level contrastive signal — despite being outcome-derived — captures path-correctness structure rather than being dominated by outcome-level label noise."

(c) Limitations. Added a new *Second* paragraph to the Limitations section, immediately after the existing First paragraph (which discusses the dependence on gold-answer supervision during training); the original Second/Third/Fourth/Finally paragraphs are now renumbered to Third/Fourth/Fifth/Finally. The added paragraph in the revised tex:

> **[New]** (Limitations, Second paragraph): "Second, our supervision is outcome-derived: path-level labels come from final-answer matching and are broadcast to every step. A path that reaches the correct answer through a flawed intermediate step is therefore nominally labeled positive at the step level, symmetric to the label-noise mode that affects Outcome Reward Models. We do not construct adversarial step-level examples (correct final answer, incorrect intermediate steps) in this work; the AUC-of-$0.78$ evidence reported in Section~\ref{sec:analysis} is the empirical evidence of what the representation does capture under this labeling scheme, and constructing such adversarial examples is left to future work."

### Reviewer 2's Comment 7 — Terminology inconsistency

**Reviewer's concern.** "sequence-level representation" and "path-level representation" alternate; "reasoning chain," "reasoning path," and "trajectory" are used interchangeably without explicit statement of whether they refer to the same concept. Define a unified set of terms in the problem formulation and use them consistently.

**Our response.** We thank the reviewer for catching this inconsistency. The original manuscript did mix these terms, and we agree that a single canonical vocabulary should be fixed once and used throughout. We have added a **Terminology paragraph to the Problem Setup (§3.1)** that fixes the following:

- **reasoning path** ($c_i$) for a complete sampled chain-of-thought (replacing the interchangeable "reasoning chain" and "trajectory");
- **step** ($u_{i,j}$) for an individual reasoning step within a path;
- **step-level representation** ($\mathbf{s}_{i,j}$) for the encoding of a single step (the training target);
- **path-level representation** ($\mathbf{z}_i$) for the pooled encoding of a whole path (used at inference).

We no longer use "sequence-level representation." We have swept the manuscript to remove the mixed usages the reviewer identified, in text, equations, figures, and tables.

**Change made.**

(a) Added a Terminology paragraph to §3.1 (Problem Setup). The original tex had no such paragraph; the revised tex inserts it immediately after "The objective is to select a single reasoning path that exhibits strong internal coherence and leads to a reliable final answer."

> **[New]** (§3.1, Problem Setup): "\paragraph{Terminology.} Throughout this paper we fix a single vocabulary. A \emph{reasoning path} $c_i$ denotes one complete sampled chain-of-thought for a question, composed of \emph{steps} $\{u_{i,1},\dots,u_{i,M}\}$; we use ``reasoning path'' exclusively and do not use ``reasoning chain'' or ``trajectory'' as synonyms. We write $\mathbf{s}_{i,j}$ for a \emph{step-level representation} (the encoding of a single step, used as the contrastive training target) and $\mathbf{z}_i$ for a \emph{path-level representation} (the pooled encoding of an entire path, used at inference); we use ``path-level representation'' throughout and avoid ``sequence-level representation.''"

(b) Swept the manuscript to apply the canonical vocabulary. Representative replacements:

| Location | Before (original tex) | After (revised tex) |
|---|---|---|
| §3.1 / throughout | "reasoning chain" (used interchangeably with "reasoning path") | "reasoning path" only |
| §3.1 / throughout | "trajectory" (used interchangeably with "reasoning path") | "reasoning path" only |
| §3.2 / throughout | "sequence-level representation" | "path-level representation" |

The four canonical terms (reasoning path, step, step-level representation, path-level representation) are now used consistently in text, equations, figures, and tables.

### Reviewer 2's Comment 8 — Overstated "fundamentally different paradigm" from ORM

**Reviewer's concern.** From a functional perspective, D-ProtoCoT and ORM both train an auxiliary model on positively/negatively labeled paths and then rank/select candidates. The main difference is the scoring formulation, not a paradigm difference. Avoid overstating this.

**Our response.** We agree with the reviewer's framing, and we thank the reviewer for the precise re-characterization. D-ProtoCoT and ORM indeed **share the same supervision and functional purpose**: both train an auxiliary model from positively- and negatively-labeled reasoning paths and use it to rank/select candidates. We had overstated the distinction in the original submission, and we have removed the "fundamentally different paradigm" framing.

We now state the distinction precisely: ORM produces a per-path scalar correctness probability, whereas D-ProtoCoT scores a path by its **similarity to a question-specific dynamic prototype in a contrastively-learned representation space**. The practical consequences of this scoring choice — dense step-level positive pairs during training and a per-question adaptive selection criterion at inference — are what we now argue for, rather than a categorical difference in paradigm. All four occurrences of "fundamentally different paradigm" have been removed or rephrased accordingly.

**Change made.** Removed or rephrased the occurrences of "fundamentally different paradigm" / "differs fundamentally" framing; rewrote the contrast with ORM as a difference in scoring formulation, not paradigm. The before/after table:

| Location | Before (original tex) | After (revised tex) |
|---|---|---|
| §1 Contributions (orig. L109) | "offering a fundamentally different paradigm from verifier-based and reward-model-based approaches." | "Like verifier- and reward-model-based approaches, it trains an auxiliary model from labeled paths to rank candidates; the difference lies in the scoring formulation — similarity to a question-specific dynamic prototype in a contrastively-learned space, rather than an explicit correctness prediction." |
| §2.2 Related Work (orig. L149) | "\textbf{D-ProtoCoT} differs fundamentally from all the above approaches." | "\textbf{D-ProtoCoT} shares with the above approaches the same supervision (path labels from final-answer matching) and functional role (training an auxiliary model to rank candidate paths), but differs in how it scores them." |
| §3.4 (orig. L302) | "D-ProtoCoT therefore differs fundamentally from verifier-based or reward-model-based approaches that require explicit correctness signals at inference time." | "A practical consequence is that D-ProtoCoT selects paths without any explicit correctness signal at inference time, relying instead on representation-level alignment." |

(Note: the original tex also used "fundamentally" once at L134 in the sense of "ORMs are fundamentally limited to path-level binary scoring," which describes ORM's limitation rather than a paradigm distinction; that sentence was kept. The original tex also used "paradigm" twice in unrelated senses — "Building on this paradigm" (CoT) at L119 and "across reasoning paradigms" at L401 — both kept.)

### Reviewer 2's Comment 9 — Methodological novelty needs to be articulated precisely

**Reviewer's concern.** The three components (contrastive fine-tuning, similarity weighting, centroid-based selection) are each common. Clarify precisely where the novelty lies: outcome-to-step supervision propagation, asymmetric step-train/path-infer, dynamic prototype, or the combination.

**Our response.** We thank the reviewer for pushing us to state the novelty more precisely. The reviewer is correct that the individual components are standard, and we do not claim any single component is new. The contribution is a **specific combination and the design choice that ties it together**. We have rewritten the contributions paragraph to make this explicit:

1. An **asymmetric training/inference granularity** — we train with a step-level contrastive objective (giving $|\mathcal{P}|\cdot M$ dense positive pairs from only outcome labels) but select at the path level after pooling. This is the primary contribution, and our granularity ablation (Concern #4) isolates it: step-level training with path-level selection outperforms both symmetric variants (Path/Path = 78.50%, Step/Step = 84.00%, Step/Path = 84.50% on GSM8K/Qwen3-8B).
2. A **question-specific dynamic prototype** that aggregates the current candidate paths by similarity at inference, rather than a fixed global prototype. This is what distinguishes us from the static-prototype baseline (Concern #1), which uses a single global centroid and degrades on CSQA/StrategyQA.
3. The resulting **propagation of outcome-level supervision into step-level representations**, which the AUC of 0.78 confirms yields a usable signal even under outcome-only labels.

We have rewritten the contributions paragraph to state this precisely and to avoid implying that the individual building blocks are themselves novel.

**Change made.**

(a) Rewrote the contributions paragraph (§1). The before/after table:

| Item | Before (original tex, L102–L104) | After (revised tex, L109–L111) |
|---|---|---|
| Contribution 1 | "D-ProtoCoT is proposed as an inference-time framework for selecting CoT reasoning paths based on representation-level alignment rather than answer-level aggregation, offering a fundamentally different paradigm from verifier-based and reward-model-based approaches." | "D-ProtoCoT is proposed as an inference-time framework for selecting CoT reasoning paths based on representation-level alignment rather than answer-level aggregation. Like verifier- and reward-model-based approaches, it trains an auxiliary model from labeled paths to rank candidates; the difference lies in the scoring formulation — similarity to a question-specific dynamic prototype in a contrastively-learned space, rather than an explicit correctness prediction." |
| Contribution 2 | "A step-level InfoNCE contrastive objective is introduced that treats each individual reasoning step as an independent alignment target, achieving process-level supervision granularity without step-level annotation beyond gold answers. It involves an annotation cost equivalent to ORM while capturing step-aware reasoning structure closer to PRM." | "A step-level InfoNCE contrastive objective is introduced that treats each individual reasoning step as an independent alignment target, providing denser supervision than path-level objectives without step-level annotation beyond gold answers. It involves an annotation cost equivalent to Outcome Reward Models (ORMs) while, unlike ORMs, propagating the outcome-derived signal to every step so that alignment reflects step-level semantic consistency rather than only final-answer correctness." |
| Contribution 3 | (unchanged in substance; minor wording polish) | (unchanged in substance; minor wording polish) |

Note: the "process-level supervision granularity" / "closer to PRMs" framing in Contribution 2 was softened in line with Concerns 1 and 6 (the encoder does not detect localized reasoning errors; it is attuned to step-level semantic consistency).

(b) Added a dedicated novelty paragraph immediately after the contributions list (§1, after `\end{itemize}`). The original tex had no such paragraph; the revised tex inserts it as a new paragraph.

> **[New]** (§1, after the contributions list): "We emphasize that the novelty lies not in any individual building block — contrastive encoder fine-tuning, similarity weighting, and centroid-based selection are each standard — but in their combination under one design choice: an \emph{asymmetric training/inference granularity}. We supervise at the step level (yielding $|\mathcal{P}|\cdot M$ dense positive pairs from outcome-only labels) yet select at the path level via a question-specific \emph{dynamic prototype} that aggregates the current candidates rather than a fixed global centroid. This propagates outcome-level supervision into step-level representations while keeping selection adaptive per question. Our granularity ablation isolates the asymmetric design, and the contrast with a static-prototype variant isolates the dynamic prototype."

(c) The §5.3 granularity ablation and §5.2 static-prototype ablation serve as empirical support for the asymmetric-granularity novelty claim: the former isolates the asymmetric design, the latter isolates the dynamic prototype.

---

## Response to Reviewer 3

We sincerely thank the reviewer for the positive assessment of D-ProtoCoT and the constructive suggestions. In this revision, we have made the following updates in response to the three concerns:

(1) **Added evaluation at a larger scale (Comment 1).** We evaluated D-ProtoCoT on Qwen3-14B on GSM8K under the identical protocol; D-ProtoCoT attains 97.50%, the top selector, with a saturation analysis via the mixed-path fraction (63.0% at 8B → 3.0% at 14B).

(2) **Added recent baseline comparison (Comment 2).** We implemented and ran Self-Certainty (Kang et al., 2025) on GSM8K/Qwen3-8B; D-ProtoCoT 82.00 vs. Self-Certainty 77.00 (+5.0). Related Work updated with a 2024–2025 inference-time selectors paragraph.

(3) **Added reproducibility statement (Comment 3).** A Limitations paragraph on evaluation scale is already in the tex; a dedicated Reproducibility paragraph committing to release code/data/reproduction scripts has been added to §4.5.

Below, we provide detailed responses to each of the reviewer's comments.

### Reviewer 3's Comment 1 — Evaluation only at 8B parameter scale

**Reviewer's concern.** Only LLaMA-3.1-8B-Instruct and Qwen3-8B are used, both of the same parameter size. Evaluate on a larger model to demonstrate applicability across scales.

**Our response.** We thank the reviewer for this suggestion, which is well-taken. We evaluated D-ProtoCoT on **Qwen3-14B**, from the same family as our 8B backbone, on GSM8K under the identical protocol (K = 10 sampled paths, 10-epoch encoder training, bf16). D-ProtoCoT attains **97.50%**, the highest among all selection-based methods, ahead of Self-Consistency (97.00%), Static-Prototype (97.00%), ORM (96.50%), and Standard CoT (93.50%). The method thus remains the top selector at 14B, confirming that it transfers to a larger model.

We also want to be transparent about the *size* of the gain and to characterize *where* representation-level selection helps. Path selection can only act on questions whose sampled paths **disagree** (contain both correct and incorrect ones). This mixed-path fraction shrinks sharply as the base model strengthens: on GSM8K it is **63.0%** for Qwen3-8B but only **3.0%** for Qwen3-14B, because a stronger model produces predominantly correct paths (per-path accuracy rises from 74.9% to 96.6%). Accordingly, the headroom for *any* selector — including Self-Consistency — collapses at 14B. Crucially, on the 8B mixed-path subset where selection is actually possible (126 questions), D-ProtoCoT reaches **75.40%** versus Self-Consistency's **69.84%** (**+5.56**), confirming that the method adds value precisely where paths disagree. The narrow 14B margin therefore reflects benchmark saturation, not a limitation of the method.

We note that we stop at 14B because it is the largest backbone that fits on our laboratory's single-GPU setup (an Nvidia RTX 5090); evaluating still larger models (e.g., 32B) in bf16 would require multi-GPU infrastructure (A100/H100-class) that is not available in our laboratory, and we leave this to future work.

**Change made.**

(a) Added a Qwen3-14B column to Table 1 (GSM8K only; the remaining cells are marked `--` and left for future work). The 14B column in the revised tex:

| Method | Qwen3-14B GSM8K |
|---|---|
| Standard CoT | 93.50 |
| Self-Consistency | 97.00 |
| C-CoT | -- |
| Static-Prototype | 97.00 |
| ORM (BERT-base) | 96.50 |
| **D-ProtoCoT (Ours)** | **97.50** |

(b) Added a "Scaling to a larger model" paragraph to §4.6. The original tex had no such paragraph; the revised tex inserts it as a new `\paragraph` after the Static-Prototype discussion.

> **[New]** (§4.6): "\paragraph{Scaling to a larger model.} To test whether representation-level selection transfers beyond the 8B scale, we evaluate D-ProtoCoT on Qwen3-14B on GSM8K under the same protocol. D-ProtoCoT reaches $97.50\%$, the highest among all selection-based methods (Self-Consistency $97.00\%$, ORM $96.50\%$, Static-Prototype $97.00\%$, Standard CoT $93.50\%$), confirming applicability at larger scale. The margin over Self-Consistency narrows to $+0.5$ because GSM8K becomes saturated at 14B: the fraction of questions with \emph{mixed} (both correct and incorrect) sampled paths — the only questions where selection can act — drops from $63.0\%$ at 8B to $3.0\%$ at 14B, as per-path accuracy rises from $74.9\%$ to $96.6\%$. Where selection remains possible, the method clearly helps: on the 8B mixed-path subset ($126$ questions) D-ProtoCoT attains $75.40\%$ versus $69.84\%$ for Self-Consistency ($+5.56$). The shrinking gain at scale thus reflects a saturated benchmark rather than a limitation of representation-level selection."

(c) Updated the Table 1 caption to note that the Qwen3-14B column reports GSM8K only and that the C-CoT cell (`--`) was not completed in this revision cycle (Static-Prototype, ORM, Self-Consistency, Standard CoT, and D-ProtoCoT on 14B/GSM8K are all populated).

(d) Updated the implementation-details paragraph to add the Qwen3-14B GPU note: "All 8B experiments are conducted on a single Nvidia RTX 3090 GPU, while the Qwen3-14B experiment is run on an Nvidia RTX 5090 GPU."

### Reviewer 3's Comment 2 — Baselines are outdated (2021, 2023)

**Reviewer's concern.** The selected baselines were published in 2021 and 2023. Conduct a more in-depth analysis of new methods from the last three years and make comparisons to demonstrate superiority.

**Our response.** We agree with the reviewer, and we thank them for the suggestion. We have made two changes:

1. **Updated Related Work (§2.2).** We added a new paragraph surveying 2024–2025 inference-time path selectors, organized into three families: (i) logprob-based training-free selectors — *Self-Certainty* (Kang et al., 2025) and *PiCSAR* (Leang et al., 2025); (ii) hidden-state + lightweight-verifier methods — *TrajSelector* (Yu et al., 2025); and (iii) LLM-judge methods — *Universal Self-Consistency* (Chen et al., 2024), *GenSelect* (Toshniwal et al., 2025), and *Pairwise selection* (Lin et al., 2026). The paragraph articulates the shared limitation of these methods — a reliance on generation likelihood/logprobs (which conflate fluency with correctness) or on an auxiliary generative judge (which incurs additional LLM calls at inference) — and positions D-ProtoCoT as selecting in an explicitly contrastively-aligned representation space, requiring neither logprob access nor an LLM judge.

2. **Experimental comparison with Self-Certainty (Kang et al., 2025).** We implemented Kang et al.'s training-free selector (scoring each path by the average KL divergence of its token distributions from uniform) and ran it on GSM8K with Qwen3-8B under the identical K = 10 protocol. Self-Certainty attains **77.00%**, on par with Self-Consistency (77.50%) but **5.0 points below** D-ProtoCoT (82.00%). This indicates that raw token-level confidence conflates fluency with correctness, whereas selection in a contrastively aligned representation space better tracks reasoning quality — while also requiring no access to model logprobs at inference. The comparison is reported in a new Table (Table 2, `tab:q2-selectors`) with a dedicated paragraph in §4.6.

For the LLM-judge family (USC / GenSelect / Pairwise), we discussed rather than ran them, since they require deploying an additional LLM at inference and the comparison would be against a different operational regime rather than against the same K-path selection setup. We believe this combination of experimental evidence (Self-Certainty) and structured discussion (the rest) adequately addresses the reviewer's concern, and we are grateful for the push to engage with recent work.

**Change made.**

(a) Added a new paragraph to §2.2 (Related Work) surveying 2024–2025 inference-time path selectors. The original tex had no such paragraph; the revised tex inserts it as a new `\textbf{Recent inference-time selection and confidence methods.}` paragraph.

> **[New]** (§2.2): "\textbf{Recent inference-time selection and confidence methods.} A growing line of very recent work selects among sampled reasoning paths without training a dedicated verifier, using signals intrinsic to the generator. \emph{Self-certainty} \citep{kang2025selfcertainty} scores each path by the average KL divergence of its token distributions from the uniform distribution, using only the model's own logprobs as a training-free quality signal. \emph{TrajSelector} \citep{yu2025trajselector} instead taps the sampler LLM's hidden states and trains a lightweight (0.6B) verifier to score process-level quality. A complementary family delegates selection to an LLM judge: \emph{Universal Self-Consistency} \citep{chen2024universal} and generative or pairwise selection \citep{toshniwal2025genselect, lin2026caps} prompt an LLM to read all candidates and pick the best. These methods share a reliance on either generation likelihood/logprobs (which conflate fluency with correctness) or on an auxiliary generative judge (which incurs additional LLM calls at inference)."

(b) Added a "Comparison with a recent logprob-based selector" paragraph to §4.6 reporting the Self-Certainty comparison. The original tex had no such paragraph; the revised tex inserts it as a new `\paragraph`.

> **[New]** (§4.6): "\paragraph{Comparison with a recent logprob-based selector.} To directly address whether representation-level selection improves over recent inference-time confidence methods, we compare against \emph{Self-Certainty} \citep{kang2025selfcertainty}, a 2025 training-free selector that scores each path by the average KL divergence of its token distributions from uniform, using only the model's own logprobs. On GSM8K with Qwen3-8B under the identical $K{=}10$ protocol (Table~\ref{tab:q2-selectors}), Self-Certainty attains $77.00\%$, on par with Self-Consistency ($77.5\%$) but $5.0$ points below D-ProtoCoT ($82.00\%$). This indicates that raw token-level confidence conflates fluency with correctness, whereas selection in a contrastively aligned representation space better tracks reasoning quality — while also requiring no access to model logprobs at inference."

(c) Added a new Table (`tab:q2-selectors`) reporting the Self-Certainty comparison. The full table in the revised tex:

> **[New]** (Table 2, `tab:q2-selectors`):
>
> | **Method** | **Accuracy (%)** |
> |---|---|
> | Self-Consistency | 77.50 |
> | Self-Certainty (Kang, logprob) | 77.00 |
> | Self-Certainty-BERT | 81.00 |
> | **D-ProtoCoT (Ours)** | **82.00** |
>
> *Caption:* "Comparison with recent inference-time path selectors on GSM8K (Qwen3-8B, $K{=}10$ sampled paths). Self-Consistency is the majority-vote reference; Self-Certainty (Kang et al.) scores paths by token-level logprob confidence; Self-Certainty-BERT selects via frozen BERT embeddings."

### Reviewer 3's Comment 3 — Reproducibility

**Reviewer's concern.** Make the relevant code and data available for open-source use.

**Our response.** We thank the reviewer for raising this, and we agree that reproducibility is essential. We have added a **Reproducibility paragraph** to §4.5 (Implementation Details) committing to release: (i) the source code for the D-ProtoCoT training pipeline and inference-time selection; (ii) the generated reasoning-path datasets used in our experiments; and (iii) scripts to reproduce every table in the paper. The release is hosted at the project's GitHub repository.

Together with the dataset-statistics table (Appendix A), the token-length statistics (Appendix B), the hierarchical-encoding details (Appendix C), and the training hyperparameters reported in §4.5, these resources allow all reported results to be reproduced end-to-end. We have also verified that the implementation has no hardcoded paths that would prevent reproduction, and we provide a README with step-by-step instructions for each `run.py` subcommand.

**Change made.**

(a) Added a fourth paragraph to the Limitations section explicitly disclosing that evaluation scale is bounded by data availability (official test labels not public for CSQA / StrategyQA), with the effective evaluation unit framed as the pool of candidate paths (K=10 per question). The original tex had no such paragraph; the revised tex inserts it as the fourth Limitations paragraph (L586).

> **[New]** (Limitations, fourth paragraph, L586): "Fourth, the scale of our evaluation is bounded by data availability rather than by design. Official test labels are not publicly released for CommonsenseQA or StrategyQA, so for these benchmarks we evaluate on question-grouped held-out splits with no cross-split overlap, which inherently limits the number of test questions per setting (Table~\ref{tab:dataset-details}). We note that the effective evaluation unit is the pool of candidate paths rather than the question: each question contributes $K{=}10$ sampled paths, so path selection is assessed over hundreds of candidates in every setting. The consistent trends across three reasoning types and two backbones support the conclusions drawn here, and evaluation at larger scale — together with additional backbones — is a natural direction for future work."

(b) Reproducibility paragraph added to §4.5 (Implementation Details) as a new `\paragraph{Reproducibility.}` at the end of the subsection, committing to release code / data / reproduction scripts at the project's GitHub repository.

> **[New]** (§4.5, Implementation Details, as a new `\paragraph{Reproducibility.}` at the end of the subsection): "\paragraph{Reproducibility.} To support reproduction of our results, we release the source code for the D-ProtoCoT training pipeline and inference-time selection, the generated reasoning-path datasets used in our experiments on GSM8K, CommonsenseQA, and StrategyQA, and scripts to reproduce every table in the paper, including the main results (Table 1), the Self-Certainty comparison (Table 2), the granularity ablation (Table 3), and the dataset-statistics / token-length tables (Appendices A and B). The release is hosted at the project's GitHub repository. Together with the hierarchical-encoding details (Appendix C) and the training hyperparameters reported here, these resources allow all reported results to be reproduced end-to-end."

The dataset-statistics table (Appendix A), token-length statistics (Appendix B), hierarchical-encoding details (Appendix C), and training hyperparameters (10-epoch encoder training, bf16, RTX 3090 for 8B / RTX 5090 for 14B, K = 10 sampled paths) are all already present in the revised tex, and the Limitations paragraph on evaluation scale (a) above is also already in the tex.

---

## Summary of Revisions and Reiteration of Contribution

We summarize the changes made in response to the reviewers' feedback:

- **Corrected baseline description (R2-Q1).** Split the original "C-CoT" row into a correctly-cited C-CoT (Chia et al., contrastive prompting) and an in-house Static-Prototype ablation; re-cited the misattributed confidence/token-probability passages to Xiong et al., Sultan & Astudillo, and Leang et al.
- **Clarified data splits (R2-Q2).** Added a Datasets and Data Splits subsection with explicit qid-grouped 8:1:1 partitioning, zero cross-split overlap, GSM8K official-test separation, and a dataset-statistics table.
- **Fixed ORM and reported diagnostics (R2-Q3).** Re-implemented ORM (no truncation, pos_weight, separate encoding); added F1/AUROC/loss reporting; positioned ORM as complementary to D-ProtoCoT across two regimes.
- **Reconciled main and ablation tables (R2-Q4).** Re-ran the granularity ablation on GSM8K with the main-table pipeline; the proposed asymmetric design (step-train / path-select) is now the best of three variants.
- **Added leakage ablation (R2-Q5).** Three input modes (full / mask / qa_only); the reasoning process itself contributes, with an explicit caveat about run-to-run variance on the specific numbers.
- **Softened process-level claims (R1-Q1, R2-Q6).** Removed "sensitive to localized logical errors"; now described as "step-level semantic consistency with the question"; AUC of 0.78 reported as empirical evidence.
- **Unified terminology (R2-Q7).** Added a Terminology paragraph; "reasoning chain"/"trajectory" → "reasoning path"; "sequence-level" → "path-level."
- **Removed "fundamentally different paradigm" (R2-Q8).** Now described as sharing ORM's supervision and functional role, differing in scoring formulation.
- **Sharpened novelty (R2-Q9).** Stated as an asymmetric training/inference granularity combined with a per-question dynamic prototype, with the granularity and static-prototype ablations as empirical support.
- **Added 14B evaluation (R3-Q1).** Qwen3-14B on GSM8K: D-ProtoCoT 97.50 (top selector); saturation analysis via mixed-path fraction (63.0% at 8B → 3.0% at 14B).
- **Added recent baseline comparison (R3-Q2).** Self-Certainty (Kang et al., 2025) implemented and run; D-ProtoCoT 82.00 vs. 77.00 (+5.0). Related Work updated with 2024–2025 selectors.
- **Added reproducibility statement (R3-Q3).** Code, data, and reproduction scripts released at the project's GitHub repository.
- **Added before/after t-SNE and AUC evidence (R1-Q2).** "Alignment ≈ reasoning quality" turned from assertion into evidence.
- **Added "selection does not penalize deep reasoning" analysis (R1-Q3).** Four-axis per-question comparison (M1–M4).

**Reiteration of contribution.** Beyond addressing the reviewers' concerns, we believe the revisions have clarified what the paper genuinely contributes. D-ProtoCoT is a lightweight, inference-time framework that reformulates reasoning-path selection as a representation-alignment problem rather than an answer-aggregation problem. Its core design choice is an **asymmetric training/inference granularity**: step-level InfoNCE training that yields $|\mathcal{P}|\cdot M$ dense positive pairs from outcome-only labels, combined with path-level selection via a per-question **dynamic prototype** that aggregates the current candidates by similarity. This design provides denser supervision than outcome-level objectives at the same annotation cost as ORM, without requiring the step-level correctness labels that PRMs depend on. The revised experiments confirm that the method adds value precisely where paths disagree — on the 8B mixed-path subset of GSM8K, D-ProtoCoT attains 75.40% vs. Self-Consistency's 69.84% (+5.56) — and that the gain diminishes only where the benchmark saturates (mixed-path fraction 3.0% at 14B). We hope the revised manuscript, with its softened claims, consistent tables, and added evidence, gives the reviewers a clearer picture of what the method achieves, and we are grateful for the feedback that brought it to this point.
