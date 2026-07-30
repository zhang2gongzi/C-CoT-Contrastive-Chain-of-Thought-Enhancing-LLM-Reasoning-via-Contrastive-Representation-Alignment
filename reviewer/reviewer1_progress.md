# 审稿人 #1 回复进度

---

## 审稿意见概述

D-ProtoCoT 的动机明确（PRM 需要昂贵的人工标注），但有以下担忧：

---

## 问题 1：步级标签噪声
- **问题**：监督仅来自最终答案匹配，路径级标签被广播到每一个步骤。包含有缺陷步骤但碰巧得出正确答案的路径会被视为正样本。这是 ORM 的典型失败模式，但本文却声称对局部逻辑错误具有敏感性。
- **状态**：待处理

---

## 问题 2：对齐 ≈ 推理质量 仅被断言
- **问题**：建议在图 2 中添加训练后的对比可视化图，并补充定量指标（如相似度预测路径正确性的 AUC）
- **状态**：待处理

---

## 问题 3：相似度选择可能惩罚深度推理
- **问题**：按与问题的相似度选择路径，可能会惩罚那些远离问题原始措辞、引入新想法或进行非显而易见逻辑跳跃的路径，同时奖励只是重复问题的浅层路径
- **状态**：待处理

---

## 附录：审稿人 #1 原话

Reviewer #1: This paper proposes D-ProtoCoT, an inference-time framework for selecting chain-of-thought reasoning paths. Instead of fine-tuning the language model, it trains a lightweight auxiliary encoder with a step-level InfoNCE objective (supervised by gold-answer matching) to build a representation space in which correct paths align with their question. At inference, it aggregates the sampled paths into a similarity-weighted dynamic prototype and selects the path best aligned with it.

Motivation is clear which PRM requires expensive human label is well-known problem but I do have some concerns.

1. Supervision comes only from final-answer matching, with path-level labels broadcast to every step. How do you ensure the selected path's intermediate steps are correct, e.g., a path with flawed steps that reaches the right answer by coincidence (and is thus treated as positive)? This is ORM's failure mode, yet the paper claims sensitivity to localized logical errors.

2. The premise "alignment ≈ reasoning quality" is asserted but not demonstrated. I recommend adding a post-training counterpart to Figure 2 (representation space after contrastive alignment), plus a quantitative measure (e.g., AUC of similarity predicting path correctness), to make the claim convincing.

3. My main concern is conceptual. D-ProtoCoT picks the path that is most similar to the question, and the prototype weights favor paths that stay close to it. But good reasoning often moves away from the wording of the question, bringing in new ideas not in the prompt or making non-obvious jumps. Selecting by similarity may punish these deep but less-similar paths, while rewarding shallow ones that just repeat the question. Does choosing paths by similarity to the question hurt the model's deeper thinking, favoring paths that stay on topic over paths that actually reason better?
