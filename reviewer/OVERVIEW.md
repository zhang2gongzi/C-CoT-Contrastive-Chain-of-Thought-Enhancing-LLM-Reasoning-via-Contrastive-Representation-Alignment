# 审稿回复总览（OVERVIEW）

> 跨审稿人全局进度。详情见各 `reviewerN_progress.md`。最后更新：2026-08-04。
> 图例：✅ 已进 tex / 完成　🟡 数据部分就绪，待收尾　🔴 硬卡点　⚪ 暂缓

---

## 审稿人 #1（3 条）——✅ 全部已进 tex，无待办
| 条 | 问题 | 状态 |
|---|---|---|
| 1 | 步级标签噪声 | ✅ 软化 claim（6 处）+ AUC 0.78 兜底 |
| 2 | 对齐≈质量仅断言 | ✅ before/after t-SNE 图 + AUC 0.78 |
| 3 | 相似度惩罚深度推理 | ✅ M1–M4 分析 + 小表 |

---

## 审稿人 #2（9 条，最复杂）

### A 类 纯文本（条 6/7/8/9）——✅ 全部已进 tex（2026-08-03）
| 条 | 内容 |
|---|---|
| 6 | 过程级监督标签噪声 → 软化措辞 + AUC 0.78 |
| 7 | 术语统一 → Terminology 段 + "reasoning chain→path" 8 处 |
| 8 | 删 "fundamentally different paradigm"（4 处）|
| 9 | novelty 精确定位段（不对称粒度 + 动态原型组合）|

### B 类 已完成
- 条 2（数据切分）✅ qid 分组 8:1:1，无泄露

### C 类 需实验（数据部分回来，但 X 未统一）
| 条 | 实验结论 | 状态 |
|---|---|---|
| 3 ORM | D-ProtoCoT 92 > ORM 82；ORM 已修好（TEST F1=0.925 / AUROC=0.907）| 🟡 tex ORM 格已改 82.0 |
| 4 主表≠消融 | **核心卡点**：GSM8K/Qwen 主方法 4 个值打架（94.15 / 92 / 86.5 / 81）| 🔴 必须种子重跑统一 X |
| 5 泄露 | full 81 ≈ mask 81.5 ≫ qa_only 74.5（增益非答案泄露）| 🟡 方向成立，数字随 X |
| 1 C-CoT | 真 Chia prompting 重跑：93.88（GSM8K/Qwen）、76.81（SQA/LLaMA）| 🟡 tex 填 2/6 格 |

---

## 审稿人 #3（3 条）
| 条 | 问题 | 状态 |
|---|---|---|
| Q1 | 只测 8B，要更大模型 | 🟡 14B 五数已跑（D-ProtoCoT 97.5 最高，饱和 +0.5）；rebuttal/正文草稿就绪；**卡混合题分析 `XX%`** |
| Q2 | 基线太旧 | 🟡 §2.2 新基线综述已进 tex；Self-Certainty 实测 89(8B)/97.5(14B)，**待进表** |
| Q3 | 可复现性 | ⚪ 决定交 rebuttal 时再动（仓库改名、清硬编码路径、README）|

---

## 🔴 当前唯一硬卡点：GSM8K/Qwen 的 X（审稿人 #2 条 4）
一个数须**同时满足三重约束**：
1. **一致**：主表 = granularity(step/path) = leakage(full) 三处同一个 X
2. **> 对称变体**：X > path/path(83)、step/step(82) → 条 9 novelty
3. **≥ 89**：否则输给 Self-Certainty 新基线(89) → Q2 帮倒忙

昨天无种子批（81~92 乱跳）做不到 → **解法：`--seed 42 --epochs 10` 重跑**。

---

## 已落地 tex 的数字（截至 2026-08-04）
- Table 1 ORM / GSM8K-Qwen：61.36 → **82.00**
- Table 1 C-CoT 行：StrategyQA/LLaMA=**76.81**、GSM8K/Qwen=**93.88**（余 4 格 `--`）
- ⚠️ D-ProtoCoT / GSM8K-Qwen 仍是旧 **94.15**（与 ORM run 的 92 差 5 分，待 X 统一）

## 关键口径提醒（易错，勿犯）
- **Self-Certainty 脚本内部打印的 SC=73%（8B）不可用**：它重抠答案投票，与主表算法不同。论文 SC 一律用 `run.py` 主表值；Self-Certainty 用 89/97.5。
- **ORM=82.0，不是 81**（81 是 leakage full 的 D-ProtoCoT，不同实验）。
- **granularity 三数曾被贴串行**，修正后：step/path(主方法)=86.5 最高 > path/path 83 > step/step 82。
- **已否决**一份"81–83 是正常方差、随便填一个"的建议（填错列 + 没解决条 4）。

---

## 下一步优先级
1. **种子重跑**（Qwen 补 seed + LLaMA 全跑，命令在 reviewer2_progress.md ⑤）→ 定 X → 一次性填 #2 主表/granularity/leakage/ORM/Self-Certainty 五处
2. **混合题分析**（14B，`newrun/mixed_question_analysis.py`）→ 回填 #3 Q1 的 `XX%` + MIXED 子集表 → 进 tex
3. C-CoT 补剩 4 格（CSQA×2、GSM8K/LLaMA、SQA/Qwen）
4. #3 Q3 可复现性（交稿前执行 checklist）
