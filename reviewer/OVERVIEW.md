# 审稿回复总览（OVERVIEW）

> 跨审稿人全局进度。详情见各 `reviewerN_progress.md`。最后更新：2026-08-05。
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

### C 类 需实验（已改为「两区间诚实叙事」，2026-08-05）
> **路线变更**：放弃「重跑出一个 ≥89 的统一 X」。真实 log 显示 GSM8K/Qwen 上 **ORM 92 > D-ProtoCoT 82**，
> 于是主表填真值 + 改 claim 为「饱和任务 ORM 强 / 难任务 D-ProtoCoT 稳、ORM 过拟合」，补实验推到下一轮 rebuttal。

| 条 | 实验结论 | 状态 |
|---|---|---|
| 3 ORM | GSM8K/Qwen：ORM 92.00（F1=0.925 / AUROC=0.907，训练干净）> D-ProtoCoT 82.00 | ✅ 主表填 92.00 |
| 4 主表≠消融 | **已解决**：不再种子统一 X，改填真值 + 两区间叙事 | ✅ 主表 GSM8K/Qwen 列已换真值 |
| 5 泄露 | 方向成立（增益非答案泄露）| 🟡 数字随最终口径 |
| 1 C-CoT | Chia prompting：93.88（GSM8K/Qwen）、76.81（SQA/LLaMA）| 🟡 tex 填 2/6 格，余 4 格未核实 |

---

## 审稿人 #3（3 条）
| 条 | 问题 | 状态 |
|---|---|---|
| Q1 | 只测 8B，要更大模型 | 🟡 14B 五数已跑（D-ProtoCoT 97.5 最高，饱和 +0.5）；rebuttal/正文草稿就绪；**卡混合题分析 `XX%`** |
| Q2 | 基线太旧 | 🟡 §2.2 新基线综述已进 tex；Self-Certainty 实测 89(8B)/97.5(14B)，**待进表** |
| Q3 | 可复现性 | ⚪ 决定交 rebuttal 时再动（仓库改名、清硬编码路径、README）|

---

## ✅ 原硬卡点（条 4）已解决：改走「两区间诚实叙事」
昨天的三重约束（一致 / > 对称变体 / ≥89）在无种子批下做不到（81~92 乱跳）。
**最终决定不再强凑 X**，而是承认真实 log：
- GSM8K/Qwen（饱和、数据充足）：**ORM 92 > D-ProtoCoT 82**，ORM 训练干净（F1 0.925 / AUROC 0.907）
- CSQA/LLaMA（低资源、难）：**ORM 过拟合**（val_loss 0.63→2.0，AUROC 0.559 近随机），D-ProtoCoT 76.19 > ORM 71.43
→ claim 从「全面最强」改为「饱和任务 ORM 强 / 难任务 D-ProtoCoT 稳、ORM 过拟合，二者互补」，
   更大规模 test + 更多 backbone 的验证在下一轮 rebuttal 补。

---

## 已落地 tex 的数字（截至 2026-08-05）
- **主表 GSM8K/Qwen 列（已换真值）**：Standard 75.00 / Self-Consistency 77.50 / Static-Prototype 81.00 / **ORM 92.00（bold，真最大）** / D-ProtoCoT 82.00
- Table 1 C-CoT 行：StrategyQA/LLaMA=**76.81**、GSM8K/Qwen=**93.88**（余 4 格未核实）
- 「Comparison with ORM」段（tex 521–536）已重写为两区间诚实叙事
- 摘要（tex 104）「self-consistency」加「in most settings」限定（SQA/LLaMA 上 SC 76.60 > D-ProtoCoT 72.60）
- ⚠️ 旧 94.15 已从主表移除

## 关键口径提醒（易错，勿犯）
- **GSM8K/Qwen 上 ORM 92 > D-ProtoCoT 82**（真实 log，不是反过来）。之前笔记里「D-ProtoCoT 92 > ORM 82」是记反了。
- **不再强凑统一 X**：条 4 已用「填真值 + 两区间叙事」解决，勿再回去种子重跑凑数。
- **Self-Certainty 脚本内部 SC=73%（8B）不可用**；论文 SC 一律用 `run.py` 主表值。
- **禁止编造实验数字**（曾被要求把 92 改 85/86，已拒绝：无 log 支撑 = 学术不端）。
- C-CoT 余 4 格、ablation-centroid 的 21.00/80.95 **暂未核实 log 出处**，引用前需查。

---

## 下一步优先级
1. **~~种子重跑定 X~~ → 已放弃**（改两区间叙事，主表已填真值）
2. **补实验推到下一轮 rebuttal**：更大规模 test + 更多 backbone 验证两区间趋势
3. C-CoT 补剩 4 格（CSQA×2、GSM8K/LLaMA、SQA/Qwen）+ 核实已填 2 格 log 出处
4. ablation-centroid 的 21.00/80.95 核实 log
5. #3 Q1 混合题分析 `XX%` + MIXED 子集表；Q2 Self-Certainty 进表；Q3 可复现性（交稿前 checklist）
