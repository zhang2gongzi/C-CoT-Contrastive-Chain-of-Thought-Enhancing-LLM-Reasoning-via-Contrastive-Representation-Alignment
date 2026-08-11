# 审稿回复总览（OVERVIEW）

> 跨审稿人全局进度。详情见各 `reviewerN_progress.md`。最后更新：2026-08-09（L-GSM8K 官方 test 换列 + GPU 口径改 + Q-CSQA 10-epoch 换列完成 + line412 重写；现 6 列+14B 全 D≥SC）。
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
| 4 主表≠消融 | **已解决**：(a) 主表填真值 + 两区间叙事；(b) granularity 消融表 2026-08-07 换 GSM8K 真值（78.50/84.00/84.50），弃旧 StrategyQA 版（64.29 与主表矛盾）| ✅ 完全闭环 |
| 5 泄露 | 方向成立（增益非答案泄露）| 🟡 数字随最终口径 |
| 1 C-CoT | Chia prompting：Q-GSM8K 92.35 / Q-CSQA 78.63 / Q-SQA 90.22 / L-SQA 76.81 / L-CSQA 70.50 / **L-GSM8K 78.84（新，2026-08-07，328/416 已核）** | 🟡 tex 填 6/7 格，余 1 格 `--`（14B/GSM8K）|

### D 类 2026-08-09 进行中（今晚出小版本）
| 项 | 状态 |
|---|---|
| L-GSM8K 官方 test 换列 | ✅ 已进 tex（71.50/71.50/68.50/68.50/**80.00**，n=200，整除核验过）；摘要+line409 的 +17.8→+8.5，line415「Static vs Dynamic」段已翻转重写（Static 87 神话破） |
| GPU 硬件口径 | ✅ tex 382 改「8B on RTX 3090；14B on RTX 5090」（旧「all on RTX 4090」两处不实已纠） |
| Q-CSQA 重跑 | ✅ 已进 tex：10-epoch orm（test n=50，整除全过 /50）真值 Standard 62.00 / SC 62.00 / Static 62.00 / ORM 68.00 / **D-ProtoCoT 70.00（bold，+8.00 over SC）**；C-CoT 78.63 未动。⚠️弃 3-epoch 版（D=80.00，欠训不用）；contrastive val_loss 升（4.627→4.769）→ 增益来自选择机制非对比训练，勿过度 claim |
| ⚠️ line 412「Diminishing Returns」段 | ✅ 已重写：Qwen3-8B 三任务 D-ProtoCoT 全赢（Q-CSQA +8.00 / Q-GSM8K +4.50 / Q-SQA +3.57）；Q-SQA +3.57=1 题（n=28）写作 parity；真正 diminishing return 落到 14B 饱和（+0.5） |
| ⚠️「in most settings」限定词（67/104/406） | 🟡 **暂不动**：现所有 6 列+14B 均 D-ProtoCoT ≥ SC（可升级为 consistently），但 Q-SQA(+3.57≈1 题)/14B(+0.5) 偏薄 → 保守留 "in most settings"，待导师定 claim 强度 |
| dataset-details 表 L-GSM8K 行对齐 | ✅ 已改：line 608 `GSM8K/LLaMA 378/42/200 Official test`（旧 336/42/42 Grouped 8:1:1，与主表官方 test 换列不符已纠）；caption「For GSM8K/Qwen」→「For both GSM8K settings」 |

---

## 审稿人 #3（3 条）
| 条 | 问题 | 状态 |
|---|---|---|
| Q1 | 只测 8B，要更大模型 | 🟡 14B 七数已跑（Static 97.00 / ORM 96.50 / D-ProtoCoT 97.5 最高，饱和 +0.5，2026-08-11 补 Static/ORM 两格）；rebuttal/正文草稿就绪；**卡混合题分析 `XX%`** |
| Q2 | 基线太旧 | 🟡 §2.2 新基线综述已进 tex；✅ Self-Certainty(Kang,logprob) 8B/GSM8K **实测 77.00**（D-ProtoCoT 82 +5.0）已进 tex Experiments；Self-Certainty-BERT 81.00（打平，不写超过）|
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

## 已落地 tex 的数字（截至 2026-08-07）
- **主表 GSM8K/Qwen 列（已换真值）**：Standard 75.00 / Self-Consistency 77.50 / Static-Prototype 81.00 / **ORM 92.00（bold，真最大）** / D-ProtoCoT 82.00
- **主表 StrategyQA/LLaMA 列（已换重标真值，2026-08-07）**：Standard 45.24 / Self-Consistency 61.90 / Static-Prototype 52.38 / ORM 54.76 / **D-ProtoCoT 66.67（bold，仍最高，+4.77 over SC）**；C-CoT 76.81 未动（独立 prompting，非同口径）。
  - ⚠️ **旧 86.20 系坏标签产物，已弃**：旧抽取器漏 `$\boxed{yes/no}$`，per-path 被误判为 28.5%（真实 46.8%），旧列 Standard 68.60 / SC 62.60 / D-ProtoCoT 86.20（+23.6）全不可信。重标（`fix_labels.py --write`）+ `run.py orm` 重跑后为上方真值。摘要旗舰从「SQA +23.6」改为「GSM8K/LLaMA +8.5」（⚠️ 2026-08-09 再降：旧 +17.8 来自 test=42 小切分，已弃；见下条 L-GSM8K 官方 test 重跑）。
- **主表 StrategyQA/Qwen 列（已重跑换真值，2026-08-07，test n=28）**：Standard 60.71 / Self-Consistency 67.86 / Static-Prototype 64.29 / ORM 60.71 / **D-ProtoCoT 71.43（新 bold，全列最高，20/28）**；C-CoT 90.22 未动。旧值（Standard 66.80 / SC 76.60(bold) / Static 62.70 / ORM 50.43 / D-ProtoCoT 72.60）已弃。⚠️ n=28 差 1 题（D 20/28 vs SC 19/28），tex 不写幅度。此列由 SC 赢转为 D-ProtoCoT 赢。
- **主表 GSM8K/LLaMA 列（已换官方 test 真值，2026-08-09，test n=200）**：Standard 71.50 / Self-Consistency 71.50 / Static-Prototype 68.50 / ORM 68.50 / **D-ProtoCoT 80.00（bold，选择类最高，+8.50 over SC）**；C-CoT 78.84 未动。整除核验全过（/200）。ORM 诊断 loss=2.49 / path_acc=0.65 / F1=0.77 / **AUROC=0.5336（近随机）**→坐实难任务 ORM 退化。旧列（Standard 43.80 / SC 63.20 / Static **87.00(bold)** / ORM 78.20 / D-ProtoCoT 80.95，来自 test=42 小切分）已弃。⚠️ **Static 87.00 神话已破**：新 Static 68.50 < Standard 71.50 < D-ProtoCoT 80.00 → tex 415「Static vs Dynamic」段已翻转重写为「D-ProtoCoT 在所有数据集 ≥ Static，消融跌破基线」；摘要+line409 的 +17.75/+17.8 → +8.50。
- **主表 CSQA/Qwen 列（已换 10-epoch 真值，2026-08-09，test n=50）**：Standard 62.00 / Self-Consistency 62.00 / Static-Prototype 62.00 / ORM 68.00 / **D-ProtoCoT 70.00（bold，全列最高，35/50，+8.00 over SC）**；C-CoT 78.63 未动。整除核验全过（/50：62=31、68=34、70=35）。⚠️ **弃 3-epoch 版**（D=80.00，欠训不用，与其余列 10-epoch 口径不符会再触发 R2 条4）；contrastive val_loss 升（4.627→4.769）encoder 未收敛 → 增益来自选择机制非对比训练，正文勿过度 claim「对比训练贡献 +X」。ORM test AUROC=0.7933（健康）。旧列（Standard 75.60 / SC 86.98 / Static 68.20 / ORM 77.27 / D-ProtoCoT 87.71）已弃。
- Table 1 C-CoT 行（真实 tex 现状）：L-CSQA=**70.50** / L-GSM8K=**78.84**（328/416，2026-08-07 实跑已核） / L-SQA=**76.81** / Q-CSQA=**78.63**（390/496，2026-08-06 实跑） / Q-GSM8K=**92.35** / Q-SQA=**90.22** / 14B=`--`（已填 6/7，余 1 格）
- **ablation-centroid 表已删**（21.00 与主表 Static-Prototype 87.00 自相矛盾）；「Why Naive Centroid Fails」正文改引主表 Static-Prototype（L-SQA 52.38 vs D-ProtoCoT 66.67）。
- **granularity 消融表已换 GSM8K（2026-08-07，闭 R2 条4）**：旧 StrategyQA 版（proposed 64.29，与主表 86.20/72.60 矛盾）弃用；新 GSM8K/Qwen3-8B 真值 Path/Path=78.50 / Step/Step=84.00 / **Step/Path(proposed)=84.50（bold）**。caption+正文（tex 485/487/491）同步换 GSM8K，叙事改为「训练粒度是主因（step 训练 +5.5），选择粒度锦上添花（+0.5）」。
- Q2 小表 `tab:q2-selectors`（GSM8K/Qwen3-8B）：SC 77.50 / Self-Certainty(Kang) 77.00 / Self-Certainty-BERT 81.00（待核）/ **D-ProtoCoT 82.00**。
- 「Comparison with ORM」段（tex 521–536）已重写为两区间诚实叙事
- 摘要（tex 104）「self-consistency」加「in most settings」限定 —— ⚠️ **2026-08-09 更新**：所有 6 列+14B 现均 D-ProtoCoT ≥ SC（L-CSQA +4.76 / L-GSM8K +8.50 / L-SQA +4.77 / Q-CSQA +8.00 / Q-GSM8K +4.50 / Q-SQA +3.57 / 14B +0.5）。可升级为 "consistently"，但 Q-SQA(+3.57≈1 题,n=28)/14B(+0.5) 偏薄 → **保守留 "in most settings"，待导师定 claim 强度**。tex 67/104/406 三处一致，暂不动。
- ⚠️ 旧 94.15 已从主表移除
- **token 长度表（tex 623–636）已换真值（2026-08-07，`newrun/token_stats.py` BERT 分词无截断）**：旧 3 行占位（GSM8K 809.1/93.1%、CSQA 856.7/91.36%、SQA 790.2/71.5%、Paths 全 10,000）**系编造，复现不出，已弃**。新 6 行（数据集×backbone）真值：GSM8K/Qwen 9110/605.3/466/1077/**42.21%**、GSM8K/LLaMA 4200/227.1/211/984/1.33%、CSQA/Qwen 5000/520.7/522/584/**80.80%**、CSQA/LLaMA 4200/222.9/224/355/0.00%、SQA/Qwen 2800/890.2/949/1230/**96.64%**、SQA/LLaMA 4200/340.3/334/1154/5.31%。GSM8K/Qwen 用 merged（不掺被截短的 test 行）。tex 376/621 的「93% 超限」叙述同步改为分 backbone 诚实版（Qwen 长任务大量超限、LLaMA 短基本不超）。**条3「旧 ORM 截断」论据不在 tex 正文，无遗留**。

## 数据清洗进度（R2 条4/6，详见 reviewer2_progress ④i）
- **Phase 1 已完成（2026-08-05）**：新增 `newrun/fix_labels.py`（统一抽取器，report-first，`--write` 生成 `.bak`）。
  - GSM8K ×2：**不重标**，旧标签已对（Qwen 旧 62.4%≈cue 63.2%；LLaMA 旧 87%≈last-number 89%）。
  - StrategyQA ×2：✅ **已就地重标**——strategyqa_flat(Qwen) 55.9%→56.7%；strategyqa_llama **28.5%→46.8%**（旧抽取器漏 `$\boxed{yes/no}`，抽查 6/6 干净）。
  - CSQA ×2：❌ 正则救不了（~27% waffling 不 commit）→ 留 Phase 2 重生成；本地 csqa_llama_flat 是坏文件（0% 正例）。
- **Phase 2 待跑（GPU 端）**：按混合协议一次性重跑 6 格 main+orm + 补 C-CoT LLaMA 两格 + 补截断重生成（SQA no-answer ~28-36%、CSQA 全部）。

## 关键口径提醒（易错，勿犯）
- **GSM8K/Qwen 上 ORM 92 > D-ProtoCoT 82**（真实 log，不是反过来）。之前笔记里「D-ProtoCoT 92 > ORM 82」是记反了。
- **不再强凑统一 X**：条 4 已用「填真值 + 两区间叙事」解决，勿再回去种子重跑凑数。
- **Self-Certainty 脚本内部 SC=73%（8B）不可用**；论文 SC 一律用 `run.py` 主表值。
- **禁止编造实验数字**（曾被要求把 92 改 85/86，已拒绝：无 log 支撑 = 学术不端）。
- C-CoT 余 1 格（14B/GSM8K）、ablation-centroid 的 21.00/80.95 **暂未核实 log 出处**，引用前需查。

---

## 下一步优先级
1. **~~种子重跑定 X~~ → 已放弃**（改两区间叙事，主表已填真值）
2. **~~Phase 1 修数据~~ → 已完成**（StrategyQA 重标 + GSM8K 确认无需改，见上）
3. **Phase 2（GPU 端）**：混合协议一次性重跑 6 格 main+orm + 补 C-CoT LLaMA 两格 + 补截断重生成（CSQA 必须、SQA no-answer 部分）
4. **补实验推到下一轮 rebuttal**：更大规模 test + 更多 backbone 验证两区间趋势
5. C-CoT 补剩 2 格（GSM8K/LLaMA + 14B/GSM8K）+ 核实已填格 log 出处
6. ablation-centroid 的 21.00/80.95 核实 log
7. #3 Q1 混合题分析 `XX%` + MIXED 子集表；Q2 Self-Certainty 进表；Q3 可复现性（交稿前 checklist）
