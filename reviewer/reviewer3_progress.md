# 审稿人 #3 回复进度

---

## 审稿意见概述

论文逻辑清晰，具有良好的创新性和应用价值，但需进一步修改：

---

## 问题 1：仅在 8B 参数规模的模型上评估
- **问题**：仅使用了 LLaMa3.1-8B 和 Qwen3-8B，两个模型参数量相同。建议在更大参数规模的语言模型上评估以证明方法的适用性
- **方案**：加一组 GSM8K 实验

### 模型选择分析

| 模型 | bf16 显存 | 4-bit 显存 | 24GB (4090/3090) |
|------|----------|-----------|-------------------|
| Qwen3-14B | ~28GB | ~8-10GB | ✅ 4-bit 可跑 |
| Qwen3-32B | ~64GB | ~16-20GB | ⚠️ 4-bit 勉强，推理时 KV cache 可能 OOM |

**结论**：选 **Qwen3-14B 4-bit 量化**。和 8B 同系列对比有意义，24GB 稳够跑。

### 操作步骤
1. 服务器下载模型：`huggingface-cli download Qwen/Qwen3-14B --local-dir /home2/zzl/model/Qwen3-14B`
2. 对 GSM8K 测试集 200 题生成 10 条 CoT，`load_in_4bit=True`
3. 用现有 GSM8K 训练数据 + 新生成的 14B 测试 CoT，跑 `run.py main`
4. 将 14B 结果加入 Table 1 新列

- **状态**：待下载模型，等当前 GSM8K 测试集 CoT 生成完

---

## 问题 2：基线方法较旧
- **问题**：基线方法发表于 2021 和 2023 年。建议对近三年的新方法进行更深入的分析和比较

### 已实现的新基线（`baseline/dprotocot/baselines.py`）

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
```

- **状态**：Self-Certainty-BERT 随时可跑；LLM 基线等 vLLM 部署

---

## 问题 3：可复现性
- **问题**：应公开相关代码和数据以确保可复现性
- **状态**：待处理

---

## 附录：审稿人 #3 原话

Reviewer #3: This paper proposed D-ProtoCoT, an inference-time framework for selecting high-quality reasoning paths based on representation-level alignment. The overall logic of the thesis is clear, and it has good innovativeness and application value. However, the current manuscript requires further revision, which mainly involves the following aspects.

(1) During the experiment, the authors only used two base LLMs, LLaMa3.1-8B and QWEN3-8B. These two models have the same parameter size. Therefore, it is recommended that the authors further evaluate the performance of the proposed method on larger language models with larger parameter sizes. This can demonstrate the applicability of the proposed method across different language models.

(2) The baseline models selected for this paper were published in 2021 and 2023. Therefore, the authors should conduct a more in-depth analysis of the new methods in the last three years and make comparisons to demonstrate the superiority of the proposed methods.

(3) In order to ensure the reproducibility of the proposed method, the authors should make the relevant code and data available for open-source use in their paper.
