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

- **状态**：命令就绪，待 AutoDL 上跑（bf16，10 epoch）

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

# Self-Certainty (Kang et al., NeurIPS 2025) —— 2024+ 新基线，AutoDL 上跑（需 GPU 加载 LLM）
python newrun/self_certainty.py --model_path /root/autodl-tmp/Qwen3-8B \
    --data_path newrundata/gsm8k_test_flat.jsonl --num_paths 10
```

**Self-Certainty (Kang) 预期输出**：终端 `Self-Certainty: XX.XX%` + `Self-Consistency: XX.XX%`，并写出 `self_certainty_results.json`（逐题明细）。用于回应"基线太旧"，加进对比表。

- **状态**：Self-Certainty-BERT 随时可跑；Self-Certainty(Kang) 待 AutoDL 跑；LLM 基线等 vLLM 部署

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
