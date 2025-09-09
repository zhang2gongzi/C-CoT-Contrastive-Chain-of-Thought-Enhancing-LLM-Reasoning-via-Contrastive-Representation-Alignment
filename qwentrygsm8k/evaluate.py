import os
import re
import torch
import pandas as pd
from typing import List, Tuple
# 关键：补充 transformers 库的必要导入
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# 1. 从 main.py 复用配置和工具函数（确保 main.py 中这些函数/类可正常导出）
from main import (
    Config,          # 配置类
    is_correct,      # 正确性判断函数
    generate_cot_paths  # CoT 生成函数（从 main.py 直接复用，避免重复定义）
)

# 2. 初始化配置
cfg = Config()

# 3. 加载所有模型和分词器（补充完整导入逻辑，避免依赖缺失）
# 3.1 加载 LLaMA 分词器和模型（生成 CoT 需用到）
llama_tokenizer = AutoTokenizer.from_pretrained(cfg.LLAMA_DIR, trust_remote_code=True)
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(
    cfg.LLAMA_DIR, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
llama_model.eval()  # 评估阶段，LLaMA 仅用于生成，不训练

# 3.2 加载 BERT 分词器和训练好的编码器（核心评估模型）
bert_tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_MODEL)
bert_encoder = AutoModel.from_pretrained(cfg.BERT_MODEL).to(cfg.device)
# 加载训练好的 BERT 权重
bert_encoder.load_state_dict(
    torch.load(os.path.join(cfg.OUTPUT_DIR, "c_cot_bert.pt"), map_location=cfg.device)
)
bert_encoder.eval()  # 切换到评估模式，禁用梯度计算


# 4. 复用并确认 CoT 生成函数（若 main.py 中该函数依赖局部变量，需确保参数完整）
# （注：若 main.py 中 generate_cot_paths 已定义，此处可跳过；若有依赖冲突，可重新定义如下）
# def generate_cot_paths(question: str, num_paths: int = 5) -> List[str]:
#     prompt = f"Question: {question}\nLet's reason step by step."
#     inputs = llama_tokenizer(prompt, return_tensors="pt").to(cfg.device)
#     paths = []
#     with torch.no_grad():  # 评估阶段不计算梯度，节省显存
#         for _ in range(num_paths):
#             output = llama_model.generate(
#                 **inputs,
#                 max_new_tokens=200,
#                 temperature=0.7,
#                 do_sample=True,
#                 pad_token_id=llama_tokenizer.pad_token_id
#             )
#             text = llama_tokenizer.decode(output[0], skip_special_tokens=True)
#             paths.append(text)
#     return paths


# 5. 提取序列级表征（评估核心：用训练好的 BERT 编码 CoT 路径）
def get_sequence_representation(text: str) -> torch.Tensor:
    with torch.no_grad():  # 评估阶段禁用梯度，避免显存占用
        tokens = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_len,
            padding="max_length"  # 补充 padding，避免 batch 维度问题
        ).to(cfg.device)
        outputs = bert_encoder(**tokens)
        # 序列级表征：取最后一层隐藏态的均值（[1, H]，H 为 BERT 隐藏层维度）
        return outputs.last_hidden_state.mean(dim=1)


# 6. 选择最佳 CoT 路径（基于 BERT 表征的相似度）
def select_best_path(question: str, num_paths: int = None) -> str:
    """
    生成多个 CoT 路径，通过表征相似度选择最佳路径
    :param question: 测试问题
    :param num_paths: 生成路径数量（默认用配置中的 num_paths）
    :return: 最佳路径文本
    """
    num_paths = num_paths if num_paths is not None else cfg.num_paths
    # 生成多路径 CoT
    paths = generate_cot_paths(question, num_paths)
    if not paths:  # 极端情况：无路径生成，返回空
        return ""
    
    # 提取所有路径的表征
    path_reprs = [get_sequence_representation(path) for path in paths]
    
    # 计算每个路径与其他所有路径的平均余弦相似度（相似度高的视为更可能正确）
    similarities = []
    for i in range(len(path_reprs)):
        sim_sum = 0.0
        count = 0
        for j in range(len(path_reprs)):
            if i != j:  # 排除自身
                sim = torch.nn.functional.cosine_similarity(
                    path_reprs[i], path_reprs[j], dim=-1
                ).item()
                sim_sum += sim
                count += 1
        # 平均相似度（避免除零）
        avg_sim = sim_sum / count if count > 0 else 0.0
        similarities.append(avg_sim)
    
    # 选择相似度最高的路径作为最佳路径
    best_idx = similarities.index(max(similarities))
    return paths[best_idx]


# 7. 核心评估函数：计算模型准确率
def evaluate_accuracy(test_dataset: pd.DataFrame, sample_size: int = 100) -> float:
    """
    评估模型在测试集上的准确率
    :param test_dataset: 测试数据集（含 "question" 和 "answer" 列）
    :param sample_size: 采样数量（全量测试较慢，建议先小样本验证）
    :return: 准确率（正确路径数 / 总测试数）
    """
    correct_count = 0
    total_count = min(sample_size, len(test_dataset))  # 避免采样数超过数据集大小
    
    # 随机采样测试数据（固定 random_state 保证可复现）
    test_sample = test_dataset.sample(
        n=total_count, 
        random_state=42, 
        replace=False  # 不重复采样
    ).reset_index(drop=True)
    
    # 逐样本评估
    for idx, row in test_sample.iterrows():
        question = row["question"]
        gold_answer = row["answer"]
        
        # 1. 选择最佳路径
        best_path = select_best_path(question, cfg.num_paths)
        # 2. 判断路径是否正确（复用 main.py 的 is_correct 函数）
        is_right = is_correct(best_path, gold_answer)
        
        if is_right:
            correct_count += 1
        
        # 打印进度（每 10 个样本更新一次）
        if (idx + 1) % 10 == 0 or (idx + 1) == total_count:
            current_acc = correct_count / (idx + 1)
            print(f"Processed {idx+1}/{total_count} | Current Accuracy: {current_acc:.4f}")
    
    # 计算最终准确率
    final_acc = correct_count / total_count if total_count > 0 else 0.0
    return final_acc


# 8. 执行评估（主函数）
if __name__ == "__main__":
    # 加载测试数据（若有独立测试集，需替换为测试集路径）
    print(f"Loading test data from {cfg.GSM8K_PARQUET_PATH}...")
    test_df = pd.read_parquet(cfg.GSM8K_PARQUET_PATH)
    # 过滤无效数据（确保有 question 和 answer）
    test_dataset = test_df[["question", "answer"]].dropna().reset_index(drop=True)
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # 开始评估（可调整 sample_size，如全量测试设为 len(test_dataset)）
    print("\nStarting evaluation...")
    final_accuracy = evaluate_accuracy(test_dataset, sample_size=200)
    
    # 打印并保存结果
    result_str = f"\nFinal Evaluation Result:\n" \
                 f"Sample Size: {200}\n" \
                 f"Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)"
    print(result_str)
    
    # 保存结果到文件（避免丢失）
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(cfg.OUTPUT_DIR, "evaluation_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Time: {pd.Timestamp.now()}\n")
        f.write(result_str)
    print(f"\nResult saved to {result_path}")