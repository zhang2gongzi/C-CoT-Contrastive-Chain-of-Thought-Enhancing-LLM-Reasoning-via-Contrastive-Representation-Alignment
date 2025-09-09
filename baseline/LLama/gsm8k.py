import os
import re
import torch
import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # 进度条，方便查看推理进度

# ======================
# 1. 配置参数（根据你的环境修改）
# ======================
class Config:
    # LLaMA 模型路径（与你训练时一致）
    LLAMA_DIR = "/home2/zzl/model/Llama-2-7b-chat-hf"
    # GSM8K 测试集路径
    GSM8K_TEST_PATH = "/home2/zzl/C-CoT/database/gsm8k/test-00000-of-00001.parquet"
    # 输出结果保存路径
    OUTPUT_DIR = "/home2/zzl/C-CoT/baseline/LLama/llama_gsm8k_results"
    # 推理超参数
    MAX_NEW_TOKENS = 300  # 生成推理步骤的最大长度（GSM8K需足够长写步骤）
    TEMPERATURE = 0.7     # 控制生成多样性（0.7适合推理任务）
    TOP_P = 0.95          # 采样Top-p，避免生成离谱内容
    BATCH_SIZE = 2        # 批量推理（根据显存调整，7B模型建议1-4）
    SAMPLE_SIZE = 100     # 测试样本量（全量测试设为None，约1k样本）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
# 创建输出目录
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# ======================
# 2. 工具函数（答案提取+正确性判断）
# ======================
def extract_final_answer(text: str) -> str:
    """
    从 LLaMA 生成的推理文本中提取最终答案（GSM8K 答案多以数字结尾，且常含 "Answer:" 标记）
    :param text: LLaMA 生成的完整推理文本
    :return: 提取的最终答案（纯数字字符串，提取失败返回空）
    """
    # 模式1：匹配 "Answer: X" 或 "Answer is X"（常见于明确结尾）
    answer_pattern = r"(?i)answer\s*[:=]\s*(\d+)"  # (?i) 忽略大小写
    match = re.search(answer_pattern, text)
    if match:
        return match.group(1)
    
    # 模式2：若无明确标记，提取文本中最后一个连续数字（GSM8K 答案多在结尾）
    numbers = re.findall(r"\d+", text)
    if numbers:
        return numbers[-1]
    
    # 提取失败
    return ""

def is_answer_correct(pred_answer: str, gold_answer: str) -> bool:
    """
    比对预测答案与标准答案是否正确（基于数字匹配）
    :param pred_answer: 从 LLaMA 生成文本中提取的答案
    :param gold_answer: GSM8K 数据集中的标准答案
    :return: 正确返回True，错误/提取失败返回False
    """
    # 先从标准答案中提取数字（处理如 "The answer is 5." 这类格式）
    gold_num = extract_final_answer(gold_answer)
    if not gold_num or not pred_answer:
        return False
    # 数字完全匹配则正确
    return pred_answer == gold_num


# ======================
# 3. 加载 LLaMA 模型和分词器
# ======================
def load_llama_model():
    """加载 LLaMA 分词器和模型，启用高效推理配置"""
    print(f"Loading LLaMA model from {cfg.LLAMA_DIR}...")
    # 加载分词器（补充pad_token，LLaMA默认无pad_token）
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.LLAMA_DIR,
        trust_remote_code=True,
        padding_side="right"  # 右侧padding，避免生成时混乱
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 用eos_token作为pad_token
    
    # 加载模型（启用FP16节省显存，device_map="auto"自动分配设备）
    model = AutoModelForCausalLM.from_pretrained(
        cfg.LLAMA_DIR,
        torch_dtype=torch.float16,  # 7B模型FP16约占13GB显存
        device_map="auto",          # 自动分配CPU/GPU
        load_in_8bit=False,         # 若显存不足，设为True（需安装bitsandbytes）
        trust_remote_code=True
    )
    model.eval()  # 推理模式，禁用训练相关层（如dropout）
    print(f"LLaMA model loaded successfully (device: {cfg.DEVICE})")
    return tokenizer, model


# ======================
# 4. 批量推理函数（提高效率）
# ======================
def batch_generate_answers(
    questions: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM
) -> List[str]:
    """
    批量生成 LLaMA 的推理答案
    :param questions: 批量问题列表
    :param tokenizer: LLaMA 分词器
    :param model: LLaMA 模型
    :return: 批量推理文本（每个问题对应一个推理结果）
    """
    # 1. 构建提示词（引导 LLaMA 按 "问题→分步推理→答案" 格式生成）
    prompts = [
        f"Question: {q}\nPlease solve this math problem step by step, and clearly state the final answer at the end (e.g., 'Answer: X')."
        for q in questions
    ]
    
    # 2. 分词（批量处理，padding到最长序列）
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # 问题最大长度（GSM8K问题较短，512足够）
        padding=True,
        add_special_tokens=True
    ).to(cfg.DEVICE)
    
    # 3. 生成推理结果（禁用梯度计算，节省显存）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            do_sample=True,  # 采样生成（推理更灵活）
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3  # 避免重复生成（如连续重复步骤）
        )
    
    # 4. 解码生成结果（跳过特殊token）
    generated_texts = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # 5. 提取"生成的推理部分"（去掉原始prompt，只保留模型生成内容）
    final_results = []
    for prompt, gen_text in zip(prompts, generated_texts):
        # 只保留模型生成的部分（prompt之后的内容）
        if gen_text.startswith(prompt):
            gen_part = gen_text[len(prompt):].strip()
            final_results.append(f"{prompt}\n{gen_part}")  # 保留完整上下文，方便后续检查
        else:
            final_results.append(gen_text)  # 异常情况直接保留
    
    return final_results


# ======================
# 5. 核心评估流程
# ======================
def evaluate_llama_gsm8k():
    # 步骤1：加载模型和分词器
    tokenizer, model = load_llama_model()
    
    # 步骤2：读取并预处理GSM8K测试集
    print(f"\nLoading GSM8K test set from {cfg.GSM8K_TEST_PATH}...")
    df = pd.read_parquet(cfg.GSM8K_TEST_PATH)
    # 保留有效数据（必须有question和answer）
    df_valid = df[["question", "answer"]].dropna().reset_index(drop=True)
    # 采样测试样本（全量测试设为df_valid）
    if cfg.SAMPLE_SIZE and cfg.SAMPLE_SIZE < len(df_valid):
        df_test = df_valid.sample(n=cfg.SAMPLE_SIZE, random_state=42, replace=False)
    else:
        df_test = df_valid
    print(f"Test set loaded: {len(df_test)} samples")
    
    # 步骤3：批量推理+准确率统计
    correct_count = 0
    total_count = len(df_test)
    # 存储结果（用于后续分析）
    results = []
    
    # 按批次处理（tqdm显示进度）
    batch_num = (total_count + cfg.BATCH_SIZE - 1) // cfg.BATCH_SIZE  # 总批次数
    for batch_idx in tqdm(range(batch_num), desc="LLaMA Inference Progress"):
        # 取当前批次的问题和标准答案
        start = batch_idx * cfg.BATCH_SIZE
        end = min((batch_idx + 1) * cfg.BATCH_SIZE, total_count)
        batch_questions = df_test.iloc[start:end]["question"].tolist()
        batch_golds = df_test.iloc[start:end]["answer"].tolist()
        batch_indices = df_test.iloc[start:end].index.tolist()
        
        # 批量生成推理结果
        batch_gens = batch_generate_answers(batch_questions, tokenizer, model)
        
        # 逐个判断正确性
        for q, gold, gen, idx in zip(batch_questions, batch_golds, batch_gens, batch_indices):
            # 提取预测答案和标准答案
            pred_answer = extract_final_answer(gen)
            gold_answer = extract_final_answer(gold)
            # 判断是否正确
            is_correct = is_answer_correct(pred_answer, gold)
            if is_correct:
                correct_count += 1
            # 保存结果到列表
            results.append({
                "sample_idx": idx,
                "question": q,
                "gold_answer": gold,
                "gold_num": gold_answer,
                "llama_generation": gen,
                "pred_num": pred_answer,
                "is_correct": is_correct
            })
    
    # 步骤4：计算准确率并保存结果
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"\n=== LLaMA GSM8K Evaluation Result ===")
    print(f"Total Samples: {total_count}")
    print(f"Correct Samples: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 保存详细结果到Excel（方便后续分析错误案例）
    results_df = pd.DataFrame(results)
    results_save_path = os.path.join(cfg.OUTPUT_DIR, "llama_gsm8k_results.xlsx")
    results_df.to_excel(results_save_path, index=False)
    print(f"\nDetailed results saved to: {results_save_path}")
    
    # 保存汇总结果到文本文件
    summary_save_path = os.path.join(cfg.OUTPUT_DIR, "llama_gsm8k_summary.txt")
    with open(summary_save_path, "w", encoding="utf-8") as f:
        f.write(f"LLaMA GSM8K Evaluation Summary\n")
        f.write(f"Evaluation Time: {pd.Timestamp.now()}\n")
        f.write(f"LLaMA Model Path: {cfg.LLAMA_DIR}\n")
        f.write(f"Test Set Path: {cfg.GSM8K_TEST_PATH}\n")
        f.write(f"Total Samples: {total_count}\n")
        f.write(f"Correct Samples: {correct_count}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Inference Hyperparameters:\n")
        f.write(f"  Max New Tokens: {cfg.MAX_NEW_TOKENS}\n")
        f.write(f"  Temperature: {cfg.TEMPERATURE}\n")
        f.write(f"  Batch Size: {cfg.BATCH_SIZE}\n")
    print(f"Summary saved to: {summary_save_path}")


# ======================
# 6. 执行评估
# ======================
if __name__ == "__main__":
    evaluate_llama_gsm8k()