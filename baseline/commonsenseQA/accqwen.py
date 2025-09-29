import os
import torch
import pandas as pd
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

# -------------------------- 1. 核心路径配置（请核对！） --------------------------
MODEL_PATH = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"  # 你的本地Qwen模型路径
TEST_DATA_PATH = "/home2/zzl/C-CoT/database/commonsenseQA/train-00000-of-00001.parquet"  # 测试集路径
OUTPUT_DIR = "/home2/zzl/C-CoT/baseline/commonsenseQA/cot_results"  # 结果保存目录
SAMPLE_SIZE = 200  # 先测试10个样本，全量运行改为 None
ANSWER_COLUMN = "answerKey"  # 你的标准答案字段，无需修改


# -------------------------- 2. 工具函数（无需修改） --------------------------
def load_model_and_tokenizer(model_path):
    """加载Qwen模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_dataset(file_path, sample_size=None, answer_col="answerKey"):
    """加载数据集，验证标准答案字段"""
    df = pd.read_parquet(file_path)
    if answer_col not in df.columns:
        raise ValueError(f"数据集缺少 {answer_col} 字段！当前列：{df.columns.tolist()}")
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df


def build_cot_prompt(question, choices):
    """构建CoT提示词（强制Final Answer格式，确保答案可提取）"""
    choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    prompt = f"""Answer the following question with step-by-step reasoning.
Finally, you MUST output the answer in the format "Final Answer: X" (X is A/B/C/D/E), no extra content.

Question: {question}
Options:
{choices_str}

Let's think step by step:
"""
    return prompt.strip()


def extract_final_answer(cot_response, choices):
    """从CoT推理中提取最终答案（适配answerKey为字母的格式）"""
    # 1. 优先匹配 "Final Answer: X" 格式
    lines = [line.strip() for line in cot_response.split("\n") if line.strip()]
    final_line = None
    for line in reversed(lines):
        if line.lower().startswith("final answer:"):
            final_line = line
            break
    
    if final_line:
        answer = final_line.split(":", 1)[1].strip().upper()
        if answer in [chr(65+i) for i in range(len(choices))]:
            return answer
    
    # 2. 备选：匹配 "answer is X" "answer: X" 等模式
    pattern = r"answer\s*(is|:\s*)\s*([A-E])"
    match = re.search(pattern, cot_response, re.IGNORECASE)
    if match:
        return match.group(2).upper()
    
    # 3. 无法提取标记
    return "Unknown"


def calculate_accuracy(results, answer_col="answerKey"):
    """计算准确度（过滤无法提取答案的样本）"""
    valid = [r for r in results if r["extracted_answer"] != "Unknown"]
    if not valid:
        return 0.0, 0, len(results)
    correct = sum(1 for r in valid if r["extracted_answer"] == r["ground_truth_answer"])
    accuracy = round(correct / len(valid), 4)
    return accuracy, correct, len(valid)


# -------------------------- 3. 主流程（无需修改） --------------------------
def main():
    # 初始化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== 1. 加载Qwen模型与分词器 ===")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    print("\n=== 2. 加载CommonsenseQA测试集 ===")
    df = load_dataset(TEST_DATA_PATH, sample_size=SAMPLE_SIZE, answer_col=ANSWER_COLUMN)
    print(f"加载完成：{len(df)} 个样本 | 标准答案字段：{ANSWER_COLUMN}")
    
    # 处理样本
    results = []
    print(f"\n=== 3. 生成CoT推理并计算准确度 ===")
    for idx in range(len(df)):
        row = df.iloc[idx]
        print(f"\n【样本 {idx+1}/{len(df)}】ID: {row['id']}")
        
        # 构建提示
        prompt = build_cot_prompt(row["question"], row["choices"])
        
        # 生成推理
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        cot_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cot_reasoning = cot_full[len(prompt):].strip()
        
        # 提取答案与对比
        extracted_ans = extract_final_answer(cot_reasoning, row["choices"])
        ground_truth = row[ANSWER_COLUMN]
        print(f"推理预览：{cot_reasoning[:80]}...")
        print(f"提取答案：{extracted_ans} | 标准答案：{ground_truth}")
        
        # 保存结果
        results.append({
            "sample_idx": idx,
            "data_id": row["id"],
            "question_concept": row["question_concept"],  # 保留概念字段
            "question": row["question"],
            "choices": row["choices"],
            "ground_truth_answer": ground_truth,
            "cot_reasoning": cot_reasoning,
            "extracted_answer": extracted_ans,
            "is_correct": extracted_ans == ground_truth  # 标记是否正确
        })
    
    # 统计准确度
    accuracy, correct, valid = calculate_accuracy(results)
    print(f"\n=== 4. 准确度统计 ===")
    print(f"总样本数：{len(results)}")
    print(f"有效样本（成功提取答案）：{valid}")
    print(f"正确样本数：{correct}")
    print(f"最终准确度：{accuracy}（{correct}/{valid}）")
    
    # 保存完整结果
    results_df = pd.DataFrame(results)
    save_path = f"{OUTPUT_DIR}/qwen_cot_accuracy_results.jsonl"
    results_df.to_json(save_path, orient="records", lines=True, force_ascii=False)
    print(f"\n完整结果已保存至：{save_path}")


if __name__ == "__main__":
    main()