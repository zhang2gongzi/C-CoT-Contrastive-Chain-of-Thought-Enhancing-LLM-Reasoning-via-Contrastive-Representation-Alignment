import os
import torch
import pandas as pd
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# -------------------------- 1. 路径配置 --------------------------
MODEL_PATH = "${MODEL_DIR}/Llama-2-7b-chat-hf"
TEST_DATA_PATH = "${PROJECT_ROOT}/database/commonsenseQA/train-00000-of-00001.parquet"
OUTPUT_DIR = "./llama_cot_results"
SAMPLE_SIZE = 200  # 保持你的样本量
ANSWER_COLUMN = "answerKey"


# -------------------------- 2. 工具函数（核心修改：适配NumPy数组格式的choices） --------------------------
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Llama需要手动设置pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def load_dataset(file_path, sample_size=None, answer_col="answerKey"):
    df = pd.read_parquet(file_path)
    required_cols = [answer_col, "choices", "question"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要字段：{col}，当前列：{df.columns.tolist()}")
    
    # 验证并转换choices格式（将NumPy数组转为普通列表）
    df["choices"] = df["choices"].apply(convert_choices_format)
    
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df


def convert_choices_format(choices):
    """将NumPy数组格式的choices转换为标准列表格式：[{"label": "A", "text": "xxx"}, ...]"""
    try:
        # 提取label和text数组（处理NumPy数组）
        labels = choices["label"]
        texts = choices["text"]
        
        # 转换为普通列表（确保不是NumPy数组）
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # 组合成字典列表
        return [{"label": labels[i], "text": texts[i]} for i in range(len(labels))]
    except Exception as e:
        raise ValueError(f"转换choices格式失败：{str(e)}，原始格式：{choices}")


def build_cot_prompt(question, choices):
    """构建提示词（使用转换后的choices格式）"""
    choices_str = ""
    for choice in choices:
        label = choice.get("label", "").upper()
        text = choice.get("text", "")
        if label and text:
            choices_str += f"{label}. {text}\n"
    
    if not choices_str:
        raise ValueError(f"无法解析choices格式：{choices}")
    
    # Llama对话模板
    prompt = f"""[INST] Answer the following question with step-by-step reasoning.
Finally, you MUST output the answer in the format "Final Answer: X" (X is A/B/C/D/E), no extra content.

Question: {question}
Options:
{choices_str.strip()}

Let's think step by step: [/INST]"""
    return prompt.strip()


def extract_final_answer(cot_response, choices):
    valid_labels = [choice.get("label", "").upper() for choice in choices if choice.get("label")]
    
    # 1. 匹配"Final Answer: X"
    lines = [line.strip() for line in cot_response.split("\n") if line.strip()]
    final_line = None
    for line in reversed(lines):
        if line.lower().startswith("final answer:"):
            final_line = line
            break
    
    if final_line:
        answer = final_line.split(":", 1)[1].strip().upper()
        if answer in valid_labels:
            return answer
    
    # 2. 备选匹配模式
    pattern = r"answer\s*(is|:\s*)\s*([A-E])"
    match = re.search(pattern, cot_response, re.IGNORECASE)
    if match and match.group(2).upper() in valid_labels:
        return match.group(2).upper()
    
    return "Unknown"


def calculate_accuracy(results):
    valid = [r for r in results if r["extracted_answer"] != "Unknown"]
    if not valid:
        return 0.0, 0, len(results)
    correct = sum(1 for r in valid if r["extracted_answer"] == r["ground_truth_label"])
    accuracy = round(correct / len(valid), 4)
    return accuracy, correct, len(valid)


# -------------------------- 3. 主流程 --------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== 1. 加载Llama模型与分词器 ===")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    print("\n=== 2. 加载CommonsenseQA训练集 ===")
    df = load_dataset(TEST_DATA_PATH, sample_size=SAMPLE_SIZE)
    print(f"加载完成：{len(df)} 个样本（已转换choices格式）\n")
    
    results = []
    print("=== 3. 生成CoT推理并计算准确度 ===")
    for idx in range(len(df)):
        row = df.iloc[idx]
        print(f"\n【样本 {idx+1}/{len(df)}】ID: {row['id']}")
        
        # 获取标准答案
        ground_truth_label = row[ANSWER_COLUMN].upper() if pd.notna(row[ANSWER_COLUMN]) else None
        if not ground_truth_label:
            print(f"⚠️  该样本{ANSWER_COLUMN}为空，跳过")
            continue
        
        # 构建提示词
        try:
            prompt = build_cot_prompt(row["question"], row["choices"])
        except ValueError as e:
            print(f"⚠️  解析choices失败：{e}，跳过该样本")
            continue
        
        # 生成推理
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        
        # 解码响应
        prompt_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])
        cot_response = tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        ).strip()
        
        # 提取答案
        extracted_ans = extract_final_answer(cot_response, row["choices"])
        print(f"标准答案标签：{ground_truth_label}")
        print(f"提取答案：{extracted_ans}")
        print(f"推理预览：{cot_response[:80]}...")
        
        # 保存结果
        results.append({
            "sample_idx": idx,
            "data_id": row["id"],
            "question_concept": row.get("question_concept", ""),
            "question": row["question"],
            "choices": row["choices"],
            "ground_truth_label": ground_truth_label,
            "cot_reasoning": cot_response,
            "extracted_answer": extracted_ans,
            "is_correct": extracted_ans == ground_truth_label
        })
    
    # 统计准确度
    print(f"\n=== 4. 准确度统计 ===")
    accuracy, correct, valid = calculate_accuracy(results)
    print(f"总处理样本数：{len(results)}")
    print(f"有效样本（成功提取答案）：{valid}")
    print(f"正确样本数：{correct}")
    print(f"最终准确度：{accuracy}（{correct}/{valid}）")
    
    # 保存结果
    save_path = f"{OUTPUT_DIR}/llama_cot_accuracy_results.jsonl"
    pd.DataFrame(results).to_json(save_path, orient="records", lines=True, force_ascii=False)
    print(f"\n结果保存至：{save_path}")


if __name__ == "__main__":
    main()
    