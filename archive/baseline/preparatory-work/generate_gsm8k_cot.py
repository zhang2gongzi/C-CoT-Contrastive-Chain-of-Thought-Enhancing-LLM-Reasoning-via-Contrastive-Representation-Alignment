# generate_qwen_gsm8k_cot.py

import os
import json
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def extract_ground_truth(answer_str):
    """从 GSM8K answer 中提取标准答案（兼容 #### 和 \\boxed{}）"""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    match = re.search(r"\\boxed\{(.+?)\}", answer_str)
    if match:
        return match.group(1).strip()
    return ""

def extract_pred_answer(cot_text):
    """从 Qwen 生成的 CoT 中提取预测答案"""
    # 1. 优先匹配 \boxed{...}
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", cot_text)
    if boxed_match:
        ans = boxed_match.group(1).strip()
        ans = re.sub(r"[^\d\-+\.]", "", ans)
        if ans:
            return ans

    # 2. 从后往前找含 "answer" 的行，并提取最后一个数字
    lines = cot_text.strip().split("\n")
    for line in reversed(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["answer", "therefore", "thus", "so "]):
            numbers = re.findall(r"-?\d+\.?\d*", line)
            if numbers:
                return numbers[-1]

    # 3. 全文最后一个数字
    all_numbers = re.findall(r"-?\d+\.?\d*", cot_text)
    if all_numbers:
        return all_numbers[-1]
    return ""

def main():
    # === 配置路径 ===
    model_path = "${MODEL_DIR}/Qwen2.5-7B-Instruct"
    data_path = "${PROJECT_ROOT}/database/gsm8k/train-00000-of-00001.parquet"
    output_dir = "${PROJECT_ROOT}/baseline/preparatory-work"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gsm8k_train_500_cot_paths.json")

    # === 1. 加载模型和 tokenizer (Qwen2.5) ===
    print("Loading Qwen2.5-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✅ Model loaded.")

    # === 2. 加载本地 GSM8K train 数据集 (Parquet) ===
    print(f"Loading GSM8K from {data_path}...")
    df = pd.read_parquet(data_path)
    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    
    num_samples = min(500, len(questions))
    print(f"Will process first {num_samples} samples from train set.")

    # === 3. 检查是否已有结果（支持续跑）===
    existing_data = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = {item["question"]: item for item in json.load(f)}
                print(f"Found existing file with {len(existing_data)} samples. Resuming...")
            except Exception as e:
                print(f"Failed to load existing file: {e}")

    results = []
    start_idx = 0
    if existing_data:
        for i in range(num_samples):
            q = questions[i]
            if q not in existing_data:
                start_idx = i
                break
        for i in range(start_idx):
            results.append(existing_data[questions[i]])

    # === 4. 生成 CoT 路径 ===
    for i in tqdm(range(start_idx, num_samples), desc="Generating CoT paths"):
        question = questions[i]
        gt_answer = extract_ground_truth(answers[i])

        paths = []
        for _ in range(10):  # N=10
            try:
                # Qwen2.5 的 chat format
                messages = [
                    {"role": "user", "content": f"Question: {question}\nLet's think step by step."}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                reasoning = response.strip()
                pred_ans = extract_pred_answer(reasoning)

                paths.append({
                    "reasoning": reasoning,
                    "predicted_answer": pred_ans
                })

            except Exception as e:
                print(f"\nError at sample {i}: {e}")
                paths.append({
                    "reasoning": "",
                    "predicted_answer": None
                })

        results.append({
            "question": question,
            "ground_truth": gt_answer,
            "paths": paths
        })

        # 实时保存
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Done! Saved to {output_file}")

if __name__ == "__main__":
    main()