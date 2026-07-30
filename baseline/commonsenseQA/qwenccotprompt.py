"""
contrastive_cot_commonsenseqa.py

复现 Contrastive CoT 方法在 CommonsenseQA 数据集上
适配列: id, question, choices(dict with 'label' and 'text'), answerKey
"""

import json
import random
import re
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- 配置 ----------------
MODEL_PATH = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
DATA_PATH = "/home2/zzl/C-CoT/database/commonsenseQA/train-00000-of-00001.parquet"
OUTPUT_FILE = "contrastive_cot_commonsenseqa_results.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------

def load_data(path: str):
    df = pd.read_parquet(path)
    rows = df.to_dict(orient="records")
    return rows

def make_contrastive_demo(row):
    q = row["question"]
    choices = row["choices"]["text"]
    labels = row["choices"]["label"]
    gold = row["answerKey"]

    # 格式化选项
    choice_text = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))])

    # 正确/错误解释
    correct_exp = f"Based on commonsense reasoning, the best choice is {gold}."
    wrong_label = random.choice([l for l in labels if l != gold])
    incorrect_exp = f"By flawed reasoning, one might wrongly choose {wrong_label}."

    demo = (
        f"Question: {q}\nChoices:\n{choice_text}\n\n"
        f"Correct Explanation:\n{correct_exp}\nAnswer: {gold}\n\n"
        f"Incorrect Explanation:\n{incorrect_exp}\nAnswer: {wrong_label}\n\n"
        "----\n\n"
    )
    return demo

def build_prompt(demos, q, choices, labels):
    header = "Below are examples with Correct and Incorrect explanations. Learn from them.\n\n"
    demo_text = "".join(demos)
    choice_block = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))])
    query = (
        f"Question: {q}\nChoices:\n{choice_block}\n\n"
        "Please reason step by step and end with 'Answer: <LETTER>'."
    )
    return header + demo_text + query

def extract_letter(text, labels):
    m = re.search(r"Answer[:\s]*([A-E])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # fallback: 扫描选项字母
    for l in labels:
        if re.search(rf"\b{l}\b", text):
            return l
    return "Unknown"

def run_experiment(model_path, data_path, out_file, num_shots=4, max_test=50, temperature=0.0):
    # 加载数据
    data = load_data(data_path)
    print(f"[INFO] Loaded {len(data)} rows from {data_path}")

    # 加载模型
    print("[INFO] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    device = next(model.parameters()).device
    print(f"[INFO] Model loaded on {device}")

    random.shuffle(data)
    demos_src = data[:num_shots]
    tests = data[num_shots:num_shots+max_test]

    demos = [make_contrastive_demo(r) for r in demos_src]

    results = []
    correct = 0
    for i, row in enumerate(tests):
        q = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold = row["answerKey"]

        prompt = build_prompt(demos, q, choices, labels)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        pred = extract_letter(text, labels)
        ok = (pred == gold)
        if ok:
            correct += 1

        result = {
            "id": row["id"],
            "question": q,
            "choices": choices,
            "gold": gold,
            "pred_raw": text,
            "pred": pred,
            "correct": ok
        }
        results.append(result)

        print(f"[{i+1}/{len(tests)}] QID={row['id']} Gold={gold} Pred={pred} OK={ok}")

    acc = correct / len(tests)
    print(f"\n[SUMMARY] Accuracy = {acc*100:.2f}% ({correct}/{len(tests)})")

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--out_file", type=str, default=OUTPUT_FILE)
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--max_test", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    run_experiment(
        model_path=args.model_path,
        data_path=args.data_path,
        out_file=args.out_file,
        num_shots=args.num_shots,
        max_test=args.max_test,
        temperature=args.temperature,
    )
