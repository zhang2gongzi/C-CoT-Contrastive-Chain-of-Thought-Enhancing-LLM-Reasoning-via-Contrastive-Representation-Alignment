import json
import random
import re
from typing import List, Dict
import torch
import pandas as pd  # 用于读取Parquet文件
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------- 配置 -------------
DATASET_PATH = "/home2/zzl/C-CoT/database/gsm8k/test-00000-of-00001.parquet"
MODEL_PATH = "/home2/zzl/model/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "/home2/zzl/C-CoT/baseline/ccotPrompting/gsm8k_contrastive_cot_results.jsonl"
# ----------------------------

# ---------- 数据 ----------
def load_parquet(path: str) -> List[Dict]:
    """读取Parquet格式的GSM8K数据集"""
    df = pd.read_parquet(path)
    # 转换为字典列表并确保必要字段存在
    data = []
    for idx, row in df.iterrows():
        # 修复：修正了变量赋值和缩进错误
        final_answer = extract_numeric_answer(row["answer"])
        data.append({
            "id": str(idx),  # 使用索引作为ID
            "question": row["question"],
            "answer": row["answer"],
            "label": final_answer
        })
    return data

def extract_numeric_answer(answer: str) -> str:
    """从GSM8K的答案中提取最终数值"""
    # GSM8K答案格式通常以####结尾，如"#### 42"
    match = re.search(r"####\s*(\d+)", answer)
    if match:
        return match.group(1)
    # 如果没有####标记，尝试提取最后出现的数字
    numbers = re.findall(r"\b\d+\b", answer)
    return numbers[-1] if numbers else ""

def make_contrastive_demo(example: Dict) -> str:
    """构造适用于数学推理的对比示例"""
    question = example["question"]
    correct_answer = example["label"]
    
    # 生成一个错误答案（这里简单处理为正确答案±随机数）
    try:
        correct_num = int(correct_answer)
        # 生成一个与正确答案不同的错误答案
        error_offset = random.randint(1, 5) if correct_num > 5 else random.randint(1, correct_num)
        wrong_num = correct_num + error_offset if random.random() > 0.5 else correct_num - error_offset
        wrong_answer = str(wrong_num)
    except:
        # 如果转换失败，使用固定错误答案
        wrong_answer = "0" if correct_answer != "0" else "1"

    # 正确推理链
    correct_exp = f"To solve the problem '{question}', we can calculate step by step:\n"
    correct_exp += f"1. Analyze the problem and identify the required operations\n"
    correct_exp += f"2. Perform the calculations step by step\n"
    correct_exp += f"3. The final result is {correct_answer}\n"

    # 错误推理链（模拟常见计算错误）
    incorrect_exp = f"To solve the problem '{question}', here is an incorrect approach:\n"
    incorrect_exp += f"1. Misunderstand the problem requirements\n"
    incorrect_exp += f"2. Make a mistake in the calculation steps\n"
    incorrect_exp += f"3. The wrong result is {wrong_answer}\n"

    demo = (
        f"Question: {question}\n\n"
        f"Correct Explanation:\n{correct_exp}Answer: {correct_answer}\n\n"
        f"Incorrect Explanation:\n{incorrect_exp}Answer: {wrong_answer}\n\n"
        "----\n\n"
    )
    return demo

def build_prompt(demos: List[str], query_q: str) -> str:
    """构建适用于GSM8K的提示词"""
    header = "Here are examples with Correct and Incorrect mathematical reasoning. Learn from them.\n\n"
    demo_text = "".join(demos)
    query = (
        f"Question: {query_q}\n\n"
        f"Please solve this problem step by step and end with 'Answer: [number]'."
    )
    return header + demo_text + query

def extract_answer(text: str) -> str:
    """从模型输出中提取数字答案"""
    # 尝试提取最后出现的数字
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        return numbers[-1]
    
    # 尝试匹配明确的答案标记
    m = re.search(r"Answer[:\s]*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
        
    return "Unknown"

# ---------- 模型 ----------
print("[INFO] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

def call_model(prompt: str, max_new_tokens=512) -> str:
    """调用模型生成回答，增加max_new_tokens适应数学推理"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只返回模型生成的部分（去除提示词）
    return text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]

# ---------- 主流程 ----------
def main(num_shots=4, max_test=100):
    # 加载数据（使用新的Parquet加载函数）
    data = load_parquet(DATASET_PATH)
    random.shuffle(data)
    demos_src = data[:num_shots]
    tests = data[num_shots:num_shots+max_test]

    demos = [make_contrastive_demo(x) for x in demos_src]
    print(f"[INFO] Few-shot 示例 {len(demos)} 条，测试 {len(tests)} 条")

    results = []
    for i, t in enumerate(tests):
        qid = t["id"]
        q = t["question"]
        gold = t["label"]

        prompt = build_prompt(demos, q)
        out = call_model(prompt)
        pred = extract_answer(out)

        ok = (pred == gold)

        results.append({
            "id": qid,
            "question": q,
            "gold": gold,
            "pred_raw": out,
            "pred": pred,
            "correct": ok
        })

        print(f"\n[{i+1}] Q: {q[:50]}...\nGold: {gold} | Pred: {pred} | OK: {ok}\n---")

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    
    