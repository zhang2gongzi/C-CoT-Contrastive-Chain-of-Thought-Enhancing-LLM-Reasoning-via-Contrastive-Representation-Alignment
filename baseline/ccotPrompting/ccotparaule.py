"""
contrastive_cot_logic.py

适配逻辑推理数据集的 Contrastive CoT 脚本
支持 Qwen2.5-7B-Instruct (transformers 本地推理)

依赖:
  pip install transformers accelerate sentencepiece
"""

import json
import random
import re
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------- 配置 -------------
DATASET_PATH = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth5/PARARULE_Plus_Depth5_shuffled_dev_huggingface.jsonl"
MODEL_PATH = "/home2/zzl/model/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "/home2/zzl/C-CoT/baseline/ccotPrompting/depth5_contrastive_cot_results.jsonl"
# ----------------------------

# ---------- 数据 ----------
def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items

def make_contrastive_demo(example: Dict) -> str:
    """
    构造 Contrastive 示例:
    - 正确推理链: 按 label 给出合理解释
    - 错误推理链: 打乱/颠倒结论
    """
    ctx = example["context"]
    q = example["question"]
    label = example["label"]

    if label == 1:
        ans = "Yes"
        wrong_ans = "No"
    else:
        ans = "No"
        wrong_ans = "Yes"

    # 简单拼一个正确解释
    correct_exp = f"From the context, we can deduce step by step. Therefore the statement '{q}' is {ans}."
    # 错误解释
    incorrect_exp = f"From the context, we mistakenly conclude the opposite. Therefore the statement '{q}' is {wrong_ans}."

    demo = (
        f"Context: {ctx}\n"
        f"Question: {q}\n\n"
        f"Correct Explanation:\n{correct_exp}\nAnswer: {ans}\n\n"
        f"Incorrect Explanation:\n{incorrect_exp}\nAnswer: {wrong_ans}\n\n"
        "----\n\n"
    )
    return demo

def build_prompt(demos: List[str], query_ctx: str, query_q: str) -> str:
    header = "Here are examples with Correct and Incorrect explanations. Learn from them.\n\n"
    demo_text = "".join(demos)
    query = (
        f"Context: {query_ctx}\n"
        f"Question: {query_q}\n\n"
        f"Please reason step by step and end with 'Answer: Yes' or 'Answer: No'."
    )
    return header + demo_text + query

def extract_answer(text: str) -> str:
    m = re.search(r"Answer[:\s]*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        if "yes" in ans.lower():
            return "Yes"
        if "no" in ans.lower():
            return "No"
    # fallback
    if "yes" in text.lower():
        return "Yes"
    if "no" in text.lower():
        return "No"
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

def call_model(prompt: str, max_new_tokens=256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# ---------- 主流程 ----------
def main(num_shots=4, max_test=100):
    data = load_jsonl(DATASET_PATH)
    random.shuffle(data)
    demos_src = data[:num_shots]
    tests = data[num_shots:num_shots+max_test]

    demos = [make_contrastive_demo(x) for x in demos_src]
    print(f"[INFO] Few-shot 示例 {len(demos)} 条，测试 {len(tests)} 条")

    results = []
    for i, t in enumerate(tests):
        qid = t["id"]
        ctx = t["context"]
        q = t["question"]
        gold = "Yes" if t["label"] == 1 else "No"

        prompt = build_prompt(demos, ctx, q)
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

        print(f"\n[{i+1}] Q: {q}\nGold: {gold} | Pred: {pred} | OK: {ok}\n---")

    # 保存 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
