# run_standard_cot_qwen_fixed.py
# ✅ Correctly use Qwen-7B-Chat with chat template
# ✅ Use "Let's think step by step" in proper format
# ✅ Extract answer reliably

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import pandas as pd
import json

# ======================
# 配置
# ======================
MODEL_PATH = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
DATA_PATH = "/home2/zzl/C-CoT/database/bamboogle/test-00000-of-00001-fd9def31e0acf72c.parquet"
OUTPUT_PATH = "standard_cot_qwen7b_fixed_results.json"
NUM_SAMPLES = 125

# ======================
# 加载 Qwen-7B-Chat（必须用 chat 模式）
# ======================
print("🚀 Loading Qwen-7B-Chat with chat template support...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

print(f"✅ Model loaded on device: {model.device}")

# ======================
# 工具函数
# ======================
def standardize(s):
    return s.strip().lower().rstrip(' .,:;')

def extract_answer(text):
    # 多模式提取
    patterns = [
        r'therefore.*?is\s+([A-Z][^.\n,;)]+)',
        r'so.*?is\s+([A-Z][^.\n,;)]+)',
        r'answer\s*[:：]\s*([A-Z][^.\n,;)]+)',
        r'final\s+answer\s+is\s+([A-Z][^.\n,;)]+)',
        r'is\s+([A-Z][^.\n,;)]+)$'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,:;'\"()")
    # 回退：最后一句
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if sentences:
        last = sentences[-1]
        if 'is' in last or 'was' in last:
            parts = re.split(r'\bis\s+|\bwas\s+', last)
            if len(parts) > 1:
                return parts[-1].strip(" .,:;'\"()")
    return "unknown"

# ======================
# 加载数据
# ======================
print("📊 Loading Bamboogle dataset...")
df = pd.read_parquet(DATA_PATH)
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

data = [
    {"question": q, "answer": str(a).strip().title()}
    for q, a in zip(questions, answers)
]
data = data[:NUM_SAMPLES]
print(f"✅ Loaded {len(data)} samples.")

# ======================
# 主循环（使用 Qwen 正确的 chat 格式）
# ======================
print("🧠 Running Standard CoT with Qwen Chat Template...")
correct = 0
results = []

for idx, item in tqdm(enumerate(data), total=len(data)):
    q = item["question"]
    true_ans = item["answer"]

    try:
        # ✅ 正确方式：使用 Qwen 的 chat 模式
        # 第一步：用户提问 + Let's think step by step
        messages = [
            {"role": "user", "content": f"{q}\nLet's think step by step."},
            {"role": "assistant", "content": "Okay, I will reason step by step."}
        ]
        # 应用 chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 让模型继续生成
        )

        # 生成
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取模型生成部分（去掉输入）
        generated_text = full_output[len(prompt):].strip()

        # 提取答案
        pred = extract_answer(generated_text)
        is_correct = standardize(pred) == standardize(true_ans)

        if is_correct:
            correct += 1

        results.append({
            "id": idx,
            "question": q,
            "true_answer": true_ans,
            "predicted": pred,
            "correct": is_correct,
            "full_response": full_output,
            "generated_only": generated_text
        })

    except Exception as e:
        print(f"\n❌ Error at {idx}: {e}")
        results.append({"id": idx, "error": str(e), "predicted": "error"})

# ======================
# 保存结果
# ======================
accuracy = correct / len(results) if results else 0
print(f"\n✅ Fixed CoT Accuracy: {accuracy:.4f} ({correct}/{len(results)})")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump({
        "accuracy": accuracy,
        "total": len(results),
        "correct": correct,
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"💾 Results saved to {OUTPUT_PATH}")