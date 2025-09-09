import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from tqdm import tqdm
import json

# ======================
# 配置参数
# ======================
class Config:
    QWEN_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
    GSM8K_PARQUET_PATH = "/home2/zzl/C-CoT/database/gsm8k/test-00000-of-00001.parquet"
    OUTPUT_DIR = "/home2/zzl/C-CoT/baseline/selfcot"
    num_paths = 20         # 每个问题生成几条 CoT
    max_new_tokens = 256
    temperature = 0.7
    top_p = 0.9
    max_test = 200         # 只跑前 200 条
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ======================
# 1. 加载模型和分词器
# ======================
tokenizer = AutoTokenizer.from_pretrained(cfg.QWEN_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg.QWEN_DIR, torch_dtype=torch.float16, device_map="auto"
)

# ======================
# 2. 读取 GSM8K 数据
# ======================
df = pd.read_parquet(cfg.GSM8K_PARQUET_PATH)
dataset = df[["question", "answer"]].dropna().reset_index(drop=True)
dataset = dataset.iloc[:cfg.max_test]   # 只取前 200 条

# ======================
# 3. 多路径 CoT 生成
# ======================
def generate_cot_paths(question: str, num_paths: int = 5):
    prompt = f"Question: {question}\nLet's reason step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    paths = []
    for _ in range(num_paths):
        output = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        paths.append(text)
    return paths

# ======================
# 4. 答案抽取
# ======================
def extract_answer(text: str):
    # 匹配数字（整数/小数）
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        return matches[-1]  # 取最后一个数字
    return None

# ======================
# 5. Self-Consistency Voting
# ======================
def self_consistency_predict(question: str, num_paths=20):
    paths = generate_cot_paths(question, num_paths)
    answers = [extract_answer(p) for p in paths if extract_answer(p) is not None]
    if not answers:
        return None, paths
    majority = Counter(answers).most_common(1)[0][0]
    return majority, paths

# ======================
# 6. 运行实验
# ======================
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
results = []
correct = 0

for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    q, gold = row["question"], str(row["answer"])
    pred, paths = self_consistency_predict(q, cfg.num_paths)

    ok = (pred == gold)
    results.append({
        "id": int(idx),
        "question": q,
        "gold": gold,
        "pred": pred,
        "is_correct": ok,
        "paths": paths
    })

    if ok:
        correct += 1

acc = correct / len(dataset)
print(f"Self-Consistency CoT on GSM8K (first {cfg.max_test}) | Accuracy = {acc:.4f}")

# ======================
# 7. 保存结果
# ======================
out_file = os.path.join(cfg.OUTPUT_DIR, f"self_consistency_results_{cfg.max_test}.jsonl")
with open(out_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] 结果已保存到 {out_file}")
