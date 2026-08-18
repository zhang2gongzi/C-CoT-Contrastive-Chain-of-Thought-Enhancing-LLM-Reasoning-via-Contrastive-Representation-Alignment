# run_cot_eval.py
import pandas as pd
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from tqdm import tqdm
import json

# =======================
# 配置路径
# =======================
MODEL_PATH = "${MODEL_DIR}/../model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
DATA_PATH = "${PROJECT_ROOT}/database/commonsenseQA/train-00000-of-00001.parquet"

# =======================
# 加载模型和 tokenizer
# =======================
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

# 使用 pipeline（方便生成）
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# =======================
# 加载数据集
# =======================
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
# CommonsenseQA 格式：question, choices (dict), answerKey
# 我们只取前 100 条做测试（可改）
df = df.head(200)

def format_question(row):
    """格式化问题输入"""
    question = row['question']
    choices = row['choices']
    options = ""
    for i, text in enumerate(choices['text']):
        options += f"{chr(65+i)}. {text}\n"
    prompt = f"""Answer the following multiple-choice question. Think step by step.

Question: {question}
Options:
{options}
Answer with the option letter only (A, B, C, D, or E)."""
    return prompt

# =======================
# C-CoT 方法实现
# =======================
def generate_reasoning_paths(prompt, num_paths=10):
    """生成 N 条 CoT 推理路径，适用于 Qwen-Chat 模型"""
    # 手动构造 Qwen 的 chat 模板
    system_message = "You are a helpful assistant."
    user_message = prompt
    
    text = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    text += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    
    # 注意：这里不需要包含 <|im_end|> 在最后，因为模型要继续生成
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = pipe(
        text,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=256,
        num_return_sequences=num_paths,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id  # Qwen 使用 eos 作为 pad
    )
    
    # 提取生成的部分（去掉输入）
    generated_paths = []
    for output in outputs:
        full_text = output['generated_text']
        # 只保留 assistant 回复部分
        if "<|im_start|>assistant" in full_text:
            generated_text = full_text.split("<|im_start|>assistant")[-1]
            # 去掉 <|im_end|> 如果存在
            generated_text = generated_text.replace("<|im_end|>", "").strip()
            generated_paths.append(generated_text)
        else:
            generated_paths.append(full_text.strip())
            
    return generated_paths

def extract_answer(path):
    """从生成文本中提取答案（A/B/C/D/E）"""
    path = path.strip()
    if not path:
        return None
    # 优先找独立字母
    import re
    match = re.search(r'^\s*([A-E])\s*$', path)
    if match:
        return match.group(1)
    # 再找 "Answer: A" 这类
    match = re.search(r'Answer\s*[:：]\s*([A-E])', path, re.IGNORECASE)
    if match:
        return match.group(1)
    # 找结尾的字母
    match = re.search(r'([A-E])\s*$', path)
    if match:
        return match.group(1)
    return None

def majority_voting(answers):
    """简单投票（Self-Consistency）"""
    from collections import Counter
    if not answers:
        return None
    counter = Counter(answers)
    return counter.most_common(1)[0][0]

# =======================
# 评估主循环
# =======================
results = []  # 用于保存详细结果
correct = 0
total = 0

print("Running C-CoT evaluation and saving paths...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = format_question(row)
    gold_key = row['answerKey']
    
    # Step 1: 生成 20 条路径
    paths = generate_reasoning_paths(prompt, num_paths=10)
    
    # Step 2: 提取答案并标记是否正确
    path_data = []
    valid_answers = []
    for path in paths:
        pred = extract_answer(path)
        is_correct = 1 if pred == gold_key else 0
        if pred is not None:
            valid_answers.append(pred)
        path_data.append({
            "path": path,
            "predicted_answer": pred,
            "is_correct": is_correct
        })
    
    # Step 3: 投票（仅用于记录 baseline）
    pred = majority_voting(valid_answers) if valid_answers else None
    final_correct = 1 if pred == gold_key else 0
    
    # 保存整题数据
    results.append({
        "question_id": int(idx),
        "question": row['question'],
        "gold_answer": gold_key,
        "paths": path_data,
        "final_prediction": pred,
        "final_correct": final_correct
    })

    if final_correct:
        correct += 1
    total += 1
# =======================
# 输出结果
# =======================
accuracy = correct / total
print(f"\n{'='*50}")
print(f"C-CoT Evaluation on CommonsenseQA (subset)")
print(f"Model: Qwen-7B-Chat")
print(f"Num samples: {total}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"{'='*50}")

# === 保存到文件 ===
OUTPUT_PATH = "${PROJECT_ROOT}/baseline/commonsenseQA/ccot/generated_paths/commonsenseqa_qwen7b_paths.jsonl"
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ All reasoning paths saved to {OUTPUT_PATH}")