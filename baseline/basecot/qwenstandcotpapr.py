# -*- coding: utf-8 -*-
"""
基于标准 CoT（Chain-of-Thought）推理，使用 Qwen-7B-Chat 模型
处理 PARARULE-Plus Depth2 数据集的前 200 条数据，并计算准确率。
"""

import json
import os
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------------
# 1. 配置参数
# -----------------------------
MODEL_PATH = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
DATA_PATH = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth2/PARARULE_Plus_Depth2_shuffled_train_huggingface.jsonl"
OUTPUT_PATH = "./qwen7b_cot_results.jsonl"
NUM_SAMPLES = 300

# -----------------------------
# 2. 加载模型和分词器 
# -----------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

# -----------------------------
# 3. 读取数据集前200条
# -----------------------------
print(f"Loading first {NUM_SAMPLES} samples from {DATA_PATH}...")
samples = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= NUM_SAMPLES:
            break
        data = json.loads(line.strip())
        samples.append(data)

print(f"Loaded {len(samples)} samples.")

# -----------------------------
# 4. CoT 提示模板
# -----------------------------
COT_PROMPT_TEMPLATE = """
你是一个逻辑推理助手。请逐步推理以下问题，并在最后给出最终答案。

问题：
{question}

请按照以下格式回答：
推理过程：
...

最终答案：...
""".strip()

# -----------------------------
# 5. 推理函数
# -----------------------------
def run_cot_inference(question: str) -> str:
    prompt = COT_PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# -----------------------------
# 6. 答案提取函数（从模型输出中提取“最终答案”）
# -----------------------------
def extract_final_answer(response: str) -> str:
    """
    从模型输出中提取最终答案。
    假设格式为：最终答案：xxx
    """
    lines = response.split('\n')
    for line in lines:
        if line.startswith("最终答案："):
            answer = line.replace("最终答案：", "").strip()
            # 去除可能的标点或多余说明
            answer = answer.rstrip('。').rstrip('.').strip()
            return answer
    return ""  # 未找到则返回空

# -----------------------------
# 7. 主推理循环 + 结果收集
# -----------------------------
results = []
correct_count = 0

for idx, sample in enumerate(tqdm(samples, desc="Running CoT Inference")):
    question = sample.get("question", "")
    gold_answer = str(sample.get("answer", "")).strip()  # 确保是字符串

    try:
        model_response = run_cot_inference(question)
    except Exception as e:
        model_response = f"Error during inference: {str(e)}"
        print(f"Error on sample {idx}: {e}")

    # 提取模型预测答案
    model_answer = extract_final_answer(model_response)

    # 判断是否正确（简单精确匹配）
    is_correct = (model_answer.lower() == gold_answer.lower())
    if is_correct:
        correct_count += 1

    # 保存结果
    result = {
        "id": idx,
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "is_correct": is_correct,
        "full_response": model_response
    }
    results.append(result)

    # 实时追加写入文件
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

# -----------------------------
# 8. 计算并输出准确率
# -----------------------------
accuracy = correct_count / len(results) * 100
print(f"\n✅ Inference completed for {len(results)} samples.")
print(f"🎯 Accuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")
print(f"💾 Results saved to: {OUTPUT_PATH}")

# 可选：打印部分错误案例
print("\n--- Incorrect Predictions ---")
incorrect_examples = [r for r in results if not r["is_correct"]]
for r in incorrect_examples[:3]:  # 显示前3个错误
    print(f"Q: {r['question']}")
    print(f"Gold: {r['gold_answer']} | Model: {r['model_answer']}")
    print("-" * 50)