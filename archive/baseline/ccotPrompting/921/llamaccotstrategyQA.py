# run_contrastive_cot.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy
from tqdm import tqdm
import re
import random

# ======================
# 配置路径
# ======================
MODEL_PATH = "${MODEL_DIR}/../model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
DATA_PATH = "${PROJECT_ROOT}/database/StrategyQA/strategyqa_train.json"
NUM_SAMPLES = 207

# 加载英文 NLP 模型
try:
    nlp = spacy.load("${MODEL_DIR}/en_core_web_sm-3.8.0")
except OSError:
    print("Please run: python -m spacy download en_core_web_sm")
    exit()

# ======================
# 加载模型
# ======================
print("Loading Llama-2-7b-chat-hf...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# ======================
# 工具函数
# ======================

def extract_numeric_and_entities(text):
    """提取数字和实体用于扰动"""
    doc = nlp(text)
    numbers = re.findall(r'\b\d+\b', text)
    entities = [ent.text for ent in doc.ents if len(ent.text) > 1]
    return list(set(numbers + entities))

def corrupt_reasoning(text, max_replace=2):
    """打乱关键对象生成错误推理"""
    items = extract_numeric_and_entities(text)
    if len(items) < 2:
        return text + " However, some numbers were mixed up."
    
    shuffled = items.copy()
    random.shuffle(shuffled)
    
    new_text = text
    replaced = 0
    for orig, repl in zip(items, shuffled):
        if orig != repl and replaced < max_replace:
            new_text = new_text.replace(orig, repl, 1)
            replaced += 1
    return new_text

def build_positive_rationale(facts, answer):
    """用 facts 构造推理链"""
    rationale = " ".join(facts)
    answer_str = "yes" if answer else "no"
    rationale += f" Therefore, the answer is {answer_str}."
    return rationale

def format_contrastive_prompt(question, pos_rationale, neg_rationale):
    prompt = f"""[Question]
{question}

[Correct Reasoning]
{pos_rationale}

[Incorrect Reasoning]
{neg_rationale}
This reasoning contains logical or factual errors.

[Your Task]
Answer the following question. Think step by step.
{question}
Your reasoning:"""
    return prompt

def extract_answer(text):
    """提取最终答案"""
    text = text.lower()
    if "yes" in text and "no" not in text[-10:]:
        return "yes"
    elif "no" in text and "yes" not in text[-10:]:
        return "no"
    return "unknown"

# ======================
# 主逻辑
# ======================
print("Loading StrategyQA data...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

data = data[:NUM_SAMPLES]

correct = 0
results = []

for idx, item in tqdm(enumerate(data), total=len(data), desc="Evaluating"):

    question = item["question"].strip()
    gold_answer_bool = item["answer"]  # True or False
    gold_answer = "yes" if gold_answer_bool else "no"

    # 使用 facts 构造正例推理
    try:
        facts = item["facts"]
        pos_rationale = build_positive_rationale(facts, gold_answer_bool)
        neg_rationale = corrupt_reasoning(pos_rationale)

        # 构造对比提示
        prompt = format_contrastive_prompt(question, pos_rationale, neg_rationale)

        # Llama-2 格式
        full_prompt = f"""<s>[INST] <<SYS>>
You are a logical and precise assistant. Compare the correct and incorrect reasoning, then answer the new question with a short final answer.
<</SYS>>

{prompt} [/INST]"""

        # 生成
        outputs = pipe(
            full_prompt,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        raw_output = outputs[0]["generated_text"]
        model_response = raw_output[len(full_prompt):].strip()

        pred_answer = extract_answer(model_response)

        is_correct = (pred_answer == gold_answer)
        if is_correct:
            correct += 1

        results.append({
            "idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "model_response": model_response,
            "pos_rationale": pos_rationale,
            "neg_rationale": neg_rationale,
            "is_correct": is_correct
        })

    except Exception as e:
        print(f"Error at {idx}: {e}")
        results.append({"idx": idx, "error": str(e), "pred_answer": "unknown"})
        continue

# ======================
# 输出准确率
# ======================
accuracy = correct / len([r for r in results if "error" not in r])
print("\n" + "="*60)
print(f"✅ Contrastive CoT Evaluation on StrategyQA")
print(f"🧪 Model: Llama-2-7b-chat-hf")
print(f"📊 Samples: {len(results)}/{NUM_SAMPLES}")
print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("="*60)

# 保存结果
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"results_c_cot_strategyqa_{NUM_SAMPLES}_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to: results_c_cot_strategyqa_{NUM_SAMPLES}_{timestamp}.json")