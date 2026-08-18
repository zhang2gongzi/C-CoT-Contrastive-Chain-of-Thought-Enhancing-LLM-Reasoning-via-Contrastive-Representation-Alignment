import json
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

###########################################################
# Config
###########################################################
MODEL_PATH = "${MODEL_DIR}/Qwen3-8B"
DATA_PATH = "${PROJECT_ROOT}/database/StrategyQA/strategyqa_train_filtered.json"
OUT_PATH = "${PROJECT_ROOT}/data/cot_train20_first500.json"

MAX_SAMPLES = 500
NUM_PATHS = 20
MAX_NEW_TOKENS = 512

###########################################################
# English CoT Prompt Template (strict rules)
###########################################################
COT_TEMPLATE = """
You are a reasoning assistant. Use the following background information to answer the question.

Background: {description}
Term: {term}

Rules:
1. Analyze the core requirement of the question and relate it to the background information.
2. Provide a step-by-step reasoning process, explaining the rationale of each step.
3. The final answer MUST be a single word: "yes" or "no". Do NOT output anything else.

Question: {question}

Reasoning process:
"""

###########################################################
# Load model
###########################################################
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.eval()
print(f"Model loaded on device: {model.device}")

###########################################################
# Load dataset
###########################################################
print(f"Loading dataset: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

dataset = dataset[:MAX_SAMPLES]
print(f"Loaded {len(dataset)} samples.")

###########################################################
# Extract final answer ("yes"/"no") from generated CoT
###########################################################
def extract_final_answer(text):
    text = text.lower()

    # Look for exact final answer
    match = re.findall(r"(yes|no)", text)
    if match:
        return match[-1]  # last occurrence is the final answer

    return None   # will fallback

###########################################################
# Generate all paths
###########################################################
output_data = []

for sample in tqdm(dataset, desc="Generating CoT"):
    qid = sample["qid"]
    question = sample["question"]
    term = sample["term"]
    desc = sample["description"]
    gt_answer = "yes" if sample["answer"] else "no"

    paths = []

    for _ in range(NUM_PATHS):
        prompt = COT_TEMPLATE.format(
            question=question,
            term=term,
            description=desc
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part
        gen_text = full_output[len(prompt):].strip()

        # Extract final answer
        final_ans = extract_final_answer(gen_text)
        final_ans = final_ans if final_ans in ["yes", "no"] else "yes"  # fallback

        paths.append({
            "cot": gen_text,
            "final_answer": final_ans
        })

    # Store final structured record
    output_data.append({
        "qid": qid,
        "question": question,
        "term": term,
        "description": desc,
        "answer": gt_answer,
        "paths": paths
    })

###########################################################
# Save file
###########################################################
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\nDone! Saved CoT paths to:\n  {OUT_PATH}")
