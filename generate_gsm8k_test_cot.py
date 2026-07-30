# -*- coding: utf-8 -*-
"""
Generate 10 CoT paths for each question in the GSM8K official TEST set.

Usage (run on Linux GPU server):
    python generate_gsm8k_test_cot.py                          # all 1319 questions
    python generate_gsm8k_test_cot.py --max_questions 200      # sample 200 questions

Output: gsm8k_test_flat.jsonl (dprotocot flat jsonl format)
"""

import argparse
import json
import os
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Config =====
MODEL_PATH = "/home2/zzl/model/Qwen3-8B"
TEST_PARQUET = "database/gsm8k/test-00000-of-00001.parquet"
OUT_PATH = "gsm8k_test_flat.jsonl"

NUM_PATHS = 10
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42

COT_PROMPT = """You are a math reasoning assistant. Solve the following math problem step by step.

Rules:
1. Break down the problem into logical steps.
2. Show your calculations clearly.
3. End your response with "Final Answer: <number>"

Question: {question}

Reasoning:
"""


def extract_final_answer(text: str):
    """Extract the final numeric answer from a CoT path."""
    text = text.strip()

    # Pattern 1: "Final Answer: X"
    m = re.search(r"final\s*answer\s*[:：]\s*(-?[\d,\.]+)", text, re.I)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 2: "The answer is X"  (near the end)
    m = re.search(r"(?:the\s+)?answer\s*(?:is|:)\s*(-?[\d,\.]+)", text, re.I)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 3: "#### X" at end (GSM8K format)
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 4: last number in the text
    nums = re.findall(r"-?[\d,\.]+", text)
    return nums[-1].replace(",", "") if nums else None


def extract_gold_answer(answer_text: str):
    """Extract final number from GSM8K gold answer (format: ... #### N)."""
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    # fallback
    nums = re.findall(r"-?[\d,\.]+", answer_text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred: str, gold: str) -> bool:
    """Compare predicted and gold numeric answers."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Number of questions to sample (default: all)")
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    print("Loading GSM8K test set...")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"Test set: {len(df)} questions")

    if args.max_questions and args.max_questions < len(df):
        rng = random.Random(SEED)
        indices = sorted(rng.sample(range(len(df)), args.max_questions))
        df = df.iloc[indices].reset_index(drop=True)
        print(f"Sampled {args.max_questions} questions (seed={SEED})")

    total_paths = 0
    total_correct = 0
    n_questions = 0

    os.makedirs(os.path.dirname(OUT_PATH) if os.path.dirname(OUT_PATH) else ".", exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating CoTs"):
            question = row["question"]
            gold_raw = row["answer"]
            gold = extract_gold_answer(gold_raw)
            qid = f"gsm8k_test_{n_questions}"

            for _ in range(NUM_PATHS):
                prompt = COT_PROMPT.format(question=question)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )
                cot = tokenizer.decode(outputs[0], skip_special_tokens=True)
                cot = cot[len(prompt):].strip()

                pred = extract_final_answer(cot)
                is_correct = int(answers_match(pred, gold))

                obj = {
                    "raw_example": {
                        "id": qid,
                        "question": question,
                        "label": gold,
                    },
                    "cot": cot,
                    "gold_label": gold,
                    "is_correct": is_correct,
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_paths += 1
                if is_correct:
                    total_correct += 1
            n_questions += 1

    acc = 100.0 * total_correct / max(1, total_paths)
    print(f"\nDone: {n_questions} questions x {NUM_PATHS} paths = {total_paths} total")
    print(f"Correct: {total_correct} ({acc:.1f}%)")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
