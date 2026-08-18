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
MODEL_PATH = "${MODEL_DIR}/Qwen3-14B"
TEST_PARQUET = "database/gsm8k/test-00000-of-00001.parquet"
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
    parser.add_argument("--output", type=str, default="gsm8k_test_14b_flat.jsonl",
                        help="Output file path")
    args = parser.parse_args()

    out_path = args.output
    ckpt_path = out_path + ".ckpt"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
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

    # ---- checkpoint / resume ----
    done = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            done = set(int(l.strip()) for l in f if l.strip().isdigit())
        print(f"[resume] {len(done)} questions already done, skipping...")

    total_paths = len(done) * NUM_PATHS
    total_correct = 0  # approximate; real count is in the saved file
    n_questions = len(done)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    mode = "a" if done else "w"
    try:
        with open(out_path, mode, encoding="utf-8") as fout, \
             open(ckpt_path, "a", encoding="utf-8") as fckpt:
            for i, (_, row) in enumerate(tqdm(list(df.iterrows()), total=len(df), desc="Generating CoTs")):
                qid = f"gsm8k_test_{i}"
                if i in done:
                    continue

                question = row["question"]
                gold_raw = row["answer"]
                gold = extract_gold_answer(gold_raw)

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
                    fout.flush()  # flush after each path
                    total_paths += 1
                    if is_correct:
                        total_correct += 1

                # checkpoint: mark this question done
                done.add(i)
                fckpt.write(f"{i}\n")
                fckpt.flush()
                n_questions += 1

    except KeyboardInterrupt:
        print(f"\n[interrupted] {n_questions} questions saved. Rerun to resume.")
        return

    # clean up checkpoint on successful completion
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    acc = 100.0 * total_correct / max(1, total_paths)
    print(f"\nDone: {n_questions} questions x {NUM_PATHS} paths = {total_paths} total")
    print(f"Correct: {total_correct} ({acc:.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
