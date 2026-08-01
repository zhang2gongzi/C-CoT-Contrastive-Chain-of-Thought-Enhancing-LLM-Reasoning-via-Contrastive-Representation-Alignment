# -*- coding: utf-8 -*-
"""
Generate CoT paths using Qwen3-14B for D-ProtoCoT experiments.
Reads questions from an existing flat jsonl, generates new CoT with 14B.

Precision: bf16 by default (fits a 32GB card like RTX 5090). Pass --load_in_4bit
to run on a 24GB card at the cost of speed.

Usage (GPU server / AutoDL):
    # download model first, e.g.
    #   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-14B \
    #       --local-dir /root/autodl-tmp/Qwen3-14B

    # Test set (bf16, 32GB card)
    python generate_cot_14b.py \
        --model_path /root/autodl-tmp/Qwen3-14B \
        --input newrundata/gsm8k_test_flat.jsonl \
        --output newrundata/gsm8k_test_14b_flat.jsonl

    # On a 24GB card:
    python generate_cot_14b.py --load_in_4bit \
        --model_path /root/autodl-tmp/Qwen3-14B \
        --input newrundata/gsm8k_test_flat.jsonl \
        --output newrundata/gsm8k_test_14b_flat.jsonl
"""

import argparse
import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== Defaults (overridable via CLI) =====
DEFAULT_MODEL_PATH = "/home2/zzl/model/Qwen3-14B"
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
    text = text.strip()
    m = re.search(r"final\s*answer\s*[:：]\s*(-?[\d,\.]+)", text, re.I)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"(?:the\s+)?answer\s*(?:is|:)\s*(-?[\d,\.]+)", text, re.I)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,\.]+", text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()


def load_unique_questions(input_path):
    """Extract unique questions from a flat jsonl file."""
    questions = {}  # qid -> {question, label}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ex = obj["raw_example"]
            qid = ex["id"]
            if qid not in questions:
                questions[qid] = {
                    "question": ex["question"],
                    "label": obj["gold_label"],
                }
    print(f"Loaded {len(questions)} unique questions from {input_path}")
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Local path or HF id of the 14B model")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4-bit quantization (for 24GB cards); default is bf16")
    parser.add_argument("--input", type=str, required=True,
                        help="Input flat jsonl (to extract questions from)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output flat jsonl")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Limit number of questions")
    args = parser.parse_args()

    out_path = args.output
    ckpt_path = out_path + ".ckpt"

    questions = load_unique_questions(args.input)
    qids = list(questions.keys())
    if args.max_questions and args.max_questions < len(qids):
        qids = qids[:args.max_questions]
        print(f"Limited to {args.max_questions} questions")

    print(f"Loading {args.model_path} ({'4-bit' if args.load_in_4bit else 'bf16'})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.load_in_4bit:
        load_kwargs.update(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    # checkpoint / resume
    done = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            done = set(int(l.strip()) for l in f if l.strip().isdigit())
        print(f"[resume] {len(done)} questions already done, skipping...")

    total_paths = len(done) * NUM_PATHS
    total_correct = 0
    n_questions = len(done)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    mode = "a" if done else "w"
    try:
        with open(out_path, mode, encoding="utf-8") as fout, \
             open(ckpt_path, "a", encoding="utf-8") as fckpt:
            for i, qid in enumerate(tqdm(qids, desc="Generating CoTs")):
                if i in done:
                    continue

                q = questions[qid]
                question = q["question"]
                gold = q["label"]

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
                    fout.flush()
                    total_paths += 1
                    if is_correct:
                        total_correct += 1

                done.add(i)
                fckpt.write(f"{i}\n")
                fckpt.flush()
                n_questions += 1

    except KeyboardInterrupt:
        print(f"\n[interrupted] {n_questions} questions saved. Rerun to resume.")
        return

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    acc = 100.0 * total_correct / max(1, total_paths)
    print(f"\nDone: {n_questions} questions x {NUM_PATHS} paths = {total_paths} total")
    print(f"Correct: {total_correct} ({acc:.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
