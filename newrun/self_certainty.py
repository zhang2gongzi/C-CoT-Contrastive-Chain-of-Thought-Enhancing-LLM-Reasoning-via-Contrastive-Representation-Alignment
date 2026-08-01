# -*- coding: utf-8 -*-
"""
Self-Certainty (Kang, Zhao & Song, NeurIPS 2025).
https://arxiv.org/abs/2502.18581

Token-level KL-divergence from uniform as a quality signal.
No external reward model — uses the LLM's own output logprobs.

Usage (run on Linux GPU server):
    python newrun/self_certainty.py \
        --data_path newrundata/gsm8k_merged_flat.jsonl \
        --num_paths 10
"""

import argparse
import json
import math
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL_PATH = "/home2/zzl/model/Qwen3-8B"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

COT_PROMPT = """You are a math reasoning assistant. Solve the following math problem step by step.

Rules:
1. Break down the problem into logical steps.
2. Show your calculations clearly.
3. End your response with "Final Answer: <number>"

Question: {question}

Reasoning:
"""


# ---------- Self-Certainty score ----------

def compute_self_certainty(token_logprobs, vocab_size):
    """
    Equation from Kang et al.:
      Self-Certainty = -1/(n*V) * sum_i sum_j log(V * p_j^i)
    where p_j^i = exp(logprob of token j at position i), V = vocab size.

    token_logprobs: list of floats (one per generated token, log-softmax values)
    vocab_size: int
    """
    if not token_logprobs:
        return -float("inf")
    n = len(token_logprobs)
    total = 0.0
    log_V = math.log(vocab_size)
    for lp in token_logprobs:
        total += lp + log_V  # log(V * p) = log(p) + log(V)
    return -total / (n * vocab_size)


# ---------- Answer extraction ----------

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


# ---------- Load questions ----------

def load_unique_questions(input_path):
    questions = {}
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
    return questions


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Local path or HF id of the LLM")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Flat jsonl with questions")
    parser.add_argument("--num_paths", type=int, default=10)
    parser.add_argument("--output", type=str, default="self_certainty_results.json",
                        help="Output file for per-question results")
    parser.add_argument("--subset_questions", type=int, default=None)
    args = parser.parse_args()

    questions = load_unique_questions(args.data_path)
    qids = list(questions.keys())
    if args.subset_questions:
        qids = qids[:args.subset_questions]

    print(f"Questions: {len(qids)}, paths per question: {args.num_paths}")
    print(f"Loading model from {args.model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    vocab_size = model.config.vocab_size

    results = []
    correct = 0
    total = 0

    for qid in tqdm(qids, desc="Evaluating Self-Certainty"):
        q = questions[qid]
        question = q["question"]
        gold = q["label"]

        paths = []
        for _ in range(args.num_paths):
            prompt = COT_PROMPT.format(question=question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Decode generated text
            gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            cot = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Extract per-token logprobs from scores
            # scores: tuple of (batch_size, vocab_size) logits per step
            token_logprobs = []
            token_ids = gen_ids.tolist()
            for step_i, scores in enumerate(outputs.scores):
                log_probs = torch.log_softmax(scores, dim=-1)
                token_lp = log_probs[0, token_ids[step_i]].item()
                token_logprobs.append(token_lp)

            certainty = compute_self_certainty(token_logprobs, vocab_size)
            pred = extract_final_answer(cot)
            is_correct = int(answers_match(pred, gold))

            paths.append({
                "cot": cot,
                "pred": pred,
                "certainty": certainty,
                "is_correct": is_correct,
            })

        # Select highest-certainty path
        best = max(paths, key=lambda p: p["certainty"])
        selected_correct = best["is_correct"]
        total += 1
        if selected_correct:
            correct += 1

        # Also compute self-consistency (majority vote) for comparison
        votes = {}
        for p in paths:
            ans = p["pred"]
            if ans is not None:
                votes[ans] = votes.get(ans, 0) + 1
        sc_answer = max(votes, key=votes.get) if votes else None
        sc_correct = int(answers_match(sc_answer, gold)) if sc_answer else 0

        results.append({
            "qid": qid,
            "gold": gold,
            "num_paths": args.num_paths,
            "num_correct_paths": sum(p["is_correct"] for p in paths),
            "sel_certainty_correct": selected_correct,
            "sel_consistency_correct": sc_correct,
            "best_certainty": best["certainty"],
        })

    acc = 100.0 * correct / total
    sc_acc = 100.0 * sum(r["sel_consistency_correct"] for r in results) / total

    print(f"\nSelf-Certainty (Kang et al.): {correct}/{total} = {acc:.2f}%")
    print(f"Self-Consistency (majority): {sc_acc:.2f}%")

    # Save detailed results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "method": "Self-Certainty (Kang, Zhao & Song, NeurIPS 2025)",
            "model": args.model_path,
            "accuracy": acc,
            "self_consistency_accuracy": sc_acc,
            "total": total,
            "correct": correct,
            "per_question": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Details saved to {args.output}")


if __name__ == "__main__":
    main()
