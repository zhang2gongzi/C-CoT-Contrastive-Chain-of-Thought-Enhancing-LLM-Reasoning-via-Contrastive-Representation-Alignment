# -*- coding: utf-8 -*-
"""
Self-Certainty (Kang, Zhao & Song, NeurIPS 2025).
https://arxiv.org/abs/2502.18581

Token-level KL-divergence from uniform as a quality signal.
No external reward model — uses the LLM's own output logprobs.

This scores the ALREADY-GENERATED paths in a flat jsonl via a single
teacher-forcing forward pass per path, so Self-Certainty, Self-Consistency,
ORM, and D-ProtoCoT are all evaluated on the SAME K sampled paths and are
directly comparable to Table 1.

Usage (run on Linux GPU server):
    python newrun/self_certainty.py \
        --data_path newrundata/gsm8k_test_flat.jsonl
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


DEFAULT_MODEL_PATH = "${MODEL_DIR}/Qwen3-8B"

COT_PROMPT = """You are a math reasoning assistant. Solve the following math problem step by step.

Rules:
1. Break down the problem into logical steps.
2. Show your calculations clearly.
3. End your response with "Final Answer: <number>"

Question: {question}

Reasoning:
"""


# ---------- Self-Certainty score ----------

def compute_self_certainty_from_logprobs(logprobs_matrix, log_V):
    """
    Kang et al. self-certainty, computed over the FULL vocabulary distribution:
      SC = -1/(n*V) * sum_i sum_j log(V * p_j^i)
         = mean_i [ -mean_j(log p_j^i) - log V ]
    Higher = distribution more peaked (more confident), i.e. farther from uniform.

    logprobs_matrix: FloatTensor (n_positions, vocab_size) of log-softmax values,
        one row per scored (generated) token position — the full distribution,
        not just the sampled token.
    log_V: math.log(vocab_size)
    """
    if logprobs_matrix.numel() == 0:
        return -float("inf")
    # per-position: -mean over vocab of log p_j - log V
    per_pos = -logprobs_matrix.mean(dim=-1) - log_V  # (n_positions,)
    return per_pos.mean().item()


# ---------- Load paths (ALL K paths per question) ----------

def load_paths_by_question(input_path):
    """Group every stored path under its question id (keeps K paths/question)."""
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
                    "paths": [],
                }
            questions[qid]["paths"].append({
                "cot": obj["cot"],
                "is_correct": int(obj["is_correct"]),
            })
    return questions


# ---------- Score one stored path by teacher forcing ----------

def score_path(model, tokenizer, question, cot, log_V):
    """Single forward pass over prompt+cot; self-certainty over the cot tokens."""
    prompt = COT_PROMPT.format(question=question)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    full_ids = tokenizer(prompt + cot, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return -float("inf")

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # (seq_len, vocab)

    # logits at position pos predict token pos+1; score the cot tokens
    # (positions prompt_len .. seq_len-1), predicted by logits[prompt_len-1 .. seq_len-2]
    pred_logits = logits[prompt_len - 1: full_ids.shape[1] - 1]   # (n_cot, vocab)
    logprobs = torch.log_softmax(pred_logits.float(), dim=-1)     # full distribution
    return compute_self_certainty_from_logprobs(logprobs, log_V)


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


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Local path or HF id of the LLM")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Flat jsonl with pre-generated paths (test set)")
    parser.add_argument("--output", type=str, default="self_certainty_results.json",
                        help="Output file for per-question results")
    parser.add_argument("--subset_questions", type=int, default=None)
    args = parser.parse_args()

    questions = load_paths_by_question(args.data_path)
    qids = list(questions.keys())
    if args.subset_questions:
        qids = qids[:args.subset_questions]

    print(f"Questions: {len(qids)}, "
          f"paths total: {sum(len(questions[q]['paths']) for q in qids)}")
    print(f"Loading model from {args.model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    log_V = math.log(model.config.vocab_size)

    results = []
    correct = 0
    total = 0

    for qid in tqdm(qids, desc="Evaluating Self-Certainty"):
        q = questions[qid]
        question = q["question"]
        gold = q["label"]

        paths = []
        for p in q["paths"]:
            cot = p["cot"]
            certainty = score_path(model, tokenizer, question, cot, log_V)
            pred = extract_final_answer(cot)
            paths.append({
                "pred": pred,
                "certainty": certainty,
                "is_correct": p["is_correct"],
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
            "num_paths": len(paths),
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
