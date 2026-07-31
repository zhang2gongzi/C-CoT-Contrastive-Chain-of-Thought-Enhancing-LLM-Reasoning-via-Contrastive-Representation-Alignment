# -*- coding: utf-8 -*-
"""
PiCSAR: Probabilistic Confidence Selection And Ranking for Reasoning Chains
(Leang et al., EMNLP 2025)
https://arxiv.org/abs/2508.21787

Score(r, y) = log p(r | x) + log p(y | r, x)

- Reasoning confidence: sum of token logprobs during CoT generation
- Answer confidence: log p(answer | question + reasoning), computed via an
  extra forward pass that isolates the answer span

Usage (run on Linux GPU server):
    python newrun/picsar.py \
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


MODEL_PATH = "/home2/zzl/model/Qwen3-8B"
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

# Prompt to elicit the final answer from a reasoning chain (for answer confidence)
ANSWER_EXTRACTION_PROMPT = """{question}

{reasoning}

Based on the reasoning above, the final answer is:"""


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


# ---------- PiCSAR scoring ----------

def compute_reasoning_confidence(token_logprobs):
    """log p(r | x) = sum of token-level log-probabilities."""
    return sum(token_logprobs) if token_logprobs else -float("inf")


def compute_answer_confidence(model, tokenizer, question, reasoning, answer_str):
    """
    log p(y | r, x): probability of the answer string conditioned on
    the question + reasoning chain.
    """
    prompt = ANSWER_EXTRACTION_PROMPT.format(
        question=question, reasoning=reasoning
    )
    # Full text: prompt + answer
    full = prompt + answer_str

    inputs = tokenizer(full, return_tensors="pt").to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # outputs.logits: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Only score the answer tokens (positions after the prompt)
    answer_ids = inputs["input_ids"][0, prompt_len:]
    if len(answer_ids) == 0:
        return -float("inf")

    total_logprob = 0.0
    log_softmax = torch.log_softmax(logits, dim=-1)
    for i, token_id in enumerate(answer_ids):
        pos = prompt_len + i - 1  # logits at position pos predict token at pos+1
        total_logprob += log_softmax[pos, token_id].item()

    return total_logprob


def picsar_score(reasoning_confidence, answer_confidence):
    """PiCSAR (unnormalized): sum of both confidence terms."""
    return reasoning_confidence + answer_confidence


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
    parser.add_argument("--data_path", type=str, required=True,
                        help="Flat jsonl with questions (test set)")
    parser.add_argument("--num_paths", type=int, default=10)
    parser.add_argument("--output", type=str, default="picsar_results.json",
                        help="Output file for per-question results")
    parser.add_argument("--subset_questions", type=int, default=None)
    args = parser.parse_args()

    questions = load_unique_questions(args.data_path)
    qids = list(questions.keys())
    if args.subset_questions:
        qids = qids[:args.subset_questions]

    print(f"Questions: {len(qids)}, paths per question: {args.num_paths}")
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    correct = 0
    total = 0

    for qid in tqdm(qids, desc="PiCSAR evaluation"):
        q = questions[qid]
        question = q["question"]
        gold = q["label"]

        paths = []
        for _ in range(args.num_paths):
            # --- Step 1: Generate CoT with logprobs ---
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

            gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            cot = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Reasoning confidence: log p(r | x)
            token_logprobs = []
            token_ids_list = gen_ids.tolist()
            for step_i, scores in enumerate(outputs.scores):
                log_probs = torch.log_softmax(scores, dim=-1)
                token_lp = log_probs[0, token_ids_list[step_i]].item()
                token_logprobs.append(token_lp)

            rc = compute_reasoning_confidence(token_logprobs)

            # Extract answer
            pred = extract_final_answer(cot)
            is_correct = int(answers_match(pred, gold))

            # Answer confidence: log p(y | r, x)
            answer_str = str(pred) if pred else ""
            ac = compute_answer_confidence(model, tokenizer, question, cot, answer_str)

            score = picsar_score(rc, ac)

            paths.append({
                "cot": cot,
                "pred": pred,
                "reasoning_confidence": rc,
                "answer_confidence": ac,
                "picsar_score": score,
                "is_correct": is_correct,
            })

        # Select highest PiCSAR score
        best = max(paths, key=lambda p: p["picsar_score"])
        selected_correct = best["is_correct"]
        total += 1
        if selected_correct:
            correct += 1

        # Self-consistency baseline
        votes = {}
        for p in paths:
            ans = p["pred"]
            if ans is not None:
                votes[ans] = votes.get(ans, 0) + 1
        sc_answer = max(votes, key=votes.get) if votes else None
        sc_correct = int(answers_match(sc_answer, gold)) if sc_answer else 0

        # Upper-bound oracle (best among N paths)
        oracle_correct = int(any(p["is_correct"] for p in paths))

        results.append({
            "qid": qid,
            "gold": gold,
            "num_paths": args.num_paths,
            "num_correct_paths": sum(p["is_correct"] for p in paths),
            "picsar_correct": selected_correct,
            "sel_consistency_correct": sc_correct,
            "oracle_correct": oracle_correct,
            "best_score": best["picsar_score"],
            "best_rc": best["reasoning_confidence"],
            "best_ac": best["answer_confidence"],
        })

    acc = 100.0 * correct / total
    sc_acc = 100.0 * sum(r["sel_consistency_correct"] for r in results) / total
    oracle_acc = 100.0 * sum(r["oracle_correct"] for r in results) / total

    print(f"\nPiCSAR:              {correct}/{total} = {acc:.2f}%")
    print(f"Self-Consistency:    {sc_acc:.2f}%")
    print(f"Oracle (upper bound): {oracle_acc:.2f}%")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "method": "PiCSAR (Leang et al., EMNLP 2025)",
            "model": MODEL_PATH,
            "accuracy": acc,
            "self_consistency_accuracy": sc_acc,
            "oracle_accuracy": oracle_acc,
            "total": total,
            "correct": correct,
            "per_question": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Details saved to {args.output}")


if __name__ == "__main__":
    main()
