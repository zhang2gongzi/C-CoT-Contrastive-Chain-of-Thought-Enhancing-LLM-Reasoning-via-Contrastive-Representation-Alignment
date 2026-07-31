# -*- coding: utf-8 -*-
"""
Unified C-CoT (Contrastive Chain-of-Thought Prompting, Chia et al. 2023) baseline.

This RE-RUNS the *real* Chia et al. method as a single, consistently-described
baseline for Table 1, replacing the previously mislabeled column. Chia's method
is a GENERATION-TIME prompting technique: it prepends few-shot demonstrations
that each contain BOTH a correct and an incorrect reasoning chain, steering the
model away from erroneous reasoning. It does NOT select among pre-sampled paths.

Covers GSM8K / StrategyQA / CommonsenseQA x (Qwen3-8B / LLaMA-3.1-8B-Instruct).

Runs on the SAME test questions as the other Table-1 methods by reading the
question + gold label from the flat jsonl (raw_example.question / gold_label).

  * GSM8K / StrategyQA: the flat jsonl is self-sufficient.
  * CommonsenseQA: the flat jsonl has NO answer options, so pass the official
    CSQA file via --csqa_choices (jsonl with fields id + choices/question_concept)
    so the A-E options can be shown to the model.

Usage (server, GPU):

  # GSM8K, Qwen3-8B
  python newrun/ccot_prompting.py --dataset gsm8k \
      --data_path newrundata/gsm8k_test_flat.jsonl \
      --model_path /home2/zzl/model/Qwen3-8B \
      --output newrun/ccot_gsm8k_qwen.jsonl

  # StrategyQA, LLaMA-3.1-8B  (context available -> used in the prompt)
  python newrun/ccot_prompting.py --dataset strategyqa \
      --data_path newrundata/strategyqa_flat.jsonl \
      --model_path /home2/zzl/model/Llama-3.1-8B-Instruct \
      --output newrun/ccot_sqa_llama.jsonl

  # CommonsenseQA, Qwen3-8B  (needs official choices)
  python newrun/ccot_prompting.py --dataset csqa \
      --data_path newrundata/csqa_500_flat.jsonl \
      --csqa_choices /home2/zzl/C-CoT/database/commonsenseQA/train-00000-of-00001.parquet \
      --model_path /home2/zzl/model/Qwen3-8B \
      --output newrun/ccot_csqa_qwen.jsonl
"""

import os
import re
import json
import random
import argparse
from collections import OrderedDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_flat_questions(path):
    """flat jsonl -> OrderedDict id -> {id, question, context, gold}."""
    q = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            raw = d["raw_example"]
            qid = str(raw["id"])
            if qid in q:
                continue
            q[qid] = {
                "id": qid,
                "question": raw.get("question", ""),
                "context": raw.get("context", ""),
                "gold": str(d.get("gold_label", raw.get("label", ""))).strip(),
            }
    return list(q.values())


def load_csqa_choices(path):
    """Return dict id -> list[(letter, text)] from the official CSQA file."""
    id2choices = {}
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            ch = row["choices"]
            labels = list(ch["label"]); texts = list(ch["text"])
            id2choices[str(row["id"])] = list(zip(labels, texts))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ch = d.get("choices") or d.get("question", {}).get("choices")
                if isinstance(ch, dict):
                    pairs = list(zip(ch["label"], ch["text"]))
                else:
                    pairs = [(c["label"], c["text"]) for c in ch]
                id2choices[str(d.get("id"))] = pairs
    return id2choices


# --------------------------------------------------------------------------- #
# dataset-specific prompt / demo / answer-extraction
# --------------------------------------------------------------------------- #
def fmt_question(dataset, item):
    if dataset == "strategyqa":
        ctx = item.get("context", "")
        head = f"Context: {ctx}\n" if ctx else ""
        return f"{head}Question: {item['question']}\n(Answer Yes or No.)"
    if dataset == "csqa":
        opts = "\n".join(f"{l}. {t}" for l, t in item["choices"])
        return f"Question: {item['question']}\nOptions:\n{opts}\n(Answer with the option letter A-E.)"
    return f"Question: {item['question']}"  # gsm8k


def wrong_answer(dataset, gold, item=None):
    if dataset == "strategyqa":
        return "No" if gold.lower() == "yes" else "Yes"
    if dataset == "csqa":
        letters = [l for l, _ in item["choices"]]
        alts = [l for l in letters if l != gold] or ["A"]
        return random.choice(alts)
    try:
        g = int(re.sub(r"[^\d-]", "", gold) or "0")
        off = random.randint(1, 5)
        return str(g + off if random.random() > 0.5 else g - off)
    except ValueError:
        return "0"


def make_contrastive_demo(dataset, item):
    q_block = fmt_question(dataset, item)
    gold = item["gold"]
    wrong = wrong_answer(dataset, gold, item)
    correct = ("Reason step by step, checking each inference against the given "
               f"facts, to reach a consistent conclusion.\nAnswer: {gold}")
    incorrect = ("Skip a required step and draw an unsupported inference, leading "
                 f"to an inconsistent conclusion.\nAnswer: {wrong}")
    return (f"{q_block}\n\nCorrect reasoning:\n{correct}\n\n"
            f"Incorrect reasoning:\n{incorrect}\n\n----\n\n")


def build_prompt(dataset, demos, item):
    header = ("Below are examples showing both correct and incorrect reasoning. "
              "Learn from the contrast, then solve the final problem with correct, "
              "step-by-step reasoning and end with 'Answer: <your answer>'.\n\n")
    query = f"{fmt_question(dataset, item)}\n\nStep-by-step reasoning:"
    return header + "".join(demos) + query


def extract_answer(dataset, text):
    if dataset == "strategyqa":
        m = re.search(r"Answer[:\s]*\**\s*(yes|no)", text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        m = re.search(r"\b(yes|no)\b", text[::-1].lower())  # last yes/no
        toks = re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)
        return toks[-1].capitalize() if toks else "Unknown"
    if dataset == "csqa":
        m = re.search(r"Answer[:\s]*\**\s*([A-E])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        toks = re.findall(r"\b([A-E])\b", text)
        return toks[-1].upper() if toks else "Unknown"
    # gsm8k
    m = re.search(r"Answer[:\s]*\**\s*(-?\d[\d,]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else "Unknown"


def is_correct(dataset, pred, gold):
    if pred == "Unknown":
        return False
    if dataset == "gsm8k":
        try:
            return abs(float(pred) - float(re.sub(r"[^\d.-]", "", gold))) < 1e-6
        except ValueError:
            return pred == gold
    return pred.strip().lower() == gold.strip().lower()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "strategyqa", "csqa"])
    ap.add_argument("--data_path", required=True, help="flat jsonl with the test questions")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--csqa_choices", help="official CSQA file (parquet/jsonl) with options; required for csqa")
    ap.add_argument("--num_shots", type=int, default=4)
    ap.add_argument("--max_test", type=int, default=0, help="0 = all questions")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    items = load_flat_questions(args.data_path)

    if args.dataset == "csqa":
        if not args.csqa_choices:
            raise SystemExit("[error] --csqa_choices is required for csqa "
                             "(flat jsonl has no answer options).")
        id2ch = load_csqa_choices(args.csqa_choices)
        kept = []
        for it in items:
            if it["id"] in id2ch:
                it["choices"] = id2ch[it["id"]]
                kept.append(it)
        print(f"[data] csqa: matched choices for {len(kept)}/{len(items)} questions")
        items = kept

    random.shuffle(items)
    demos_src, tests = items[:args.num_shots], items[args.num_shots:]
    if args.max_test > 0:
        tests = tests[:args.max_test]
    demos = [make_contrastive_demo(args.dataset, x) for x in demos_src]
    print(f"[ccot] {args.dataset}: {len(demos)} demos, {len(tests)} test questions")

    print(f"[ccot] loading model {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    has_chat = hasattr(tok, "apply_chat_template") and tok.chat_template

    @torch.no_grad()
    def generate(prompt):
        if has_chat:
            msg = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=True, temperature=args.temperature, top_p=0.9)
        gen = out[0][enc["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    results, correct = [], 0
    for i, t in enumerate(tests, 1):
        prompt = build_prompt(args.dataset, demos, t)
        out = generate(prompt)
        pred = extract_answer(args.dataset, out)
        ok = is_correct(args.dataset, pred, t["gold"])
        correct += int(ok)
        results.append({"id": t["id"], "question": t["question"], "gold": t["gold"],
                        "pred": pred, "correct": ok, "raw": out})
        if i % 20 == 0 or i == len(tests):
            print(f"  [{i}/{len(tests)}] running acc = {100.0*correct/i:.2f}%")

    acc = 100.0 * correct / max(1, len(tests))
    print(f"\n==== C-CoT (Chia prompting) | {args.dataset} | model={os.path.basename(args.model_path)} ====")
    print(f"  test questions = {len(tests)}")
    print(f"  accuracy       = {acc:.2f}%  ({correct}/{len(tests)})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps({"dataset": args.dataset, "model": args.model_path,
                            "n_test": len(tests), "accuracy": acc}, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ccot] saved -> {args.output}")


if __name__ == "__main__":
    main()
