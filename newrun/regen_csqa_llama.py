# -*- coding: utf-8 -*-
"""Regenerate LLaMA / CommonsenseQA reasoning paths with clean, letter-based labels.

Why: the existing csqa_llama_flat.jsonl is unusable because the generator let the
model paraphrase the answer (gold "C" = "drug store" but the model says
"pharmacy"), so is_correct could never be matched -> 0/4200 correct, 0 mixed
questions, and D-ProtoCoT/ORM training crashes for lack of positive pairs.

This script regenerates the SAME 420 questions (question set read from the old
flat so the split stays comparable) with a prompt that forces an explicit
"Final Answer: X" option letter, samples K paths per question, extracts the
letter, and labels is_correct against the official CommonsenseQA answerKey.
Output schema matches the other *_flat.jsonl files exactly.

Run on the server / AutoDL (needs GPU + LLaMA-3.1-8B):

  python newrun/regen_csqa_llama.py \
      --model_path /home2/zzl/model/Llama-3.1-8B-Instruct \
      --parquet database/commonsenseQA/train-00000-of-00001.parquet \
      --ref_flat newrundata/csqa_llama_flat.jsonl \
      --out newrundata/csqa_llama_flat.jsonl \
      --k 10
"""

import argparse
import json
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def load_questions(ref_flat, parquet):
    """Keep the same 420 questions as the old flat; pull choices+answerKey from parquet."""
    df = pd.read_parquet(parquet)
    meta = {}
    for _, r in df.iterrows():
        ch = r["choices"]
        meta[r["question"]] = {
            "gold": str(r["answerKey"]).upper() if r["answerKey"] else "",
            "letters": [str(l).upper() for l in ch["label"]],
            "texts": [str(t) for t in ch["text"]],
        }
    seen, out = set(), []
    for line in open(ref_flat, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)["raw_example"]["question"]
        if q in seen or q not in meta or not meta[q]["gold"]:
            continue
        seen.add(q)
        out.append((q, meta[q]))
    return out


def build_messages(question, m):
    opts = "\n".join(f"{l}. {t}" for l, t in zip(m["letters"], m["texts"]))
    user = (
        "Answer the following multiple-choice question with step-by-step reasoning.\n"
        'After reasoning, you MUST end with exactly one line in the format '
        '"Final Answer: X" where X is one of A/B/C/D/E.\n\n'
        f"Question: {question}\nOptions:\n{opts}"
    )
    return [{"role": "user", "content": user}]


def extract_letter(text, m):
    """Recover the option letter: prefer the explicit 'Final Answer: X' line."""
    letters = set(m["letters"])
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        mm = re.search(r"final answer\s*[:\-]?\s*\(?([a-e])\)?", line, re.IGNORECASE)
        if mm and mm.group(1).upper() in letters:
            return mm.group(1).upper()
    # fallback: any late "answer is X"
    mm = re.search(r"answer\s*(?:is|:)\s*\(?([a-e])\)?", text[-200:], re.IGNORECASE)
    if mm and mm.group(1).upper() in letters:
        return mm.group(1).upper()
    # last resort: match a choice's exact text
    low = norm(text[-200:])
    for l, t in sorted(zip(m["letters"], m["texts"]), key=lambda kv: -len(kv[1])):
        nt = norm(t)
        if nt and re.search(r"\b" + re.escape(nt) + r"\b", low):
            return l
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home2/zzl/model/Llama-3.1-8B-Instruct")
    ap.add_argument("--parquet", default="database/commonsenseQA/train-00000-of-00001.parquet")
    ap.add_argument("--ref_flat", default="newrundata/csqa_llama_flat.jsonl")
    ap.add_argument("--out", default="newrundata/csqa_llama_flat.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    questions = load_questions(args.ref_flat, args.parquet)
    print(f"[regen] questions={len(questions)} | k={args.k} | model={args.model_path}")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    fout = open(args.out, "w", encoding="utf-8")
    n = corr = unknown = 0
    perq = 0
    mixed = 0
    for qi, (q, m) in enumerate(questions):
        prompt = tok.apply_chat_template(
            build_messages(q, m), tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.k,
                pad_token_id=tok.pad_token_id,
            )
        plen = inputs["input_ids"].shape[1]
        labs = []
        for o in outs:
            cot = tok.decode(o[plen:], skip_special_tokens=True).strip()
            pred = extract_letter(cot, m)
            if pred == "":
                unknown += 1
            ic = int(pred != "" and pred == m["gold"])
            corr += ic
            n += 1
            labs.append(ic)
            fout.write(json.dumps({
                "raw_example": {"id": f"csqa_llama_{qi}", "question": q, "label": m["gold"]},
                "cot": cot, "gold_label": m["gold"], "is_correct": ic,
            }, ensure_ascii=False) + "\n")
        perq += 1
        if 0 < sum(labs) < len(labs):
            mixed += 1
        if (qi + 1) % 20 == 0:
            print(f"  [{qi+1}/{len(questions)}] running acc={100*corr/max(1,n):.1f}% "
                  f"mixed={mixed}/{perq} unknown={unknown}")
    fout.close()
    print(f"[regen] DONE paths={n} correct={corr} ({100*corr/max(1,n):.1f}%) "
          f"unknown={unknown} mixed_questions={mixed}/{perq}")
    print(f"[regen] wrote -> {args.out}")


if __name__ == "__main__":
    main()
