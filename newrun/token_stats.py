# -*- coding: utf-8 -*-
"""
Token-length statistics of generated reasoning paths (Table: token-length).

Tokenizes the `cot` field of each flat jsonl with the BERT-base-uncased
tokenizer (the encoder used by D-ProtoCoT / ORM) WITHOUT truncation, so the
reported max length and >512 fraction reflect the true distribution that
motivates the hierarchical encoding strategy (Section 3.2).

Usage (run on Linux GPU server, where the BERT tokenizer is available):
    python newrun/token_stats.py --bert_model ${MODEL_DIR}/bert-base-uncased
    # or point at any local dir containing vocab.txt, e.g. dprotocot_runs/run
"""

import argparse
import json
import os

from transformers import BertTokenizerFast


DEFAULT_BERT_MODEL = "${MODEL_DIR}/bert-base-uncased"

# name -> flat jsonl path (relative to repo root)
FILES = {
    "GSM8K/Qwen(merged)": "newrundata/gsm8k_merged_flat.jsonl",
    "GSM8K/Qwen(test)":   "newrundata/gsm8k_test_flat.jsonl",
    "GSM8K/LLaMA":        "newrundata/gsm8k_llama_flat.jsonl",
    "CSQA/Qwen":          "newrundata/csqa_500_flat.jsonl",
    "CSQA/LLaMA":         "newrundata/csqa_llama_flat.jsonl",
    "SQA/Qwen":           "newrundata/strategyqa_flat.jsonl",
    "SQA/LLaMA":          "newrundata/strategyqa_llama_flat.jsonl",
}


def cot_text(d):
    c = d.get("cot")
    if isinstance(c, list):
        c = "\n".join(map(str, c))
    return str(c) if c is not None else ""


def stat_file(tok, path):
    lens = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ids = tok(cot_text(json.loads(line)),
                  add_special_tokens=True, truncation=False)["input_ids"]
        lens.append(len(ids))
    lens.sort()
    n = len(lens)
    return {
        "paths": n,
        "avg": sum(lens) / n,
        "median": lens[n // 2],
        "max": lens[-1],
        "over512": 100.0 * sum(1 for x in lens if x > 512) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_model", default=DEFAULT_BERT_MODEL)
    ap.add_argument("--files", nargs="*", default=None,
                    help="optional subset of keys from FILES")
    args = ap.parse_args()

    tok = BertTokenizerFast.from_pretrained(args.bert_model)

    keys = args.files if args.files else list(FILES.keys())
    print(f"{'File':22s} {'Paths':>7} {'Avg':>8} {'Median':>7} {'Max':>6} {'>512':>7}")
    for name in keys:
        path = FILES[name]
        if not os.path.exists(path):
            print(f"{name:22s}  MISSING ({path})")
            continue
        s = stat_file(tok, path)
        print(f"{name:22s} {s['paths']:7d} {s['avg']:8.1f} "
              f"{s['median']:7d} {s['max']:6d} {s['over512']:6.2f}%")


if __name__ == "__main__":
    main()
