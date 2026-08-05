# -*- coding: utf-8 -*-
"""
Unified answer extraction + is_correct re-derivation for flat jsonl datasets.

Addresses Reviewer #2 concern #4/#6 root cause (progress ④f): the original
generation-time answer extractor was too narrow, so many paths that DO state an
answer were scored is_correct=0. This re-extracts the predicted answer from the
`cot` text with dataset-appropriate rules ("take the LAST committed answer",
multiple ending formats) and recomputes is_correct against `gold_label`.

DESIGN PRINCIPLE: this is a diagnostic-first tool. By default it only REPORTS
(does not modify files). Inspect the numbers, sanity-check them, and only then
re-run with --write. We intentionally do NOT tune the regex to hit any target
accuracy: whatever the principled extractor yields is what gets reported.

Usage:
    python newrun/fix_labels.py --report                 # diagnose all known files
    python newrun/fix_labels.py --file newrundata/csqa_500_flat.jsonl --dataset csqa --report
    python newrun/fix_labels.py --file ... --dataset ... --write   # overwrite in place (makes .bak)
"""
import argparse
import json
import os
import re
import shutil

# ----------------------------------------------------------------------------
# Extractors: each returns a normalized prediction string, or None if no
# definite answer could be located in the text.
# ----------------------------------------------------------------------------

_NUM = r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?"


def _to_float(s):
    s = s.replace("$", "").replace(",", "").replace(" ", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_gsm8k(cot):
    """Return the predicted numeric answer as a float, or None.

    Only a narrow, explicit answer cue is trusted. A "last number anywhere"
    fallback was tested and rejected: 31% of Qwen paths ramble *after* stating
    the answer ("...is 72. Let me verify, 48/2..."), so the trailing number is
    only ~52% correct, versus 93.5% for cue-stated answers. Paths with no
    explicit cue are treated as no-answer (None) rather than guessed.
    """
    cue = re.compile(
        r"(?:final answer|the answer is|answer\s*[:=]|####)\s*[:=]?\s*\**\s*(" + _NUM + r")",
        re.IGNORECASE,
    )
    matches = cue.findall(cot)
    if matches:
        return _to_float(matches[-1])
    return None


def extract_csqa(cot):
    """Return the predicted option letter A-E (uppercase), or None."""
    # Priority 1: answer-cue followed by a letter (take the LAST committed one).
    cue = re.compile(
        r"(?:final answer|the answer is|answer\s*[:=]|answer is|correct answer(?:\s+is)?(?:\s+likely)?|option|choice)"
        r"\s*[:=]?\s*\**\s*\(?\s*([A-E])\b",
        re.IGNORECASE,
    )
    m = cue.findall(cot)
    if m:
        return m[-1].upper()
    # Priority 2: a standalone letter on the last non-empty line (e.g. "Answer\nA").
    lines = [ln.strip() for ln in cot.splitlines() if ln.strip()]
    for ln in reversed(lines[-3:] if len(lines) >= 3 else lines):
        m2 = re.fullmatch(r"\(?([A-E])\)?[.)]?", ln)
        if m2:
            return m2.group(1).upper()
    return None


def extract_sqa(cot):
    """Return 'yes' or 'no', or None."""
    # Priority 1: answer-cue followed (possibly via "the final answer is") by yes/no.
    cue = re.compile(
        r"(?:final answer|the answer is|answer\s*[:=]|answer is)"
        r"\s*[:=]?\s*\**\s*(?:the final answer is\s*[:=]?\s*\**\s*)?(yes|no)\b",
        re.IGNORECASE,
    )
    m = cue.findall(cot)
    if m:
        return m[-1].lower()
    # Priority 2: leading yes/no of the concluding sentence.
    tail = cot[-300:]
    m2 = re.findall(r"\b(yes|no)\b", tail, re.IGNORECASE)
    if m2:
        return m2[-1].lower()
    return None


EXTRACTORS = {"gsm8k": extract_gsm8k, "csqa": extract_csqa, "sqa": extract_sqa}


def normalize_gold(dataset, gold):
    if dataset == "gsm8k":
        return _to_float(str(gold)) if not isinstance(gold, (int, float)) else float(gold)
    if dataset == "csqa":
        return str(gold).strip().upper()
    if dataset == "sqa":
        return str(gold).strip().lower()
    return gold


def is_match(dataset, pred, gold):
    if pred is None or gold is None:
        return False
    if dataset == "gsm8k":
        return abs(pred - gold) < 1e-3
    return pred == gold


# Guess dataset from filename if not given.
def guess_dataset(path):
    n = os.path.basename(path).lower()
    if "gsm8k" in n:
        return "gsm8k"
    if "csqa" in n or "commonsense" in n:
        return "csqa"
    if "strategy" in n or "sqa" in n:
        return "sqa"
    return None


def process(path, dataset, write=False):
    extractor = EXTRACTORS[dataset]
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    old_pos = none = new_pos = changed = 0
    correct_among_extracted = extracted = 0

    for r in rows:
        cot = r.get("cot", "")
        gold = normalize_gold(dataset, r.get("gold_label", r.get("raw_example", {}).get("label")))
        old = 1 if r.get("is_correct") in (True, 1) else 0
        old_pos += old

        pred = extractor(cot)
        if pred is None:
            none += 1
            new = 0
        else:
            extracted += 1
            new = 1 if is_match(dataset, pred, gold) else 0
            correct_among_extracted += new
        new_pos += new
        if new != old:
            changed += 1
        if write:
            r["pred_answer"] = pred if not isinstance(pred, float) else pred
            r["is_correct"] = bool(new)

    print(f"[{os.path.basename(path)}] dataset={dataset} n={n}")
    print(f"    old is_correct : {old_pos:5d} ({old_pos/n:.1%})")
    print(f"    new is_correct : {new_pos:5d} ({new_pos/n:.1%})   (changed {changed}, {changed/n:.1%})")
    print(f"    extracted ans  : {extracted:5d} ({extracted/n:.1%})   no-answer: {none} ({none/n:.1%})")
    if extracted:
        print(f"    acc | extracted: {correct_among_extracted:5d}/{extracted} ({correct_among_extracted/extracted:.1%})")

    if write:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f"    backup -> {bak}")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    WROTE new is_correct in place: {path}")
    print()


DEFAULT_FILES = [
    ("newrundata/gsm8k_merged_flat.jsonl", "gsm8k"),
    ("newrundata/gsm8k_test_flat.jsonl", "gsm8k"),
    ("newrundata/gsm8k_llama_flat.jsonl", "gsm8k"),
    ("newrundata/csqa_500_flat.jsonl", "csqa"),
    ("newrundata/csqa_llama_flat.jsonl", "csqa"),
    ("newrundata/strategyqa_flat.jsonl", "sqa"),
    ("newrundata/strategyqa_llama_flat.jsonl", "sqa"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="single flat jsonl to process")
    ap.add_argument("--dataset", choices=list(EXTRACTORS), help="gsm8k/csqa/sqa (guessed from name if omitted)")
    ap.add_argument("--report", action="store_true", help="diagnose only, do not modify")
    ap.add_argument("--write", action="store_true", help="overwrite is_correct in place (.bak kept)")
    args = ap.parse_args()

    if args.write and args.report:
        ap.error("choose either --report or --write, not both")
    write = args.write

    if args.file:
        ds = args.dataset or guess_dataset(args.file)
        if ds is None:
            ap.error("could not guess --dataset from filename; pass it explicitly")
        process(args.file, ds, write=write)
    else:
        for path, ds in DEFAULT_FILES:
            if os.path.exists(path):
                process(path, ds, write=write)
            else:
                print(f"[{path}] MISSING, skipped\n")


if __name__ == "__main__":
    main()
