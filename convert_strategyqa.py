# -*- coding: utf-8 -*-
"""Convert StrategyQA grouped JSON to dprotocot flat jsonl format."""

import json
import os
import re


BASE = os.path.dirname(os.path.abspath(__file__))

SRC = os.path.join(BASE, "database/Icanuse/qwen/strategyqa_generated.json")
TRAIN = os.path.join(BASE, "database/StrategyQA/strategyqa_train.json")
OUT = os.path.join(BASE, "strategyqa_flat.jsonl")


def norm_q(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def extract_yes_no(predicted_answer: str, reasoning: str):
    """Determine if a StrategyQA path answers Yes or No.

    Tries predicted_answer first, then Final Answer markers, then
    the first declarative sentence after </think>.  Covers all the
    observed output formats of Qwen3-8B on StrategyQA.
    """
    pa = predicted_answer.strip().lower()

    # ---- step 1: clean predicted_answer -----------------------------------
    if pa in {"yes", "no", "true", "false"}:
        return pa in {"yes", "true"}
    first_word = pa.split()[0] if pa.split() else ""
    if first_word in {"yes", "no"}:
        return first_word == "yes"
    # "probably no", "no because ...", etc.
    m = re.search(r"\b(yes|no)\b", pa)
    if m:
        return m.group(1) == "yes"

    # ---- step 2: "Final Answer: Yes/No" (most reliable, ~56 % of paths) --
    m = re.search(r"\*\*Final\s+Answer\*?\*?\s*[:：]\s*(Yes|No)\b", reasoning, re.I)
    if m:
        return m.group(1).lower() == "yes"
    m = re.search(r"(?<!\*\*)Final\s+Answer\s*[:：]\s*(Yes|No)\b", reasoning, re.I)
    if m:
        return m.group(1).lower() == "yes"

    # ---- step 3: yes/no anywhere in the answer portion (post-</think>) ----
    if "</think>" in reasoning:
        post = reasoning.split("</think>")[-1].strip()
        if len(post) > 20:
            m = re.search(r"\b(yes|no)\b", post, re.I)
            if m:
                return m.group(1).lower() == "yes"
            # First sentence states the conclusion without explicit yes/no.
            # "The Beatles did not write …" / "Nickel can cause …" etc.
            first_sent = re.split(r"[.\n]", post)[0].strip().lower()
            if first_sent and not re.search(r"^(the\s+)?question\b", first_sent):
                neg_patterns = [
                    r"\bnot\b", r"\bno\b", r"\bneither\b", r"\bnor\b",
                    r"\bnever\b", r"\bcannot\b", r"\bdoesn'?t\b",
                    r"\bdon'?t\b", r"\bwon'?t\b", r"\bunlikely\b",
                ]
                for pat in neg_patterns:
                    if re.search(pat, first_sent):
                        return False  # negated → "No"

    # ---- step 4: broader search in whole reasoning ------------------------
    m = re.search(r"(?:the\s+)?answer\s+is\s+(yes|no)\b", reasoning, re.I)
    if m:
        return m.group(1).lower() == "yes"

    return None


def build_context(desc, facts):
    parts = []
    if desc:
        parts.append(desc)
    if facts:
        parts.append(" ".join(facts))
    return " ".join(parts) if parts else ""


def main():
    with open(TRAIN, encoding="utf-8") as f:
        train = json.load(f)
    with open(SRC, encoding="utf-8") as f:
        gen = json.load(f)

    # Build lookup: normalized question -> official info
    lookup = {}
    for q in train:
        lookup[norm_q(q["question"])] = q

    n_q, n_paths, n_correct, n_unmatched, n_ambiguous = 0, 0, 0, 0, 0
    with open(OUT, "w", encoding="utf-8") as out:
        for q in gen:
            nq = norm_q(q["question"])
            info = lookup.get(nq)
            if info is None:
                n_unmatched += 1
                continue

            qid = info["qid"]
            question = q["question"]
            gold = q["ground_truth"]  # "Yes" or "No"
            gold_bool = gold.strip().lower() == "yes"
            context = build_context(info.get("description", ""), info.get("facts", []))

            for p in q["paths"]:
                pred_bool = extract_yes_no(p["predicted_answer"], p["reasoning"])
                if pred_bool is None:
                    n_ambiguous += 1
                    is_correct = 0
                else:
                    is_correct = int(pred_bool == gold_bool)

                obj = {
                    "raw_example": {
                        "id": qid,
                        "question": question,
                        "context": context,
                        "label": gold,
                    },
                    "cot": p["reasoning"],
                    "gold_label": gold,
                    "is_correct": is_correct,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_paths += 1
                if is_correct:
                    n_correct += 1
            n_q += 1

    print(f"[convert] {SRC}")
    print(f"  questions={n_q}  paths={n_paths}  correct={n_correct} "
          f"({100*n_correct/max(1,n_paths):.1f}%)")
    if n_unmatched:
        print(f"  unmatched questions skipped: {n_unmatched}")
    if n_ambiguous:
        print(f"  ambiguous predicted_answers (treated as wrong): {n_ambiguous}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
