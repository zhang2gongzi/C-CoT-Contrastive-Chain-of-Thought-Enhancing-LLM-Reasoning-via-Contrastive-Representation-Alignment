# -*- coding: utf-8 -*-
"""Merge two GSM8K datasets into one dprotocot flat jsonl (~911 unique questions)."""

import json
import os
import re
import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))


def norm_q(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def extract_gold_answer(answer_text: str):
    """Extract final number from GSM8K gold answer (format: ... #### N)."""
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,\.]+", answer_text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return str(pred).strip().lower() == str(gold).strip().lower()


def convert_xiaorong(data, skip_qids):
    """xiaorong data already has is_correct. Return flat jsonl lines."""
    lines = []
    for q in data:
        qid = q["qid"]
        if qid in skip_qids:
            continue
        for p in q["paths"]:
            lines.append(json.dumps({
                "raw_example": {
                    "id": qid,
                    "question": q["question"],
                    "label": q["ground_truth"],
                },
                "cot": p["text"],
                "gold_label": q["ground_truth"],
                "is_correct": int(bool(p["is_correct"])),
            }, ensure_ascii=False))
    return lines


def convert_database(data, skip_qs, train_qs, db_q_to_qid):
    """database/Icanuse data needs is_correct computed. Return flat jsonl lines."""
    lines = []
    for q in data:
        nq = norm_q(q["question"])
        if nq in skip_qs:
            continue

        gt_raw = q["ground_truth"]
        gold = extract_gold_answer(gt_raw)

        # assign a unique qid (use train parquet to look up)
        qid = db_q_to_qid.get(nq, f"gsm8k_db_{len(lines)}")

        for p in q["paths"]:
            pred = str(p["predicted_answer"]).strip()
            # clean predicted answer
            pred_clean = pred.replace("boxed", "").replace("\\boxed", "").replace("{", "").replace("}", "").strip()
            is_correct = int(answers_match(pred_clean, gold))

            lines.append(json.dumps({
                "raw_example": {
                    "id": qid,
                    "question": q["question"],
                    "label": gold,
                },
                "cot": p["reasoning"],
                "gold_label": gold,
                "is_correct": is_correct,
            }, ensure_ascii=False))
    return lines


def main():
    # Load data
    with open(os.path.join(BASE, "xiaorong/gsm8k_500_cot_qwen3.json"), encoding="utf-8") as f:
        xiaorong = json.load(f)
    with open(os.path.join(BASE, "database/Icanuse/qwen/gsm8k_generated.json"), encoding="utf-8") as f:
        db = json.load(f)

    # Build a map from normalized question text -> official GSM8K train qid
    df_train = pd.read_parquet(os.path.join(BASE, "database/gsm8k/train-00000-of-00001.parquet"))
    db_q_to_qid = {}
    for i, row in df_train.iterrows():
        db_q_to_qid[norm_q(row["question"])] = f"gsm8k_train_{i}"

    # Find overlapping questions
    xiao_qs = {norm_q(q["question"]) for q in xiaorong}
    db_qs = {norm_q(q["question"]) for q in db}
    overlap = xiao_qs & db_qs

    # Prefer xiaorong for overlap → skip those from db
    xiao_lines = convert_xiaorong(xiaorong, skip_qids=set())  # keep all xiaorong
    db_lines = convert_database(db, skip_qs=overlap, train_qs=xiao_qs, db_q_to_qid=db_q_to_qid)

    all_lines = xiao_lines + db_lines
    out_path = os.path.join(BASE, "gsm8k_merged_flat.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    # Stats
    qids = set()
    n_correct = 0
    for line in all_lines:
        d = json.loads(line)
        qids.add(d["raw_example"]["id"])
        if d["is_correct"]:
            n_correct += 1

    print(f"Merged: {len(qids)} unique questions, {len(all_lines)} total paths")
    print(f"Correct: {n_correct}/{len(all_lines)} ({100*n_correct/len(all_lines):.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
