# -*- coding: utf-8 -*-
"""Convert grouped JSON files to flat jsonl for D-ProtoCoT."""

import json
import os
import sys


def convert_grouped_to_flat(input_path: str, output_path: str):
    """
    Convert grouped JSON [{qid, question, ground_truth, paths: [{text, is_correct}]}]
    to flat jsonl [{raw_example, cot, gold_label, is_correct}].
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    n_questions = len(data)
    n_paths_total = 0
    n_correct = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for q in data:
            qid = q["qid"]
            question = q["question"]
            gold_label = q["ground_truth"]

            for p in q["paths"]:
                obj = {
                    "raw_example": {
                        "id": qid,
                        "question": question,
                        "label": gold_label,
                    },
                    "cot": p["text"],
                    "gold_label": gold_label,
                    "is_correct": int(bool(p["is_correct"])),
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_paths_total += 1
                if obj["is_correct"]:
                    n_correct += 1

    print(f"[convert] {input_path}")
    print(f"  questions={n_questions}  paths={n_paths_total}  "
          f"correct={n_correct} ({100*n_correct/n_paths_total:.1f}%)")
    print(f"  -> {output_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    files = [
        ("xiaorong/gsm8k_500_cot_qwen3.json", "gsm8k_500_flat.jsonl"),
        ("xiaorong/commonsenseqa_500_cot_qwen3.json", "csqa_500_flat.jsonl"),
    ]

    for src, dst in files:
        convert_grouped_to_flat(
            os.path.join(base, src),
            os.path.join(base, dst),
        )

    print("\nDone. Usage:")
    print("  python dprotocot/run.py main --data_path gsm8k_500_flat.jsonl --output_dir runs/gsm8k")
    print("  python dprotocot/run.py main --data_path csqa_500_flat.jsonl --output_dir runs/csqa")
