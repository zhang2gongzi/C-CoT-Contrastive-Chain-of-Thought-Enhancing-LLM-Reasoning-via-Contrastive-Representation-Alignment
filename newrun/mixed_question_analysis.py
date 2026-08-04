# -*- coding: utf-8 -*-
"""
Mixed-question analysis for Reviewer #3 Q1 (larger-model saturation).

Motivation
----------
On a strong model (e.g. Qwen3-14B on GSM8K) the base accuracy saturates: most
questions have K sampled paths that are ALL correct, so there is nothing to
select and every method converges near the ceiling. D-ProtoCoT -- like any
path-selection method -- can only help on questions whose sampled paths
DISAGREE (some correct, some wrong). This script quantifies that boundary.

It partitions the test set into:
  * mixed      : the K paths contain BOTH correct and incorrect ones
                 (there IS something to select; the method can act)
  * unanimous  : all K paths agree (all correct or all wrong; no headroom)

and reports the mixed fraction, the per-path base accuracy (how saturated the
model is), and every method's accuracy on the FULL set and on each subset.
The intended story for the rebuttal: as the base model strengthens the mixed
fraction collapses and full-set gains shrink, yet on the MIXED subset
D-ProtoCoT still matches or beats Self-Consistency -- the method helps exactly
where paths disagree.

Usage (server / AutoDL, same data + encoder settings as `run.py main`):

  python newrun/mixed_question_analysis.py \
      --bert_model /root/autodl-tmp/bert-base-uncased \
      --train_path gsm8k_train_14b_flat.jsonl \
      --test_path  gsm8k_test_14b_flat.jsonl \
      --epochs 10
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "baseline", "dprotocot"))

from config import Config                                    # noqa: E402
from data import load_splits, trainable_questions            # noqa: E402
from train import train_encoder                              # noqa: E402
from evaluate import evaluate_all, print_results             # noqa: E402


def build_cfg(args) -> Config:
    cfg = Config()
    for k, v in vars(args).items():
        if v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg.resolve()


def is_mixed(group) -> bool:
    """True iff the group's sampled paths contain both correct and incorrect."""
    labs = [int(p["is_correct"]) for p in group["paths"]]
    return 0 < sum(labs) < len(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_model")
    ap.add_argument("--train_path")
    ap.add_argument("--test_path")
    ap.add_argument("--data_path")
    ap.add_argument("--use_context", action="store_true", default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--input_mode", choices=["full", "mask", "qa_only"])
    args = ap.parse_args()

    cfg = build_cfg(args)
    train_g, val_g, test_g = load_splits(cfg)

    n = len(test_g)
    mixed = [g for g in test_g if is_mixed(g)]
    unan = [g for g in test_g if not is_mixed(g)]
    frac = 100.0 * len(mixed) / max(1, n)

    tot = sum(len(g["paths"]) for g in test_g)
    corr = sum(int(p["is_correct"]) for g in test_g for p in g["paths"])

    print(f"\n==== Mixed-question breakdown | test questions = {n} ====")
    print(f"  per-path base accuracy            : {100.0 * corr / max(1, tot):6.2f}%"
          f"   ({corr}/{tot} paths)   <- saturation level")
    print(f"  mixed (correct+incorrect present) : {len(mixed):4d}/{n} = {frac:5.1f}%"
          f"   <- where selection can act")
    print(f"  unanimous (all paths agree)       : {len(unan):4d}/{n} = {100 - frac:5.1f}%")

    enc = train_encoder(cfg, trainable_questions(train_g), trainable_questions(val_g))

    print_results("FULL test", evaluate_all(cfg, test_g, enc))
    if mixed:
        print_results(f"MIXED subset (n={len(mixed)})", evaluate_all(cfg, mixed, enc))
    if unan:
        print_results(f"UNANIMOUS subset (n={len(unan)})", evaluate_all(cfg, unan, enc))

    print("\n[read] On the FULL set gains are compressed by saturation; the MIXED "
          "subset is where D-ProtoCoT should still be >= Self-Consistency.")


if __name__ == "__main__":
    main()
