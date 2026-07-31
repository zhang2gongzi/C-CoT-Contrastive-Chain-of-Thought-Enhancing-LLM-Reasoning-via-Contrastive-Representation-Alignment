# -*- coding: utf-8 -*-
"""
Reviewer #1, concern #3 rebuttal (with data).

Reviewer worry: selecting a path by similarity to the question favors SHALLOW
paths that merely echo the question wording, and PUNISHES deep paths that move
away from the prompt / introduce new ideas.

We show the opposite, on the real test set, using the TRAINED D-ProtoCoT encoder
(same pipeline as run.py main). Four measurements per question over its K paths:

  M1  Lexical overlap vs. selection
        - token Jaccard(path, question)  and  q_coverage = |P∩Q| / |Q|
        - if the SELECTED path is NOT higher-overlap than the pool average,
          the model is NOT picking paths that "repeat the question".

  M2  Reasoning depth vs. selection
        - depth proxies: #steps, #tokens, novel_ratio = |P\Q| / |P|
        - if the SELECTED path is NOT shorter / less-novel than unselected,
          selection does NOT punish deep reasoning.

  M3  Semantic similarity != lexical similarity
        - Pearson corr between (alignment cosine to the dynamic prototype)
          and (lexical Jaccard to the question), across all paths.
        - LOW correlation => the learned score tracks semantics, not wording.

  M4  (bonus, Reviewer #1 concern #2) AUC of alignment-cosine predicting path
        correctness. High AUC => "alignment ~ reasoning quality" is demonstrated,
        not merely asserted.

Usage (server, GPU), from the repo root:

  python newrun/analyze_reviewer1_q3.py \
      --train_path newrundata/gsm8k_merged_flat.jsonl \
      --test_path  newrundata/gsm8k_test_flat.jsonl \
      --epochs 10 --output newrun/reviewer1_q3_gsm8k.json

  # single-file ratio split:
  python newrun/analyze_reviewer1_q3.py \
      --data_path newrundata/strategyqa_flat.jsonl --use_context --epochs 10
"""

import os
import sys
import json
import math
import argparse

import torch
import torch.nn.functional as F

# make the dprotocot package importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_DPROTO = os.path.join(_HERE, "..", "baseline", "dprotocot")
sys.path.insert(0, os.path.abspath(_DPROTO))

from config import Config                       # noqa: E402
from data import load_splits, trainable_questions, question_text, path_text  # noqa: E402
from train import train_encoder                 # noqa: E402
from prototype import build_prototype           # noqa: E402


# --------------------------------------------------------------------------- #
# lightweight tokenization for lexical measures
# --------------------------------------------------------------------------- #
import re

_STOP = set("""a an the of to and or is are was were be been being in on at for with as by
that this these those it its from into than then so such but if while do does did have has
had you your we our they their he she his her i me my will would can could should may might
""".split())


def _tokens(text: str):
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in _STOP]


def _lexical(path_text_str: str, q_toks_set):
    p = _tokens(path_text_str)
    p_set = set(p)
    if not p_set:
        return 0.0, 0.0, 0.0, 0
    inter = len(p_set & q_toks_set)
    union = len(p_set | q_toks_set) or 1
    jaccard = inter / union
    q_cov = inter / (len(q_toks_set) or 1)          # how much of Q is echoed
    novel_ratio = len(p_set - q_toks_set) / len(p_set)  # fraction of NEW content
    return jaccard, q_cov, novel_ratio, len(p)


# --------------------------------------------------------------------------- #
# stats helpers (no sklearn dependency)
# --------------------------------------------------------------------------- #
def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _auc(scores, labels):
    """Mann-Whitney U based AUC. labels in {0,1}."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0             # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def build_cfg(args) -> Config:
    cfg = Config()
    for k, v in vars(args).items():
        if k in {"output"} or v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg.resolve()


# --------------------------------------------------------------------------- #
@torch.no_grad()
def analyze(cfg, test_groups, encoder):
    encoder.eval()

    # per-path accumulators
    sel_jac, sel_qcov, sel_nov, sel_steps, sel_len = [], [], [], [], []
    uns_jac, uns_qcov, uns_nov, uns_steps, uns_len = [], [], [], [], []
    all_align, all_jac, all_label = [], [], []
    n_sel_correct, n_q = 0, 0

    for g in test_groups:
        texts = [path_text(p["cot"], g, cfg) for p in g["paths"]]
        if len(texts) < 2:
            continue
        n_q += 1
        q_toks = set(_tokens(question_text(g, cfg)))

        z_q = encoder.encode_text_pooled(question_text(g, cfg))
        _, path_mat = encoder.encode_paths(texts)          # [K, H]
        prototype, _w = build_prototype(z_q, path_mat)
        proto = F.normalize(prototype, dim=-1)
        zp = F.normalize(path_mat, dim=-1)
        align = (zp @ proto).tolist()                      # cosine to prototype
        sel = int(max(range(len(align)), key=lambda i: align[i]))

        for i, p in enumerate(g["paths"]):
            jac, qcov, nov, plen = _lexical(texts[i], q_toks)
            n_steps = len([s for s in p["cot"].split(cfg.step_delimiter) if s.strip()])
            all_align.append(align[i]); all_jac.append(jac)
            all_label.append(int(p["is_correct"]))
            if i == sel:
                sel_jac.append(jac); sel_qcov.append(qcov); sel_nov.append(nov)
                sel_steps.append(n_steps); sel_len.append(plen)
            else:
                uns_jac.append(jac); uns_qcov.append(qcov); uns_nov.append(nov)
                uns_steps.append(n_steps); uns_len.append(plen)
        n_sel_correct += int(g["paths"][sel]["is_correct"])

    def _m(x):
        return sum(x) / len(x) if x else float("nan")

    report = {
        "n_test_questions": n_q,
        "selected_accuracy_pct": 100.0 * n_sel_correct / max(1, n_q),

        # M1: lexical overlap — selected should NOT exceed unselected
        "M1_lexical": {
            "selected_jaccard": _m(sel_jac),
            "unselected_jaccard": _m(uns_jac),
            "selected_q_coverage": _m(sel_qcov),
            "unselected_q_coverage": _m(uns_qcov),
        },
        # M2: depth — selected should NOT be shorter / less novel
        "M2_depth": {
            "selected_steps": _m(sel_steps),
            "unselected_steps": _m(uns_steps),
            "selected_tokens": _m(sel_len),
            "unselected_tokens": _m(uns_len),
            "selected_novel_ratio": _m(sel_nov),
            "unselected_novel_ratio": _m(uns_nov),
        },
        # M3: semantic != lexical — correlation should be LOW
        "M3_align_vs_lexical_pearson": _pearson(all_align, all_jac),
        # M4: alignment predicts correctness — AUC should be HIGH (>0.5)
        "M4_align_correctness_auc": _auc(all_align, all_label),
    }
    return report


def _pp(report):
    m1, m2 = report["M1_lexical"], report["M2_depth"]
    print("\n" + "=" * 68)
    print(f"Reviewer #1 Q3 analysis | test questions = {report['n_test_questions']}")
    print(f"D-ProtoCoT selected-path accuracy: {report['selected_accuracy_pct']:.2f}%")
    print("-" * 68)
    print("M1  lexical overlap (selected vs unselected) — lower/equal = NOT echoing Q")
    print(f"    Jaccard      selected={m1['selected_jaccard']:.4f}  "
          f"unselected={m1['unselected_jaccard']:.4f}")
    print(f"    Q-coverage   selected={m1['selected_q_coverage']:.4f}  "
          f"unselected={m1['unselected_q_coverage']:.4f}")
    print("M2  depth (selected vs unselected) — higher/equal = NOT punishing depth")
    print(f"    #steps       selected={m2['selected_steps']:.2f}  "
          f"unselected={m2['unselected_steps']:.2f}")
    print(f"    #tokens      selected={m2['selected_tokens']:.1f}  "
          f"unselected={m2['unselected_tokens']:.1f}")
    print(f"    novel_ratio  selected={m2['selected_novel_ratio']:.4f}  "
          f"unselected={m2['unselected_novel_ratio']:.4f}")
    print("M3  Pearson(align_cosine, lexical_jaccard) — LOW = semantic, not wording")
    print(f"    r = {report['M3_align_vs_lexical_pearson']:.4f}")
    print("M4  AUC(align_cosine -> path correctness) — HIGH(>0.5) = alignment~quality")
    print(f"    AUC = {report['M4_align_correctness_auc']:.4f}")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_model")
    ap.add_argument("--data_path")
    ap.add_argument("--train_path")
    ap.add_argument("--test_path")
    ap.add_argument("--use_context", action="store_true", default=None)
    ap.add_argument("--subset_questions", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--output", default="newrun/reviewer1_q3_report.json")
    args = ap.parse_args()

    cfg = build_cfg(args)
    train_g, val_g, test_g = load_splits(cfg)
    train_t = trainable_questions(train_g)
    val_t = trainable_questions(val_g)

    print(f"[analyze] training encoder (epochs={cfg.epochs}) ...")
    enc = train_encoder(cfg, train_t, val_t)

    report = analyze(cfg, test_g, enc)
    _pp(report)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[analyze] saved -> {args.output}")


if __name__ == "__main__":
    main()
