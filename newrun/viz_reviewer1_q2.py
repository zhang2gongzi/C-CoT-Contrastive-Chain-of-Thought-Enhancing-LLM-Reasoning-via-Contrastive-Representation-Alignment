# -*- coding: utf-8 -*-
"""
Reviewer #1 Q2 — post-training counterpart to Figure 2.

This is the MECHANISM-FAITHFUL, K=10 version. Unlike the earlier cot100.csv
variant (~2 paths/question, which makes the per-question dynamic prototype
degenerate), this reads the SAME standard K=10 flat jsonl used to produce
reviewer1_q3_gsm8k.json, so the figure and the M4 AUC are computed on identical
data.

What it does:
  * trains the D-ProtoCoT encoder on --train_path (same pipeline as run.py),
  * embeds every path of every test question (--test_path, K=10 per question),
  * draws an AFTER-alignment t-SNE of all path embeddings, colored by
    correctness (green=correct, red=incorrect),
  * prints the prototype-alignment AUC (path <-> per-question dynamic prototype
    -> correctness) as a sanity check; it should match M4 in the analysis JSON.

Per the agreed rebuttal strategy: AFTER-only figure, NO numbers in the title
(M4 is reported in text, not baked into the plot). No before/after comparison.

Usage (server, GPU), from repo root:

  python newrun/viz_reviewer1_q2.py \
      --train_path newrundata/gsm8k_merged_flat.jsonl \
      --test_path  newrundata/gsm8k_test_flat.jsonl \
      --epochs 10 \
      --out_png newrun/tsne_gsm8k_after.png

       python newrun/viz_reviewer1_q2.py \
      --before_only \
      --bert_model /home2/zzl/model/bert-base-uncased \
      --train_path newrundata/gsm8k_merged_flat.jsonl \
      --test_path newrundata/gsm8k_test_flat.jsonl \
      --out_png newrun/tsne_gsm8k_before.png \
      --out_pdf newrun/tsne_gsm8k_before.pdf
"""
import os
import sys
import argparse
import collections

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "baseline", "dprotocot")))
from config import Config                                   # noqa: E402
from data import load_splits, trainable_questions, question_text, path_text  # noqa: E402
from train import train_encoder                             # noqa: E402
from encoder import MultiGranularEncoder                    # noqa: E402
from prototype import build_prototype                       # noqa: E402


def _auc(scores, labels):
    """Mann-Whitney U based AUC. labels in {0,1}. No sklearn."""
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
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def embed_test(encoder, cfg, test_groups):
    """Embed all paths of all test questions.

    Returns X [N,H] (path embeddings), y [N] (is_correct int),
    and per-path prototype-alignment cosine (mechanism-faithful, M4 anchor).
    """
    encoder.eval()
    X, y, align = [], [], []
    for g in test_groups:
        texts = [path_text(p["cot"], g, cfg) for p in g["paths"]]
        if len(texts) < 2:
            continue
        z_q = encoder.encode_text_pooled(question_text(g, cfg))
        _, path_mat = encoder.encode_paths(texts)           # [K, H]
        proto, _ = build_prototype(z_q, path_mat)           # [H]
        pn = F.normalize(proto, dim=-1)
        pmn = F.normalize(path_mat, dim=-1)
        a = (pmn @ pn).tolist()                             # [K]
        pm = path_mat.cpu().numpy()
        for i, p in enumerate(g["paths"]):
            X.append(pm[i]); y.append(int(p["is_correct"])); align.append(a[i])
    return np.asarray(X), np.asarray(y), align


def tsne2d(X):
    from openTSNE import TSNE
    from sklearn.decomposition import PCA
    n = X.shape[0]
    Xp = PCA(n_components=min(50, n - 1, X.shape[1]), random_state=42).fit_transform(X)
    return np.asarray(TSNE(perplexity=min(30, n - 1), n_jobs=-1, random_state=42).fit(Xp))


def build_cfg(args):
    cfg = Config()
    for k, v in vars(args).items():
        if v is None or k in {"out_png", "out_pdf"}:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg.resolve()


def main():
    ap = argparse.ArgumentParser()
    for a in ["--bert_model", "--data_path", "--train_path", "--test_path",
              "--epochs", "--seed", "--device"]:
        ap.add_argument(a, type=int if a in ("--epochs", "--seed") else None)
    ap.add_argument("--use_context", action="store_true", default=None)
    ap.add_argument("--before_only", action="store_true",
                    help="Use an UNTRAINED encoder (before alignment); same paths/projection as after.")
    ap.add_argument("--out_png", default="newrun/tsne_gsm8k_after.png")
    ap.add_argument("--out_pdf", default="newrun/tsne_gsm8k_after.pdf")
    args = ap.parse_args()
    cfg = build_cfg(args)

    train_g, val_g, test_g = load_splits(cfg)
    if args.before_only:
        import torch as _torch
        device = _torch.device(cfg.device if _torch.cuda.is_available() else "cpu")
        print("[viz] BEFORE mode: using an UNTRAINED encoder (no contrastive alignment).")
        encoder = MultiGranularEncoder(cfg).to(device)
    else:
        print(f"[viz] training encoder (epochs={cfg.epochs}) ...")
        encoder = train_encoder(cfg, trainable_questions(train_g), trainable_questions(val_g))

    X, y, align = embed_test(encoder, cfg, test_g)
    auc = _auc(align, y.tolist())
    print("\n" + "=" * 60)
    print(f"[viz] test paths = {len(y)}  correct = {int(y.sum())}  "
          f"incorrect = {int((1 - y).sum())}")
    print(f"[viz] prototype-alignment AUC (sanity vs M4): {auc:.4f}")
    print("=" * 60)

    T = tsne2d(X)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12})
    colors = ["#2E8B57" if c else "#DC143C" for c in y]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(T[:, 0], T[:, 1], c=colors, s=18, alpha=0.7, edgecolors="none")
    ax.set_title("Before contrastive alignment" if args.before_only
                 else "After contrastive alignment")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.legend(handles=[Patch(facecolor="#2E8B57", label="Correct"),
                       Patch(facecolor="#DC143C", label="Incorrect")],
              loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, dpi=600, bbox_inches="tight")
    plt.savefig(args.out_pdf, dpi=600, bbox_inches="tight", format="pdf")
    print(f"[viz] saved -> {args.out_png}")
    plt.close()


if __name__ == "__main__":
    main()
