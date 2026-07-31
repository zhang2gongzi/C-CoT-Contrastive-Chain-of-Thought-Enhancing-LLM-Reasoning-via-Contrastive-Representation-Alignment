# -*- coding: utf-8 -*-
"""
Reviewer #1, concern #2 — the post-training counterpart to Figure 2 (fig:tsne),
plus a quantitative separability number.

The current Figure 2 shows FROZEN BERT embeddings of ~100 StrategyQA paths:
correct (green) and incorrect (red) are heavily intermixed, i.e. surface
semantics alone cannot separate reasoning quality.

Reviewer #1 asks us to DEMONSTRATE that contrastive alignment fixes this, not
just assert it. This script encodes the SAME 100 paths with:
    (a) the untrained encoder   -> "Before alignment"  (matches Figure 2)
    (b) the trained D-ProtoCoT  -> "After alignment"
projects both with t-SNE, and reports a separability metric for each panel:
leave-one-out 1-NN accuracy in cosine space (chance = majority class).
Rising 1-NN accuracy = correct/incorrect paths become linearly separable =
alignment tracks reasoning quality.

Both panels use the SAME architecture (path-level mean-pooled embedding), so the
only difference is the contrastive training — an apples-to-apples before/after.

Usage (server, GPU), from repo root:

  python newrun/viz_reviewer1_q2.py \
      --train_path newrundata/strategyqa_flat.jsonl --use_context \
      --csv cot100.csv --epochs 10 \
      --out_png newrun/tsne_cot100_after.png

Dependencies (same as tsne.py): pandas, numpy, matplotlib, openTSNE, sklearn.
"""

import os
import sys
import re
import argparse

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "baseline", "dprotocot")))

from config import Config                       # noqa: E402
from data import load_splits, trainable_questions  # noqa: E402
from train import train_encoder                 # noqa: E402
from encoder import MultiGranularEncoder        # noqa: E402


def extract_clean_reasoning(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    m = re.search(r"(Step\s+1[:.\s].*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(lines[-5:]) if lines else ""


@torch.no_grad()
def encode_all(encoder, texts):
    encoder.eval()
    embs = []
    for t in texts:
        _, p = encoder.encode_path(t)          # path-level mean-pooled [H]
        embs.append(p.detach().cpu().numpy())
    return np.array(embs)


def loo_1nn_cosine(X, y):
    """Leave-one-out 1-NN accuracy in cosine space (separability proxy)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(axis=1)
    return float((y[nn] == y).mean())


def tsne2d(X):
    from openTSNE import TSNE
    from sklearn.decomposition import PCA
    n = X.shape[0]
    n_pca = min(50, n - 1, X.shape[1])
    Xp = PCA(n_components=n_pca, random_state=42).fit_transform(X)
    perp = min(30, n - 1)
    return np.asarray(TSNE(perplexity=perp, n_jobs=-1, random_state=42).fit(Xp))


def build_cfg(args) -> Config:
    cfg = Config()
    for k, v in vars(args).items():
        if v is None or k in {"csv", "out_png", "out_pdf"}:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_model")
    ap.add_argument("--data_path")
    ap.add_argument("--train_path")
    ap.add_argument("--test_path")
    ap.add_argument("--use_context", action="store_true", default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--csv", default="cot100.csv")
    ap.add_argument("--out_png", default="newrun/tsne_cot100_after.png")
    ap.add_argument("--out_pdf", default="newrun/tsne_cot100_after.pdf")
    args = ap.parse_args()

    cfg = build_cfg(args)

    # ---- load the same 100 paths ----
    df = pd.read_csv(args.csv)
    texts, labels = [], []
    for _, row in df.iterrows():
        clean = extract_clean_reasoning(row.get("model_reasoning", ""))
        if len(clean) >= 20 and "Step" in clean:
            texts.append(clean)
            labels.append(bool(row.get("is_correct", False)))
    y = np.array(labels)
    print(f"[viz] valid paths: {len(texts)}  (correct={int(y.sum())}, incorrect={int((~y).sum())})")

    # ---- BEFORE: untrained encoder (same architecture) ----
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    before_enc = MultiGranularEncoder(cfg).to(device)
    X_before = encode_all(before_enc, texts)
    del before_enc

    # ---- AFTER: trained D-ProtoCoT encoder ----
    train_g, val_g, _ = load_splits(cfg)
    train_t = trainable_questions(train_g)
    val_t = trainable_questions(val_g)
    print(f"[viz] training encoder (epochs={cfg.epochs}) ...")
    trained = train_encoder(cfg, train_t, val_t)
    X_after = encode_all(trained, texts)

    acc_before = loo_1nn_cosine(X_before, y)
    acc_after = loo_1nn_cosine(X_after, y)
    majority = max(y.mean(), 1 - y.mean())
    print("\n" + "=" * 60)
    print("Separability (leave-one-out 1-NN accuracy, cosine)")
    print(f"  majority-class baseline : {majority:.3f}")
    print(f"  BEFORE alignment        : {acc_before:.3f}")
    print(f"  AFTER  alignment        : {acc_after:.3f}")
    print("=" * 60)

    # ---- t-SNE both ----
    T_before = tsne2d(X_before)
    T_after = tsne2d(X_after)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "font.family": "serif"})
    colors = ["#2E8B57" if c else "#DC143C" for c in y]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, T, title, acc in [
        (axes[0], T_before, f"Before alignment (1-NN acc={acc_before:.2f})", acc_before),
        (axes[1], T_after, f"After alignment (1-NN acc={acc_after:.2f})", acc_after),
    ]:
        ax.scatter(T[:, 0], T[:, 1], c=colors, s=25, alpha=0.75, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].legend(handles=[Patch(facecolor="#2E8B57", label="Correct"),
                            Patch(facecolor="#DC143C", label="Incorrect")],
                   loc="upper right", framealpha=0.9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, dpi=600, bbox_inches="tight")
    plt.savefig(args.out_pdf, dpi=600, bbox_inches="tight", format="pdf")
    print(f"[viz] saved -> {args.out_png} , {args.out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
