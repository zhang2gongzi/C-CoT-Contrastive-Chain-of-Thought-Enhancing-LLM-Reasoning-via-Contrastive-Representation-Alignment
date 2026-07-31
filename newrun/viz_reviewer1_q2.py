# -*- coding: utf-8 -*-
"""
Reviewer #1, concern #2 — post-training counterpart to Figure 2 (fig:tsne),
PLUS the quantitative measure the reviewer explicitly asked for:
the AUC of (alignment cosine  path<->question) predicting path correctness,
reported BEFORE vs AFTER contrastive training.

Why align-AUC (not 1-NN) is the right before/after metric:
  * 1-NN compares paths to *each other* -> frozen BERT already separates them
    by surface features (ceiling effect), so before/after looks flat.
  * align-AUC compares each path to its *question* -> in frozen BERT,
    "looks like the question" != "is correct"  =>  AUC ~ 0.5  (this IS the
    original Figure-2 claim, now with a number). After contrastive training,
    alignment-to-question becomes a reliable quality proxy  =>  AUC >> 0.5.
  This directly matches the reviewer's requested AUC AND the D-ProtoCoT
  mechanism, so the t-SNE panels hang on a logically closed loop.

Both panels use the SAME architecture (path-level mean-pooled embedding); the
only difference is the contrastive training.

Usage (server, GPU), from repo root:
  python newrun/viz_reviewer1_q2.py \
      --data_path newrundata/strategyqa_flat.jsonl --use_context \
      --csv cot100.csv --epochs 10 \
      --out_png newrun/tsne_cot100_after.png
"""

import os
import sys
import re
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


@torch.no_grad()
def encode_texts_pooled(encoder, q_texts):
    """Encode a list of question strings -> [K, H] numpy (same weight space as
    the paths encoded by the same encoder)."""
    encoder.eval()
    embs = []
    for qt in q_texts:
        safe = qt if (qt and qt.strip()) else " "
        v = encoder.encode_text_pooled(safe)   # tensor [H]
        embs.append(torch.as_tensor(v, dtype=torch.float32).view(-1).cpu().numpy())
    return np.stack(embs, axis=0)


def loo_1nn_cosine(X, y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(axis=1)
    return float((y[nn] == y).mean())


def _auc(scores, labels):
    """Mann-Whitney U AUC, labels in {0,1}."""
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


def alignment_auc(Q, P, labels):
    """Per-row cosine(Q[i], P[i]) -> correctness AUC.
    Q, P : [K, H]. Works for both a single shared question (rows identical)
    and per-row questions."""
    Q = torch.as_tensor(Q, dtype=torch.float32)
    P = torch.as_tensor(P, dtype=torch.float32)
    qn = F.normalize(Q, dim=-1)
    pn = F.normalize(P, dim=-1)
    align = (pn * qn).sum(dim=-1).tolist()      # [K] per-row cosine
    return _auc(align, labels)


def _find_question_col(df):
    cols = list(df.columns)
    low = {c: str(c).strip().lower() for c in cols}
    for c in cols:
        if low[c] == "question":
            return c
    for c in cols:
        if ("question" in low[c]) or ("prompt" in low[c]):
            return c
    for c in cols:
        if low[c] in ("q", "raw", "raw_question", "input", "query"):
            return c
    return None


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

    # ---- load the same 100 paths (+ their question, for the align-AUC) ----
    df = pd.read_csv(args.csv)
    print(f"[viz] csv columns: {df.columns.tolist()}")
    q_col = _find_question_col(df)
    if q_col is None:
        print("[viz][ERROR] 找不到 question 列！对齐-AUC 需要 question 文本作为锚点。")
        print("[viz][ERROR] 请把上面打印的 csv columns 列表发给助手，30 秒即可修好。")
        sys.exit(1)
    print(f"[viz] using question column: {q_col!r}")

    texts, labels, q_texts = [], [], []
    for _, row in df.iterrows():
        clean = extract_clean_reasoning(row.get("model_reasoning", ""))
        if len(clean) >= 20 and "Step" in clean:
            texts.append(clean)
            labels.append(bool(row.get("is_correct", False)))
            q_texts.append(str(row.get(q_col, "")))
    y = np.array(labels)
    y_int = y.astype(int).tolist()
    if not any(qt.strip() for qt in q_texts):
        print(f"[viz][ERROR] question 列 {q_col!r} 全为空，无法计算对齐-AUC。请发列名给助手。")
        sys.exit(1)
    print(f"[viz] valid paths: {len(texts)}  (correct={int(y.sum())}, incorrect={int((~y).sum())})")

    # ---- BEFORE: untrained encoder (same architecture) ----
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    before_enc = MultiGranularEncoder(cfg).to(device)
    X_before = encode_all(before_enc, texts)
    Q_before = encode_texts_pooled(before_enc, q_texts)   # same frozen weights
    del before_enc

    # ---- AFTER: trained D-ProtoCoT encoder ----
    train_g, val_g, _ = load_splits(cfg)
    train_t = trainable_questions(train_g)
    val_t = trainable_questions(val_g)
    print(f"[viz] training encoder (epochs={cfg.epochs}) ...")
    trained = train_encoder(cfg, train_t, val_t)
    X_after = encode_all(trained, texts)
    Q_after = encode_texts_pooled(trained, q_texts)       # same trained weights

    acc_before = loo_1nn_cosine(X_before, y)
    acc_after = loo_1nn_cosine(X_after, y)
    auc_before = alignment_auc(Q_before, X_before, y_int)
    auc_after = alignment_auc(Q_after, X_after, y_int)
    majority = max(y.mean(), 1 - y.mean())
    print("\n" + "=" * 60)
    print("Separability (leave-one-out 1-NN, cosine)  [reference only]")
    print(f"  majority-class baseline : {majority:.3f}")
    print(f"  BEFORE alignment        : {acc_before:.3f}")
    print(f"  AFTER  alignment        : {acc_after:.3f}")
    print("-" * 60)
    print("Alignment-AUC (cosine path<->question -> correctness)  [KEY]")
    print(f"  BEFORE alignment        : {auc_before:.3f}   (frozen BERT; expect ~0.5)")
    print(f"  AFTER  alignment        : {auc_after:.3f}    (trained; expect >> 0.5)")
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
        (axes[0], T_before, f"Before alignment (align-AUC={auc_before:.2f})", acc_before),
        (axes[1], T_after, f"After alignment (align-AUC={auc_after:.2f})", acc_after),
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
