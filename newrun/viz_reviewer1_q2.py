# -*- coding: utf-8 -*-
"""
Reviewer #1 Q2 — post-training counterpart to Figure 2, with the CORRECT
before/after metric: AUC of (path <-> PROTOTYPE alignment) predicting
correctness.  Prototype (not the raw question) is what D-ProtoCoT actually
selects by, so this is the mechanism-faithful measure (same anchor as M4).

Guardrail: run ONCE. If after proto-AUC is not clearly above before,
ABANDON the before/after comparison and report M4 + after-only t-SNE instead.
"""
import os, sys, re, argparse, collections
import numpy as np, pandas as pd
import torch, torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "baseline", "dprotocot")))
from config import Config                                   # noqa: E402
from data import load_splits, trainable_questions           # noqa: E402
from train import train_encoder                             # noqa: E402
from encoder import MultiGranularEncoder                    # noqa: E402
from prototype import build_prototype                       # noqa: E402


def extract_clean_reasoning(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    m = re.search(r"(Step\s+1[:.\s].*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(lines[-5:]) if lines else ""


@torch.no_grad()
def encode_paths_tensor(encoder, texts):
    encoder.eval()
    _, path_mat = encoder.encode_paths(texts)        # [K, H] tensor on device
    return path_mat


@torch.no_grad()
def encode_q_tensor(encoder, q_text):
    encoder.eval()
    return encoder.encode_text_pooled(q_text if q_text.strip() else " ")  # [H]


def loo_1nn_cosine(X, y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf)
    return float((y[S.argmax(axis=1)] == y).mean())


def _auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (sum(r for r, l in zip(ranks, labels) if l == 1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def proto_align_auc(encoder, texts, q_texts, y_int):
    """Per-question prototype alignment -> correctness AUC (mechanism-faithful)."""
    X = encode_paths_tensor(encoder, texts)               # [K, H] on device
    groups = collections.OrderedDict()
    for i, qt in enumerate(q_texts):
        groups.setdefault(qt, []).append(i)
    align = [0.0] * len(texts)
    for qt, idxs in groups.items():
        zq = encode_q_tensor(encoder, qt)                 # [H] on device
        pm = X[idxs]                                      # slice, same device
        proto, _ = build_prototype(zq, pm)                # [H]
        pn = F.normalize(proto, dim=-1)
        pmn = F.normalize(pm, dim=-1)
        a = (pmn @ pn).tolist()                           # [|idxs|]
        for j, ii in enumerate(idxs):
            align[ii] = a[j]
    return _auc(align, y_int), X.cpu().numpy()


def tsne2d(X):
    from openTSNE import TSNE
    from sklearn.decomposition import PCA
    n = X.shape[0]
    Xp = PCA(n_components=min(50, n - 1, X.shape[1]), random_state=42).fit_transform(X)
    return np.asarray(TSNE(perplexity=min(30, n - 1), n_jobs=-1, random_state=42).fit(Xp))


def build_cfg(args):
    cfg = Config()
    for k, v in vars(args).items():
        if v is None or k in {"csv", "out_png", "out_pdf"}:
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
    ap.add_argument("--csv", default="cot100.csv")
    ap.add_argument("--out_png", default="newrun/tsne_cot100_protoAUC.png")
    ap.add_argument("--out_pdf", default="newrun/tsne_cot100_protoAUC.pdf")
    args = ap.parse_args()
    cfg = build_cfg(args)

    df = pd.read_csv(args.csv)
    q_col = next((c for c in df.columns if str(c).strip().lower() == "question"), None)
    if q_col is None:
        print("[viz][ERROR] no question column:", df.columns.tolist()); sys.exit(1)
    texts, labels, q_texts = [], [], []
    for _, row in df.iterrows():
        clean = extract_clean_reasoning(row.get("model_reasoning", ""))
        if len(clean) >= 20 and "Step" in clean:
            texts.append(clean); labels.append(bool(row.get("is_correct", False)))
            q_texts.append(str(row.get(q_col, "")))
    y = np.array(labels); y_int = y.astype(int).tolist()
    nq = len(set(q_texts))
    print(f"[viz] valid paths={len(texts)} correct={int(y.sum())} unique_questions={nq}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    before_enc = MultiGranularEncoder(cfg).to(device)
    auc_before, X_before = proto_align_auc(before_enc, texts, q_texts, y_int)
    del before_enc

    train_g, val_g, _ = load_splits(cfg)
    print(f"[viz] training encoder (epochs={cfg.epochs}) ...")
    trained = train_encoder(cfg, trainable_questions(train_g), trainable_questions(val_g))
    auc_after, X_after = proto_align_auc(trained, texts, q_texts, y_int)

    print("\n" + "=" * 60)
    print("Prototype-alignment AUC (path<->prototype -> correctness) [KEY]")
    print(f"  BEFORE alignment : {auc_before:.3f}")
    print(f"  AFTER  alignment : {auc_after:.3f}   (delta = {auc_after - auc_before:+.3f})")
    print("1-NN (reference)  BEFORE=%.3f AFTER=%.3f" % (loo_1nn_cosine(X_before, y), loo_1nn_cosine(X_after, y)))
    print("=" * 60)

    T_before, T_after = tsne2d(X_before), tsne2d(X_after)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 11})
    colors = ["#2E8B57" if c else "#DC143C" for c in y]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, T, title in [
        (axes[0], T_before, f"Before alignment (proto-AUC={auc_before:.2f})"),
        (axes[1], T_after,  f"After alignment (proto-AUC={auc_after:.2f})")]:
        ax.scatter(T[:, 0], T[:, 1], c=colors, s=25, alpha=0.75, edgecolors="none")
        ax.set_title(title); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    axes[1].legend(handles=[Patch(facecolor="#2E8B57", label="Correct"),
                            Patch(facecolor="#DC143C", label="Incorrect")], loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, dpi=600, bbox_inches="tight")
    plt.savefig(args.out_pdf, dpi=600, bbox_inches="tight", format="pdf")
    print(f"[viz] saved -> {args.out_png}")
    plt.close()


if __name__ == "__main__":
    main()
