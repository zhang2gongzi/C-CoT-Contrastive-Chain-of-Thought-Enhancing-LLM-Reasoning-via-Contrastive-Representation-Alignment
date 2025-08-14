import os, json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# 可选 UMAP（环境没有就自动跳过）
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

REPR_DIR = "/home2/zzl/C-CoT/repr_cache_qwen7b_clear"     # 与上一步一致
SEQ_PATH = os.path.join(REPR_DIR, "seq_vecs.npy")
META_PATH = os.path.join(REPR_DIR, "meta.jsonl")
OUT_DIR = os.path.join(REPR_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 读取向量与元信息 =====
X = np.load(SEQ_PATH)  # [N, hidden]
meta = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        meta.append(json.loads(line))

# 提取着色标签
is_correct = np.array([m.get("is_correct") if m.get("is_correct") is not None else -1 for m in meta])
pred_label = np.array([m.get("pred_label") if m.get("pred_label") is not None else -1 for m in meta])

def scatter_plot_2d(Z, labels, title, filename):
    plt.figure(figsize=(7,6))
    # 不指定颜色（遵循你的要求）
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(Z[idx, 0], Z[idx, 1], s=12, label=str(lab))
    plt.title(title)
    plt.legend(title="label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()

# ===== t-SNE =====
tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=10, n_iter=1000, verbose=1)
# perplexity原本30，perplexity的值需要调整为小于你的样本数量
Z_tsne = tsne.fit_transform(X)

scatter_plot_2d(Z_tsne, is_correct, "t-SNE colored by is_correct (1=correct,0=wrong,-1=unknown)", "tsne_is_correct.png")
scatter_plot_2d(Z_tsne, pred_label, "t-SNE colored by pred_label (1=yes,0=no,-1=unknown)", "tsne_pred_label.png")

# ===== UMAP（如可用）=====
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", verbose=True)
    Z_umap = reducer.fit_transform(X)
    scatter_plot_2d(Z_umap, is_correct, "UMAP colored by is_correct", "umap_is_correct.png")
    scatter_plot_2d(Z_umap, pred_label, "UMAP colored by pred_label", "umap_pred_label.png")
    print(f"Saved UMAP plots to {OUT_DIR}")
else:
    print("umap-learn 未安装，已跳过 UMAP（如需，请先: pip install umap-learn）")

print(f"Saved t-SNE plots to {OUT_DIR}")
