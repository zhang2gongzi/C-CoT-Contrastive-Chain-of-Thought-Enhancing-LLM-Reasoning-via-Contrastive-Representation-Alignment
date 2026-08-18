import pandas as pd
import numpy as np
import torch
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from openTSNE import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "${MODEL_DIR}/bert-base-uncased"  # 本地模型路径
INPUT_CSV = "${PROJECT_ROOT}/cot100.csv"
OUTPUT_DIR = "${PROJECT_ROOT}/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 提取干净推理文本
# ----------------------------
def extract_clean_reasoning(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    match = re.search(r"(Step\s+1[:.\s].*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return ' '.join(lines[-5:]) if lines else ""

# ----------------------------
# 加载模型
# ----------------------------
print("Loading BERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ----------------------------
# 读取数据
# ----------------------------
df = pd.read_csv(INPUT_CSV)
all_reasonings, all_labels = [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    raw = row.get("model_reasoning", "")
    correct = bool(row.get("is_correct", False))
    clean = extract_clean_reasoning(raw)
    if len(clean) >= 20 and "Step" in clean:
        all_reasonings.append(clean)
        all_labels.append(correct)

print(f"✅ Valid samples: {len(all_reasonings)}")

# ----------------------------
# BERT 编码
# ----------------------------
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEVICE)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    return emb[0]

embeddings = [encode_text(r) for r in tqdm(all_reasonings, desc="Encoding")]
X = np.array(embeddings)
y = np.array(all_labels)

# ----------------------------
# 降维
# ----------------------------
n_samples = X.shape[0]
n_pca = min(50, n_samples - 1, X.shape[1])
X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)
perplexity = min(30, n_samples - 1)
X_tsne = TSNE(perplexity=perplexity, n_jobs=-1, random_state=42).fit(X_pca)

# ----------------------------
# 绘图（论文优化版）
# ----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "serif"
})

fig, ax = plt.subplots(figsize=(5.5, 4.5))  # 适合双栏宽度

# 散点图
scatter = ax.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=["#2E8B57" if correct else "#DC143C" for correct in y],
    s=25,              # 点更小，避免重叠
    alpha=0.75,        # 适度透明
    edgecolors='none'  # 学术图通常无边框
)

# 坐标轴
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E8B57', edgecolor='none', label='Correct'),
    Patch(facecolor='#DC143C', edgecolor='none', label='Incorrect')
]
ax.legend(handles=legend_elements, loc="upper right", frameon=True, fancybox=True, shadow=False, framealpha=0.9)

# 紧凑布局
plt.tight_layout()

# 保存为矢量 PDF 和高DPI PNG
pdf_path = os.path.join(OUTPUT_DIR, "tsne_cot100.pdf")
png_path = os.path.join(OUTPUT_DIR, "tsne_cot100.png")

plt.savefig(pdf_path, dpi=600, bbox_inches='tight', format='pdf')
plt.savefig(png_path, dpi=600, bbox_inches='tight', format='png')

print(f"✅ Final figures saved:")
print(f"   PDF (vector): {pdf_path}")
print(f"   PNG (600 DPI): {png_path}")

plt.close()