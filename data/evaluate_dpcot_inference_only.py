# evaluate_dpcot_3path.py
# Inference-only evaluation for 3-path D-ProtoCoT

import json
import torch
from transformers import BertTokenizer, BertModel
from collections import Counter

# =========================
# 1. Model & Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("/home2/zzl/model/bert-base-uncased")
encoder = BertModel.from_pretrained("/home2/zzl/model/bert-base-uncased").to(device).eval()

# =========================
# 2. Encoding function
# =========================
def encode_texts(texts):
    """
    texts: List[str]
    return: Tensor [N, D]
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = encoder(**inputs)
        hidden = outputs.last_hidden_state  # [B, L, D]
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled  # [B, D]

# =========================
# 3. Cosine similarity
# =========================
def cosine_sim(x, y):
    """
    x: [N, D]
    y: [1, D]
    return: [N]
    """
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)
    return torch.matmul(x, y.T).squeeze(-1)

# =========================
# 4. Load data
# =========================
with open("/home2/zzl/C-CoT/data/cot_train20_first500.json", "r") as f:
    data = json.load(f)

total = len(data)
correct_sc = 0
correct_dpcot = 0
oracle_correct = 0

# =========================
# 5. Evaluation loop
# =========================
for item in data:
    question = item["question"]
    desc = item.get("description", "")
    gold = item["answer"]
    paths = item["paths"]

    texts = []
    answers = []

    for p in paths:
        cot = p["cot"]
        full_text = f"{question} {desc} {cot}"
        texts.append(full_text)
        answers.append(p["final_answer"])

    # ---------- Oracle ----------
    if any(a == gold for a in answers):
        oracle_correct += 1

    # ---------- Self-Consistency ----------
    vote = Counter(answers).most_common(1)[0][0]
    if vote == gold:
        correct_sc += 1

    # ---------- D-ProtoCoT ----------
    embeddings = encode_texts(texts)  # [K, D]

    # (a) Uniform prototype (mean)
    prototype = embeddings.mean(dim=0, keepdim=True)  # [1, D]

    # (b) Select prototype-aligned path
    sims = cosine_sim(embeddings, prototype)
    best_idx = torch.argmax(sims).item()
    dpcot_pred = answers[best_idx]

    if dpcot_pred == gold:
        correct_dpcot += 1

# =========================
# 6. Results
# =========================
print(f"Total questions: {total}")
print(f"Oracle Accuracy:          {oracle_correct / total * 100:.2f}%")
print(f"Self-Consistency Accuracy:{correct_sc / total * 100:.2f}%")
print(f"D-ProtoCoT Accuracy:      {correct_dpcot / total * 100:.2f}%")
