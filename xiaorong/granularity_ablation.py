"""
粒度消融实验：对比三种训练/选择粒度组合
  1. Path-train / Path-select
  2. Step-train / Step-select
  3. Step-train / Path-select  ← D-ProtoCoT proposed design

用法:
  python granularity_ablation.py --data data/strategyqa_generated.json
  python granularity_ablation.py --data data/gsm8k_generated.json
"""

import json
import re
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import numpy as np
from sklearn.model_selection import train_test_split

# ── 配置 ──────────────────────────────────────────────────────────────────────
BERT_MODEL = "/home2/zzl/model/bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
TEMPERATURE = 0.07
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 答案抽取 ──────────────────────────────────────────────────────────────────
def extract_answer(text: str) -> str:
    """从 reasoning 或 predicted_answer 里抽取最终答案."""
    # 去掉 <think>...</think> 块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Yes/No
    yn = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if yn:
        return yn.group(1).lower()

    # 数字（GSM8K）
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if nums:
        return nums[-1]

    return text.strip().lower()[:50]


def is_correct(predicted: str, ground_truth: str) -> bool:
    pred = extract_answer(predicted).lower().strip()
    gt = ground_truth.lower().strip()
    # Yes/No 映射
    gt = "yes" if gt in ("true", "1", "yes") else ("no" if gt in ("false", "0", "no") else gt)
    return pred == gt


# ── 步骤分割 ──────────────────────────────────────────────────────────────────
def split_steps(reasoning: str) -> list[str]:
    """把一条推理路径拆成步骤列表."""
    # 去掉 <think> 块，留下最终回答部分
    clean = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()
    if not clean:
        # 如果没有最终回答，直接用 think 块内部
        m = re.search(r"<think>(.*?)</think>", reasoning, re.DOTALL)
        clean = m.group(1).strip() if m else reasoning

    # 按换行或编号拆分
    steps = re.split(r"\n{2,}|(?=\n\d+[\.\)])", clean)
    steps = [s.strip() for s in steps if s.strip()]
    return steps if steps else [clean]


# ── 数据集 ────────────────────────────────────────────────────────────────────
class ReasoningDataset(Dataset):
    def __init__(self, samples, tokenizer, granularity: str):
        """
        granularity: 'path' | 'step'
        每条样本: (text, label)
        """
        self.tokenizer = tokenizer
        self.items = []

        for s in samples:
            question = s["question"]
            gt = s["ground_truth"]
            for path in s["paths"]:
                reasoning = path["reasoning"]
                label = 1 if is_correct(path.get("predicted_answer", ""), gt) else 0

                if granularity == "path":
                    text = question + " [SEP] " + re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()
                    self.items.append((text, label))
                else:  # step
                    steps = split_steps(reasoning)
                    for step in steps:
                        text = question + " [SEP] " + step
                        self.items.append((text, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, label = self.items[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


# ── InfoNCE loss ──────────────────────────────────────────────────────────────
def infonce_loss(q_emb, pos_embs, neg_embs, temperature=TEMPERATURE):
    """
    q_emb:    (D,)
    pos_embs: (P, D)
    neg_embs: (N, D)
    """
    if pos_embs.shape[0] == 0 or neg_embs.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True).to(q_emb.device)

    q = nn.functional.normalize(q_emb, dim=-1)
    pos = nn.functional.normalize(pos_embs, dim=-1)
    neg = nn.functional.normalize(neg_embs, dim=-1)

    pos_sim = (q @ pos.T) / temperature          # (P,)
    neg_sim = (q @ neg.T) / temperature          # (N,)

    logits = torch.cat([pos_sim, neg_sim])        # (P+N,)
    labels = torch.zeros(len(logits), device=q.device)
    labels[:pos_sim.shape[0]] = 1.0 / pos_sim.shape[0]

    loss = -torch.sum(labels * torch.log_softmax(logits, dim=0))
    return loss


# ── 编码器 ────────────────────────────────────────────────────────────────────
class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, L, D)

    def get_path_emb(self, input_ids, attention_mask):
        """整条路径 → [CLS] 向量."""
        hidden = self.forward(input_ids, attention_mask)
        return hidden[:, 0, :]  # (B, D)

    def get_step_embs(self, input_ids, attention_mask):
        """返回所有 token 的 mean pooling 作为步骤表示."""
        hidden = self.forward(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1)  # (B, D)


# ── 训练一个变体 ──────────────────────────────────────────────────────────────
def train_variant(train_data, val_data, tokenizer, train_gran: str, tag: str):
    """
    train_gran: 'path' | 'step'  — 训练用什么粒度的表示
    tag: 用于打印的名称
    返回训练好的 encoder
    """
    print(f"\n{'='*60}")
    print(f"训练变体: {tag}  (train_gran={train_gran})")
    print(f"{'='*60}")

    encoder = BertEncoder().to(DEVICE)
    optimizer = AdamW(encoder.parameters(), lr=LR)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        encoder.train()
        total_loss = 0.0
        n_batches = 0

        for sample in train_data:
            question = sample["question"]
            gt = sample["ground_truth"]
            paths = sample["paths"]

            pos_texts, neg_texts = [], []
            for p in paths:
                text = re.sub(r"<think>.*?</think>", "", p["reasoning"], flags=re.DOTALL).strip()
                if is_correct(p.get("predicted_answer", ""), gt):
                    pos_texts.append(text)
                else:
                    neg_texts.append(text)

            if not pos_texts or not neg_texts:
                continue

            def encode_texts(texts):
                enc = tokenizer(
                    texts, max_length=MAX_LEN, truncation=True,
                    padding=True, return_tensors="pt"
                )
                return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

            q_ids, q_mask = encode_texts([question])
            pos_ids, pos_mask = encode_texts(pos_texts)
            neg_ids, neg_mask = encode_texts(neg_texts)

            if train_gran == "path":
                q_emb = encoder.get_path_emb(q_ids, q_mask)[0]
                pos_embs = encoder.get_path_emb(pos_ids, pos_mask)
                neg_embs = encoder.get_path_emb(neg_ids, neg_mask)
            else:  # step
                q_emb = encoder.get_step_embs(q_ids, q_mask)[0]
                pos_embs = encoder.get_step_embs(pos_ids, pos_mask)
                neg_embs = encoder.get_step_embs(neg_ids, neg_mask)

            loss = infonce_loss(q_emb, pos_embs, neg_embs)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # 验证
        val_acc = evaluate(encoder, val_data, tokenizer, select_gran="path")
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    if best_state:
        encoder.load_state_dict(best_state)
    print(f"  最佳验证准确率: {best_val_acc:.4f}")
    return encoder


# ── 推理评估 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(encoder, test_data, tokenizer, select_gran: str) -> float:
    encoder.eval()
    correct = 0
    total = 0

    for sample in test_data:
        question = sample["question"]
        gt = sample["ground_truth"]
        paths = sample["paths"]
        if not paths: continue

        # 编码问题
        q_enc = tokenizer([question], max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
        q_ids, q_mask = q_enc["input_ids"].to(DEVICE), q_enc["attention_mask"].to(DEVICE)
        
        # 编码所有路径（清理 <think> 块）
        path_texts = [re.sub(r"<think>.*?</think>", "", p["reasoning"], flags=re.DOTALL).strip() for p in paths]
        p_enc = tokenizer(path_texts, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
        p_ids, p_mask = p_enc["input_ids"].to(DEVICE), p_enc["attention_mask"].to(DEVICE)

        if select_gran == "path":
            q_emb = nn.functional.normalize(encoder.get_path_emb(q_ids, q_mask)[0], dim=-1)
            p_embs = nn.functional.normalize(encoder.get_path_emb(p_ids, p_mask), dim=-1)
        else:  # step
            q_emb = nn.functional.normalize(encoder.get_step_embs(q_ids, q_mask)[0], dim=-1)
            # 逐路径计算步骤 mean-pool
            p_embs = []
            for pt in path_texts:
                steps = split_steps(pt)
                if not steps: 
                    p_embs.append(torch.zeros(768, device=DEVICE)); continue
                s_enc = tokenizer(steps, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
                s_embs = nn.functional.normalize(encoder.get_step_embs(s_enc["input_ids"].to(DEVICE), s_enc["attention_mask"].to(DEVICE)), dim=-1)
                p_embs.append(s_embs.mean(0))
            p_embs = torch.stack(p_embs)

        # ✅ 核心修正：构建动态原型 (论文 Eq. 4)
        weights = nn.functional.softmax(p_embs @ q_emb, dim=0)
        p_q = torch.sum(weights.unsqueeze(1) * p_embs, dim=0)
        p_q = nn.functional.normalize(p_q, dim=-1)

        # ✅ 核心修正：计算与原型对齐分数 (论文 Eq. 5)
        scores = (p_embs @ p_q).cpu().numpy()
        
        best_idx = int(np.argmax(scores))
        if is_correct(paths[best_idx].get("predicted_answer", ""), gt):
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="数据文件路径，如 data/strategyqa_generated.json")
    args = parser.parse_args()

    print(f"加载数据: {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # 8:1:1 划分
    train_val, test_data = train_test_split(all_data, test_size=100, random_state=42)
    train_data, val_data = train_test_split(train_val, test_size=0.111, random_state=42)
    print(f"train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    results = {}

    # 变体 1: Path-train / Path-select
    enc1 = train_variant(train_data, val_data, tokenizer, train_gran="path", tag="Path-train / Path-select")
    acc1 = evaluate(enc1, test_data, tokenizer, select_gran="path")
    results["Path-train / Path-select"] = acc1
    print(f"\n[测试] Path-train / Path-select: {acc1*100:.2f}%")

    # 变体 2: Step-train / Step-select
    enc2 = train_variant(train_data, val_data, tokenizer, train_gran="step", tag="Step-train / Step-select")
    acc2 = evaluate(enc2, test_data, tokenizer, select_gran="step")
    results["Step-train / Step-select"] = acc2
    print(f"\n[测试] Step-train / Step-select: {acc2*100:.2f}%")

    # 变体 3: Step-train / Path-select (proposed)
    acc3 = evaluate(enc2, test_data, tokenizer, select_gran="path")
    results["Step-train / Path-select (proposed)"] = acc3
    print(f"\n[测试] Step-train / Path-select (proposed): {acc3*100:.2f}%")

    print("\n" + "="*60)
    print("粒度消融结果汇总")
    print("="*60)
    print(f"{'训练粒度':<20} {'选择粒度':<20} {'准确率':>8}")
    print("-"*50)
    print(f"{'Path-level':<20} {'Path-level':<20} {acc1*100:>7.2f}%")
    print(f"{'Step-level':<20} {'Step-level':<20} {acc2*100:>7.2f}%")
    print(f"{'Step-level':<20} {'Path-level (ours)':<20} {acc3*100:>7.2f}%")
    print("="*60)

    # 保存结果到文件
    out_path = args.data.replace(".json", "_granularity_ablation.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("粒度消融结果\n")
        f.write("="*50 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v*100:.2f}%\n")
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
