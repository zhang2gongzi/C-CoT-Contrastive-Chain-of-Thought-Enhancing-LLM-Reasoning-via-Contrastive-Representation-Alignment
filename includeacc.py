# full.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import random
import subprocess
import os

# -------------------------------
# 1. 数据集定义
# -------------------------------
class CoTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text,
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.max_len,
                                  return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# 2. 模型定义 (BERT + 分类器)
# -------------------------------
class BertClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("/home2/zzl/model/bert-base-uncased")
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.fc(pooled)

# -------------------------------
# 3. Contrastive InfoNCE Loss
# -------------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        reps = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(reps, reps.t()) / self.temperature
        labels = torch.arange(batch_size).to(z_i.device)
        labels = torch.cat([labels, labels], dim=0)
        loss = nn.CrossEntropyLoss()(sim, labels)
        return loss

# -------------------------------
# 4. 逻辑验证模块
# -------------------------------
def logic_validate(cot_text):
    """
    调用 pyDatalog_processing.py 进行逻辑验证
    传入推理链字符串，返回 True / False
    """
    try:
        result = subprocess.run(
            ["python3", "pyDatalog_processing.py", cot_text],
            capture_output=True, text=True
        )
        output = result.stdout.strip()
        return "True" in output
    except Exception as e:
        print("逻辑验证错误:", e)
        return False

# -------------------------------
# 5. 训练 + 验证函数
# -------------------------------
def train_and_evaluate(train_texts, train_labels, val_texts, val_labels, epochs=3, batch_size=16, lr=2e-5):
    tokenizer = BertTokenizer.from_pretrained("/home2/zzl/model/bert-base-uncased")
    train_dataset = CoTDataset(train_texts, train_labels, tokenizer)
    val_dataset = CoTDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ------- Training -------
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(train_loader):.4f}")

    # ------- Evaluation -------
    model.eval()
    baseline_correct, validated_correct, passed, total = 0, 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].item()

            outputs = model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1).item()

            # Baseline accuracy
            if pred == label:
                baseline_correct += 1

            # 模拟推理链（这里简单拼接，可以替换为真实 CoT 输出）
            cot_text = f"If input => predict {pred}"

            # 逻辑验证
            if logic_validate(cot_text):
                passed += 1
                if pred == label:
                    validated_correct += 1

            total += 1

    baseline_acc = baseline_correct / total
    validated_acc = validated_correct / max(passed, 1)
    pass_rate = passed / total

    print(f"\n验证结果：")
    print(f"Baseline 准确率: {baseline_acc:.4f}")
    print(f"逻辑验证后准确率: {validated_acc:.4f}")
    print(f"逻辑验证通过率: {pass_rate:.4f}")

    return model

# -------------------------------
# 6. 主入口
# -------------------------------
if __name__ == "__main__":
    # toy 数据
    train_texts = ["this is clean", "this is corrupt", "good ethics", "bad politics"]
    train_labels = [1, 0, 1, 0]
    val_texts = ["ethics matter", "corruption issue", "integrity strong", "dirty case"]
    val_labels = [1, 0, 1, 0]

    model = train_and_evaluate(train_texts, train_labels, val_texts, val_labels)
