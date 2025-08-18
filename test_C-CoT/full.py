import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import subprocess
import json

# ======================
# 1. Dataset 定义
# ======================
class CoTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        """
        data: list of dict, e.g. [{"input": "text...", "reasoning": "cot...", "label": 1}, ...]
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_enc = self.tokenizer(item["input"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        reason_enc = self.tokenizer(item["reasoning"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "reason_ids": reason_enc["input_ids"].squeeze(0),
            "reason_mask": reason_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "reasoning_text": item["reasoning"]
        }

# ======================
# 2. BERT 编码器
# ======================
class CoTModel(nn.Module):
    def __init__(self, model_name="/home2/zzl/model/bert-base-uncased", hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        return self.fc(cls_emb)

# ======================
# 3. InfoNCE Loss
# ======================
def info_nce_loss(query, keys, temperature=0.07, logic_mask=None):
    """
    query: [batch, dim]
    keys: [batch, dim]
    logic_mask: [batch] (1=逻辑有效, 0=逻辑无效)
    """
    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)

    logits = torch.matmul(query, keys.T) / temperature
    labels = torch.arange(query.size(0)).long().to(query.device)

    if logic_mask is not None:
        # 降低逻辑错误的正样本权重
        weight = logic_mask.float() + 0.5  # 合法=1.5, 不合法=0.5
        loss = F.cross_entropy(logits, labels, reduction="none")
        loss = (loss * weight).mean()
    else:
        loss = F.cross_entropy(logits, labels)
    return loss

# ======================
# 4. 调用逻辑验证
# ======================
def logic_verify(reasoning_text):
    """
    调用 pyDatalog_processing.py 验证逻辑一致性
    """
    try:
        result = subprocess.run(
            ["python3", "/home2/zzl/ChatLogic/pyDatalog_processing.py", reasoning_text],
            capture_output=True,
            text=True
        )
        if "True" in result.stdout:
            return 1
        else:
            return 0
    except Exception as e:
        print("逻辑验证失败:", e)
        return 0

# ======================
# 5. 训练示例
# ======================
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_emb = model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        reason_emb = model.encode(batch["reason_ids"].to(device), batch["reason_mask"].to(device))

        # logic verify
        logic_mask = []
        for reasoning in batch["reasoning_text"]:
            logic_mask.append(logic_verify(reasoning))
        logic_mask = torch.tensor(logic_mask, dtype=torch.float, device=device)

        loss = info_nce_loss(input_emb, reason_emb, logic_mask=logic_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ======================
# 6. 使用示例
# ======================
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("/home2/zzl/model/bert-base-uncased")
    data = [
        {"input": "Alan is strong.", "reasoning": "+strong('Alan')", "label": 1},
        {"input": "Fiona is little.", "reasoning": "+little('Fiona')", "label": 1},
    ]
    dataset = CoTDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoTModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss={loss:.4f}")
