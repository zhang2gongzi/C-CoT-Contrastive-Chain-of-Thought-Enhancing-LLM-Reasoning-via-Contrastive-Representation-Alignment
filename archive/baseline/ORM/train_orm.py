#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D-ProtoCoT 公平对比基线：ORM (Outcome Reward Model) 训练与推理脚本
适配您提供的数据集格式：[{"question": "...", "ground_truth": "...", "paths": [{"reasoning": "...", "predicted_answer": "..."}]}]
公平性保证：
  - 使用与 D-ProtoCoT 完全相同的 bert-base-uncased 编码器
  - 动态生成 0/1 标签（答案精确匹配）
  - 超参数：lr=2e-5, epochs=3, batch_size=32, max_length=512
"""

import argparse
import json
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# 修改后（新版写法）
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  # 从 PyTorch 原生库导入
import torch.nn as nn
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 0. 答案归一化工具（防止标点/大小写导致误判）
# ==========================================
def normalize_answer(s):
    """移除标点、多余空格，转小写，适配 CoT 数据集常见格式"""
    s = re.sub(r'[^\w\s]', '', s).strip().lower()
    return re.sub(r'\s+', ' ', s)

# ==========================================
# 1. 数据加载模块（原生适配您的嵌套 JSON）
# ==========================================
class ORMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 兼容 JSON Array 和 JSONL
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split('\n') if line.strip()]

        for item in data:
            q = item["question"]
            gold = normalize_answer(item["ground_truth"])
            for path_obj in item["paths"]:
                reasoning = path_obj["reasoning"]
                pred = normalize_answer(path_obj["predicted_answer"])
                label = 1 if pred == gold else 0
                text = f"Question: {q}\nReasoning: {reasoning}"
                self.examples.append({"text": text, "label": label})
                
        logger.info(f"✅ Loaded {len(self.examples)} path-answer pairs.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.float32)
        }

# ==========================================
# 2. 模型定义（BERT主干 + 线性分类头）
# ==========================================
class ORMClassifier(nn.Module):
    def __init__(self, model_name="${MODEL_DIR}/bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # 分类头：768维 -> 1维
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的隐藏状态作为整条路径的表征
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits}

# ==========================================
# 3. 训练与验证循环
# ==========================================
def train(args, train_loader, val_loader, model, device):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # 验证
        val_acc = evaluate(val_loader, model, device)
        logger.info(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_orm.pth"))
            logger.info(f"💾 Saved best model with acc: {best_acc:.4f}")

def evaluate(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)["logits"]
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# ==========================================
# 4. 推理打分模块
# ==========================================
def run_inference(args, model, tokenizer, device):
    model.eval()
    results = []
    logger.info(f"Running inference on {args.test_data}...")
    
    with open(args.test_data, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split('\n') if line.strip()]

    for item in tqdm(data, desc="Scoring paths"):
        q = item["question"]
        paths = item["paths"]
        scores = []
        
        for path_obj in paths:
            text = f"Question: {q}\nReasoning: {path_obj['reasoning']}"
            inputs = tokenizer(text, truncation=True, padding="max_length", 
                               max_length=args.max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(inputs["input_ids"], inputs["attention_mask"])["logits"]
                prob = torch.sigmoid(logits).item()
            scores.append(prob)
            
        best_idx = int(np.argmax(scores))
        results.append({
            "question": q,
            "selected_path_idx": best_idx,
            "scores": scores,
            "predicted_answer": paths[best_idx]["predicted_answer"],
            "ground_truth": item["ground_truth"]  # ✅ 强制对齐你的数据集字段
        })
        
    out_path = os.path.join(args.output_dir, "orm_predictions.json")
    with open(out_path, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)
    logger.info(f"📄 Inference results saved to {out_path}")

# ==========================================
# 5. 主入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORM Training & Inference")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    parser.add_argument("--model_name", type=str, default="${MODEL_DIR}/bert-base-uncased")
    parser.add_argument("--train_data", type=str, default="${PROJECT_ROOT}/xiaorong/gsm8k_500_cot_qwen3.json")
    parser.add_argument("--val_data", type=str, default="data/val.json")
    parser.add_argument("--test_data", type=str, default="data/test.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/orm")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    if args.mode == "train":
        train_ds = ORMDataset(args.train_data, tokenizer, args.max_length)
        val_ds = ORMDataset(args.val_data, tokenizer, args.max_length)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        model = ORMClassifier(args.model_name)
        train(args, train_loader, val_loader, model, device)
        
    elif args.mode == "inference":
        model = ORMClassifier(args.model_name)
        ckpt_path = os.path.join(args.output_dir, "best_orm.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        run_inference(args, model, tokenizer, device)