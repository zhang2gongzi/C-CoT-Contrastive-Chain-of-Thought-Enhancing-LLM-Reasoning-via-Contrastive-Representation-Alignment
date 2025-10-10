#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

# --------------------------
# Dataset
# --------------------------
class CommonsenseQADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 获取正样本文本
        pos_text = row['question'] + " " + " ".join(row['choices']['text'])
        
        # 随机负样本（同batch中其他样本）
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = torch.randint(0, len(self.df), (1,)).item()
        neg_row = self.df.iloc[neg_idx]
        neg_text = neg_row['question'] + " " + " ".join(neg_row['choices']['text'])

        # 编码
        pos_enc = self.tokenizer(pos_text,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=self.max_length,
                                 return_tensors="pt")
        neg_enc = self.tokenizer(neg_text,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=self.max_length,
                                 return_tensors="pt")

        return {
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0)
        }

# --------------------------
# Training
# --------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --------------------------
    # Load tokenizer
    # --------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            print("✅ Set pad_token = unk_token")
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✅ Set pad_token = eos_token")
        else:
            raise ValueError("Qwen tokenizer has neither pad_token, unk_token, nor eos_token")

    # --------------------------
    # Load dataframe
    # --------------------------
    df = pd.read_parquet(args.parquet_path)
    # limit 用于快速测试
    if args.limit:
        df = df.head(args.limit)
    df = df[df['answerKey'].notnull()]
    df = df.reset_index(drop=True)
    print(f"Total rows (after head & will skip empty answers): {len(df)}")

    # --------------------------
    # Load model
    # --------------------------
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(device)
    model.train()
    print("Loaded Qwen model.")

    # --------------------------
    # Dataset & Dataloader
    # --------------------------
    dataset = CommonsenseQADataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --------------------------
    # Loss & Optimizer
    # --------------------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0

        for batch in pbar:
            optimizer.zero_grad()

            # Positive
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attn_mask = batch["pos_attention_mask"].to(device)
            pos_outputs = model(pos_input_ids, attention_mask=pos_attn_mask)
            pos_embeds = pos_outputs.last_hidden_state[:, 0, :]  # use [CLS] token or first token

            # Negative
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attn_mask = batch["neg_attention_mask"].to(device)
            neg_outputs = model(neg_input_ids, attention_mask=neg_attn_mask)
            neg_embeds = neg_outputs.last_hidden_state[:, 0, :]

            # InfoNCE loss
            embeddings = torch.cat([pos_embeds, neg_embeds], dim=0)  # [2*B, H]
            labels = torch.arange(pos_embeds.size(0), device=device)
            similarity = torch.matmul(pos_embeds, embeddings.T) / 0.07  # temperature=0.07
            loss = loss_fn(similarity, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

    # --------------------------
    # Save model
    # --------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training finished. Model saved to {args.output_dir}")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train(args)
