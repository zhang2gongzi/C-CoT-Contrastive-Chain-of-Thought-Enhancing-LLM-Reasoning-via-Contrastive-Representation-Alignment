import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import List

# ======================
# 配置参数
# ======================
class Config:
    LLAMA_DIR = "/home2/zzl/model/Llama-2-7b-chat-hf"   # 模型目录
    GSM8K_PARQUET_PATH = "/home2/zzl/C-CoT/database/gsm8k/test-00000-of-00001.parquet"
    OUTPUT_DIR = "./results_gsm8k"
    BERT_MODEL = "/home2/zzl/model/bert-base-uncased"

    num_paths = 4         # 每个问题生成几条 CoT
    max_len = 256
    batch_size = 2
    lr = 1e-5
    num_epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ======================
# 1. 加载模型和分词器
# ======================
tokenizer = AutoTokenizer.from_pretrained(cfg.LLAMA_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(
    cfg.LLAMA_DIR, torch_dtype=torch.float16, device_map="auto"
)

bert_tokenizer = AutoTokenizer.from_pretrained(cfg.BERT_MODEL)
bert_encoder = AutoModel.from_pretrained(cfg.BERT_MODEL).to(cfg.device)

# ======================
# 2. 读取 GSM8K 数据
# ======================
df = pd.read_parquet(cfg.GSM8K_PARQUET_PATH)
dataset = df[["question", "answer"]].dropna().reset_index(drop=True)

# ======================
# 3. 多路径 CoT 生成
# ======================
def generate_cot_paths(question: str, num_paths: int = 5) -> List[str]:
    prompt = f"Question: {question}\nLet's reason step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    paths = []
    for _ in range(num_paths):
        output = llama_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        paths.append(text)
    return paths

# ======================
# 4. 正负样本划分
# ======================
def is_correct(path: str, gold: str) -> bool:
    # 提取数字比对，避免子串匹配误判
    gold_num = re.findall(r"\d+", gold)
    pred_num = re.findall(r"\d+", path)
    return bool(gold_num and pred_num and gold_num[-1] == pred_num[-1])

# ======================
# 5. 表征提取
# ======================
def get_representations(text: str):
    tokens = bert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=cfg.max_len
    ).to(cfg.device)
    outputs = bert_encoder(**tokens)
    last_hidden = outputs.last_hidden_state  # [B, L, H]

    # token-level
    token_repr = last_hidden.squeeze(0)  # [L, H]

    # step-level（句号分句，降低 max_len）
    steps = text.split(". ")
    step_repr = []
    for step in steps:
        if step.strip():
            t = bert_tokenizer(
                step, return_tensors="pt", truncation=True, max_length=64
            ).to(cfg.device)
            o = bert_encoder(**t)
            step_repr.append(o.last_hidden_state.mean(dim=1))  # [1, H]
    if step_repr:
        step_repr = torch.cat(step_repr, dim=0)  # [num_steps, H]
    else:
        step_repr = token_repr.mean(dim=0, keepdim=True)

    # sequence-level
    seq_repr = last_hidden.mean(dim=1)  # [1, H]

    return token_repr, step_repr, seq_repr

# ======================
# 6. 对比损失（InfoNCE 风格）
# ======================
def contrastive_loss(z_pos, z_neg, temperature=0.07):
    """
    z_pos: [B, H] 正样本
    z_neg: [N, H] 负样本
    """
    z_pos = F.normalize(z_pos, dim=-1)
    z_neg = F.normalize(z_neg, dim=-1)

    loss = 0
    for zp in z_pos:  # 每个正样本当 anchor
        sim_pos = torch.exp(torch.matmul(zp, zp) / temperature)
        sim_neg = torch.exp(torch.matmul(zp, z_neg.T) / temperature).sum()
        loss += -torch.log(sim_pos / (sim_pos + sim_neg + 1e-8))
    return loss / len(z_pos)

# ======================
# 7. 训练流程
# ======================
optimizer = torch.optim.AdamW(bert_encoder.parameters(), lr=cfg.lr)

for epoch in range(cfg.num_epochs):
    total_loss = 0.0
    for idx, row in dataset.sample(50, random_state=42).reset_index(drop=True).iterrows():
        q, gold = row["question"], row["answer"]
        paths = generate_cot_paths(q, cfg.num_paths)

        pos_repr, neg_repr = [], []
        for p in paths:
            _, _, seq_repr = get_representations(p)
            if is_correct(p, gold):
                pos_repr.append(seq_repr.squeeze(0))
            else:
                neg_repr.append(seq_repr.squeeze(0))

        if len(pos_repr) > 0 and len(neg_repr) > 0:
            pos_repr = torch.stack(pos_repr)
            neg_repr = torch.stack(neg_repr)

            loss = contrastive_loss(pos_repr, neg_repr)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch+1} | Loss = {total_loss:.4f}")

# ======================
# 8. 保存结果
# ======================
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
torch.save(bert_encoder.state_dict(), os.path.join(cfg.OUTPUT_DIR, "c_cot_bert.pt"))
print("训练完成，模型已保存。")
