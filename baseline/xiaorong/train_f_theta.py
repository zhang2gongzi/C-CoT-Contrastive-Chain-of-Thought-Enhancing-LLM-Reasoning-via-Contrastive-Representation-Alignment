# train_f_theta.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_BERT_PATH = "/home2/zzl/model/bert-base-uncased"
TRAIN_DATA_PATH = "/home2/zzl/C-CoT/baseline/bamboogle/ccotprompting/c_cot_bamboogle_results_llama.json"

class CoTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r') as f:
            raw = json.load(f)
        self.data = []
        for item in raw['results']:
            pos = item['pos_rationale'].strip()
            neg = item['neg_rationale'].strip()
            if pos and neg:
                self.data.append((pos, neg))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos, neg = self.data[idx]
        return pos, neg

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class CoTEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        # 忽略 token_type_ids 和其他多余参数
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids  # BERT 通常支持它，传进去更安全
        )
        return mean_pooling(outputs, attention_mask)

def contrastive_loss(pos_emb, neg_emb, temperature=0.05):
    # InfoNCE: pos 应该靠近，neg 应该远离
    batch_size = pos_emb.size(0)
    labels = torch.arange(batch_size).to(DEVICE)
    logits = torch.matmul(pos_emb, neg_emb.T) / temperature
    loss = F.cross_entropy(logits, labels)
    return loss

def multigranular_loss(model, tokenizer, pos_texts, neg_texts, max_length=512):
    # 使用 model.bert 获取原始 BERT 输出（含 last_hidden_state）
    raw_bert = model.bert  # 这是原始的 AutoModel
    
    def get_step_embeddings(texts):
        embeddings = []
        for text in texts:
            steps = [s.strip() for s in text.split('.') if s.strip()]
            if not steps:
                steps = [text]
            step_embs = []
            for step in steps[:10]:  # 最多10步
                inputs = tokenizer(step, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
                with torch.no_grad():
                    # 使用 raw_bert，返回的是 ModelOutput
                    out = raw_bert(**inputs)
                # 现在 out 有 .last_hidden_state
                step_emb = mean_pooling(out, inputs['attention_mask'])
                step_embs.append(step_emb)
            step_embs = torch.cat(step_embs, dim=0)  # [num_steps, 768]
            embeddings.append(step_embs.mean(dim=0, keepdim=True))  # [1, 768]
        return torch.cat(embeddings, dim=0)  # [B, 768]

    pos_step_emb = get_step_embeddings(pos_texts)
    neg_step_emb = get_step_embeddings(neg_texts)
    return contrastive_loss(pos_step_emb, neg_step_emb)
def consistency_loss(model, tokenizer, texts, max_length=512):
    # 对同一文本做两次编码（不同 dropout），要求一致
    inputs1 = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(DEVICE)
    inputs2 = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(DEVICE)
    emb1 = model(input_ids=inputs1['input_ids'], attention_mask=inputs1['attention_mask'])
    emb2 = model(input_ids=inputs2['input_ids'], attention_mask=inputs2['attention_mask'])
    return F.mse_loss(emb1, emb2)

def train_one_epoch(model, dataloader, optimizer, scheduler, ablation_config, tokenizer):
    model.train()
    total_loss = 0
    for pos_texts, neg_texts in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # 编码
        pos_inputs = tokenizer(pos_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
        neg_inputs = tokenizer(neg_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
        
        pos_emb = model(**pos_inputs)
        neg_emb = model(**neg_inputs)
        
        loss = 0.0
        
        # 对比学习
        if not ablation_config.get('wo_contrastive', False):
            loss += contrastive_loss(pos_emb, neg_emb)
        
        # 多粒度对齐
        if not ablation_config.get('wo_multigranular', False):
            loss += 0.5 * multigranular_loss(model, tokenizer, pos_texts, neg_texts)
        
        # 一致性损失
        if not ablation_config.get('wo_consistency', False):
            loss += 0.1 * (consistency_loss(model, tokenizer, pos_texts) + consistency_loss(model, tokenizer, neg_texts))
        
        if loss > 0:
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "wo_contrastive", "wo_multigranular", "wo_consistency"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    ablation_config = {
        'wo_contrastive': args.ablation == "wo_contrastive",
        'wo_multigranular': args.ablation == "wo_multigranular",
        'wo_consistency': args.ablation == "wo_consistency",
    }

    print(f"Training with config: {ablation_config}")
    
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT_PATH)
    model = CoTEncoder(LOCAL_BERT_PATH).to(DEVICE)
    
    dataset = CoTDataset(TRAIN_DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * args.epochs
    )
    
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, ablation_config, tokenizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    model.bert.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()