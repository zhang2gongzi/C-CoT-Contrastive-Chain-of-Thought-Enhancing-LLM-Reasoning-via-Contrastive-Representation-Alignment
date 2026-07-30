import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm

# ----------------------------
# 参数配置
# ----------------------------
MODEL_PATH = "/home2/zzl/model/Llama-2-7b-chat-hf"
DATASET_PATH = "/home2/zzl/C-CoT/database/commonsenseQA/test-00000-of-00001.parquet"
BATCH_SIZE = 2
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.05
LIMIT = 200  # 👈 只跑前200条

# ----------------------------
# 数据加载
# ----------------------------
class CoTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256, limit=None):
        df = pd.read_parquet(data_path)
        if limit:
            df = df.head(limit)

        self.samples = []
        for _, row in df.iterrows():
            q = str(row.get("question", ""))

            # 如果没有现成的 CoT 字段，就临时生成 mock 正负样本
            pos = "Because " + q.split()[0] + " is true, so it makes sense."
            neg = "This reasoning is unrelated to the question."

            self.samples.append({
                "question": q,
                "cot_positive": pos,
                "cot_negative": neg
            })

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        q = item["question"]
        pos = item["cot_positive"]
        neg = item["cot_negative"]

        pos_input = self.tokenizer(
            q + " " + pos, return_tensors="pt", max_length=self.max_len,
            truncation=True, padding="max_length"
        )
        neg_input = self.tokenizer(
            q + " " + neg, return_tensors="pt", max_length=self.max_len,
            truncation=True, padding="max_length"
        )

        return {
            "input_pos": pos_input,
            "input_neg": neg_input
        }

# ----------------------------
# 模型封装
# ----------------------------
class LlamaCoTModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.pooler = nn.Identity()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state  # [B, L, D]
        cls_embed = hidden_state[:, 0, :]  # [CLS] 向量
        return cls_embed

# ----------------------------
# InfoNCE Loss
# ----------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(batch_size, device=z1.device)
        loss = nn.functional.cross_entropy(logits, labels)
        return loss

# ----------------------------
# 评测流程
# ----------------------------
def evaluate():
    print(">>> Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    # ✅ 修复 LLaMA 无 pad_token 报错
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    model = LlamaCoTModel(MODEL_PATH)
    model.encoder.resize_token_embeddings(len(tokenizer))

    dataset = CoTDataset(DATASET_PATH, tokenizer, max_len=MAX_LEN, limit=LIMIT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = InfoNCELoss(TEMPERATURE)
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating 200 samples"):
            pos = batch["input_pos"]
            neg = batch["input_neg"]

            pos_embed = model(
                pos["input_ids"].squeeze(1).to(DEVICE),
                pos["attention_mask"].squeeze(1).to(DEVICE)
            )
            neg_embed = model(
                neg["input_ids"].squeeze(1).to(DEVICE),
                neg["attention_mask"].squeeze(1).to(DEVICE)
            )

            loss = criterion(pos_embed, neg_embed)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"\n✅ Average InfoNCE Loss (first 200 samples): {avg_loss:.4f}")

if __name__ == "__main__":
    evaluate()
