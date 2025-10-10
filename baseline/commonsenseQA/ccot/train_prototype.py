# train_prototype.py
import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os

# =======================
# 配置路径
# =======================
INPUT_PATH = "/home2/zzl/C-CoT/baseline/commonsenseQA/ccot/generated_paths/commonsenseqa_qwen7b_paths.jsonl"
PROTOTYPE_SAVE_PATH = "/home2/zzl/C-CoT/baseline/commonsenseQA/ccot/prototype/positive_prototype.npy"
os.makedirs(os.path.dirname(PROTOTYPE_SAVE_PATH), exist_ok=True)

# 使用 BERT 模型提取路径表示
MODEL_NAME = "/home2/zzl/model/bert-base-uncased"

# =======================
# 加载 BERT 模型和 tokenizer
# =======================
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)
bert_model.eval()  # 推理模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# =======================
# 收集所有“正确”的推理路径
# =======================
print("Loading generated paths and filtering correct ones...")
correct_paths = []  # 用于训练 prototype
total_paths = 0
correct_count = 0

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        for path_data in item["paths"]:
            total_paths += 1
            if path_data["is_correct"] == 1:  # 只保留正确路径
                correct_paths.append(path_data["path"])
                correct_count += 1

print(f"Found {correct_count} correct paths out of {total_paths}")

if len(correct_paths) == 0:
    raise ValueError("No correct paths found! Cannot train prototype.")

# =======================
# 用 BERT 编码所有正确路径
# =======================
print("Encoding correct paths with BERT...")
embeddings = []

with torch.no_grad():
    for path in tqdm(correct_paths, desc="Encoding"):
        inputs = tokenizer(
            path,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        ).to(device)
        outputs = bert_model(**inputs)  # [batch, seq, dim]
        # 使用 [CLS] 向量作为句子表示
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [1, 768]
        embeddings.append(cls_embedding.squeeze())

# =======================
# 计算 prototype：所有正确路径的平均向量
# =======================
prototype = np.mean(embeddings, axis=0)
print(f"Prototype shape: {prototype.shape}")

# 保存 prototype
np.save(PROTOTYPE_SAVE_PATH, prototype)
print(f"✅ Prototype saved to {PROTOTYPE_SAVE_PATH}")