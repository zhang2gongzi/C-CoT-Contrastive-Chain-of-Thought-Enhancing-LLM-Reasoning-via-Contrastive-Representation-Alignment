# run_cot_inference.py
import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

# =======================
# 配置路径
# =======================
INPUT_PATH = "/home2/zzl/C-CoT/baseline/commonsenseQA/ccot/generated_paths/commonsenseqa_qwen7b_paths.jsonl"
PROTOTYPE_PATH = "/home2/zzl/C-CoT/baseline/commonsenseQA/ccot/prototype/positive_prototype.npy"
OUTPUT_RESULT_PATH = "/home2/zzl/C-CoT/baseline/commonsenseQA/ccot/results/cot_results.json"

os.makedirs(os.path.dirname(OUTPUT_RESULT_PATH), exist_ok=True)

# 使用 BERT 模型
MODEL_NAME = "/home2/zzl/model/bert-base-uncased"

# =======================
# 加载 BERT 模型和 tokenizer
# =======================
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)
bert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# =======================
# 加载训练好的 prototype
# =======================
prototype = np.load(PROTOTYPE_PATH)
print(f"Loaded prototype from {PROTOTYPE_PATH}, shape: {prototype.shape}")

# =======================
# 推理主函数
# =======================
def get_sentence_embedding(text):
    """获取文本的 BERT [CLS] 向量"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_emb.squeeze()

def select_best_path(paths_data, prototype):
    """
    选择与 prototype 最相似的路径
    paths_data: list of dict, each with "path", "predicted_answer", "is_correct"
    """
    similarities = []
    embeddings = []
    
    for item in paths_data:
        emb = get_sentence_embedding(item["path"])
        embeddings.append(emb)
    
    # 批量计算余弦相似度
    embeddings = np.array(embeddings)
    sims = cosine_similarity(embeddings, prototype.reshape(1, -1)).flatten()
    best_idx = np.argmax(sims)
    return paths_data[best_idx]

# =======================
# 主循环：对每道题选择最佳路径
# =======================
print("Running C-CoT inference with prototype...")
results = []
correct = 0
total = 0

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        gold_key = item["gold_answer"]
        
        # 选择最像“正确推理”的路径
        best_path_data = select_best_path(item["paths"], prototype)
        pred = best_path_data["predicted_answer"]
        
        # 判断是否正确
        final_correct = (pred == gold_key)
        if final_correct:
            correct += 1
        total += 1
        
        results.append({
            "question_id": item["question_id"],
            "gold_answer": gold_key,
            "final_prediction": pred,
            "is_correct": final_correct,
            "selected_path": best_path_data["path"]
        })

# =======================
# 保存结果并打印准确率
# =======================
accuracy = correct / total
print(f"\n{'='*60}")
print(f"C-CoT Inference Results (Prototype-based Selection)")
print(f"Model: Qwen-7B-Chat + BERT Prototype")
print(f"Num samples: {total}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"{'='*60}")

# 保存结果
with open(OUTPUT_RESULT_PATH, 'w', encoding='utf-8') as f:
    json.dump({
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": results
    }, f, indent=2, ensure_ascii=False)

print(f"✅ C-CoT inference results saved to {OUTPUT_RESULT_PATH}")