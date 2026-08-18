# test_retrieval.py
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. 路径配置
# =========================
MODEL_PATH = "${PROJECT_ROOT}/baseline/strategyQAccot/ccotllama/outputs_llama/best_c_cot_encoder.pt"
TEST_PATH = "${PROJECT_ROOT}/database/StrategyQA/strategyqa_test.json"
CORPUS_PATH = "${PROJECT_ROOT}/database/StrategyQA/strategyqa_train.json"  # 假设训练集作为检索库
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512

# =========================
# 2. 加载 tokenizer 和模型
# =========================
print("✅ 加载 tokenizer...")
tokenizer = BertTokenizer.from_pretrained('${MODEL_DIR}/bert-base-uncased')
model = BertModel.from_pretrained('${MODEL_DIR}/bert-base-uncased').to(DEVICE)
model.eval()

print("✅ 加载模型权重...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('bert.'):
        k = k[5:]  # 去掉 bert. 前缀
    new_state_dict[k] = v
model.load_state_dict(new_state_dict)
print("✅ 模型加载完成")

# =========================
# 3. 构建检索库（只用 question 字段）
# =========================
print("✅ 构建正样本库...")
with open(CORPUS_PATH, 'r') as f:
    corpus_data = json.load(f)

corpus_questions = [item["question"] for item in corpus_data]
corpus_ids = tokenizer(corpus_questions, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

# 向量化整个语料库
BATCH_SIZE = 32
corpus_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(corpus_ids['input_ids']), BATCH_SIZE), desc="Encoding Corpus"):
        batch = {k: v[i:i+BATCH_SIZE].to(DEVICE) for k, v in corpus_ids.items()}
        outputs = model(**batch)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        corpus_embeddings.append(embeddings)
corpus_embeddings = np.concatenate(corpus_embeddings, axis=0)
print(f"✅ 正样本库大小: {len(corpus_embeddings)}")

# =========================
# 4. 加载测试集并评估 Top-1 匹配准确率（模拟人工判断）
# =========================
with open(TEST_PATH, 'r') as f:
    test_data = json.load(f)

correct = 0
total = 0

print("🔍 开始评估...")
with torch.no_grad():
    for item in tqdm(test_data[:2000], desc="Evaluating"):  # 先测100个
        question = item["question"]

        # 获取测试问题的 embedding
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
        outputs = model(**inputs)
        query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]

        # 计算余弦相似度
        sims = cosine_similarity(query_emb, corpus_embeddings)[0]
        top_idx = np.argmax(sims)
        retrieved_item = corpus_data[top_idx]

        # === 智能提取 gold answer ===
        gold = item.get("answer") or item.get("label") or item.get("final_decision")
        if not gold:
            continue
        gold = str(gold).strip().lower()
        gold = "yes" if "yes" in gold or "true" in gold else "no"

        # === 检查检索到的样本答案是否一致（近似准确率）===
        # 注意：这是 weak supervision，理想情况应检查推理链
        ret_gold = retrieved_item.get("answer") or retrieved_item.get("label")
        if not ret_gold:
            continue
        ret_gold = str(ret_gold).strip().lower()
        ret_gold = "yes" if "yes" in ret_gold or "true" in ret_gold else "no"

        if gold == ret_gold:
            correct += 1
        total += 1

        # 可选：打印前5个看看
        if total <= 5:
            print(f"Q: {question}")
            print(f"Gold: {gold}, Retrieved: {ret_gold} -> {retrieved_item['question']}")
            print("-" * 50)

# =========================
# 5. 输出结果
# =========================
acc = correct / total if total > 0 else 0
print(f"\n📈 Top-1 相关性准确率（答案一致率）: {acc:.4f} ({correct}/{total})")