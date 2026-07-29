import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load BERT --------------------
bert_path = "/home2/zzl/model/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path).to(device)
bert_model.eval()

# -------------------- Embedding Helper --------------------
def embed_text(text):
    """Return BERT embedding for given text"""
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        output = bert_model(**tokens)
        # Use [CLS] token representation
        return output.last_hidden_state[:, 0, :]  # shape: [1, 768]

def get_path_embedding(path_text, level):
    """
    level: 'Path-only', 'Step-only', 'Path+Step', 'Full'
    """
    # 分割 steps，去掉空行，去掉前后空格
    steps = [s.strip() for s in path_text.split("\n") if s.strip().lower().startswith("step")]

    if level == "Path-only":
        return embed_text(path_text)  # [1, 768]
    elif level == "Step-only":
        if len(steps) == 0:
            # 如果没有 Step，则退回整条路径
            return embed_text(path_text)
        step_embs = [embed_text(s) for s in steps]
        return torch.mean(torch.cat(step_embs, dim=0), dim=0, keepdim=True)  # [1, 768]
    elif level == "Path+Step":
        if len(steps) == 0:
            return embed_text(path_text)
        step_embs = [embed_text(s) for s in steps]
        return torch.cat(step_embs, dim=1)  # [1, 768*num_steps]
    elif level == "Full":
        step_embs = [embed_text(s) for s in steps] if len(steps) > 0 else []
        path_emb = embed_text(path_text)
        if step_embs:
            return torch.cat([path_emb] + step_embs, dim=1)
        else:
            return path_emb
    else:
        raise ValueError(f"Unknown level {level}")

    """
    level: 'Path-only', 'Step-only', 'Path+Step', 'Full'
    """
    # Split path into steps
    steps = [s.strip() for s in path_text.split("\n") if s.startswith("Step")]
    if level == "Path-only":
        # Use the whole path text as one embedding
        return embed_text(path_text)  # [1, 768]
    elif level == "Step-only":
        # Average embedding of each step
        step_embs = [embed_text(s) for s in steps]
        return torch.mean(torch.cat(step_embs, dim=0), dim=0, keepdim=True)  # [1, 768]
    elif level == "Path+Step":
        # Concatenate all step embeddings
        step_embs = [embed_text(s) for s in steps]
        return torch.cat(step_embs, dim=1)  # [1, 768*num_steps]
    elif level == "Full":
        # Use full text (entire CoT) + each step embedding concatenated
        step_embs = [embed_text(s) for s in steps]
        path_emb = embed_text(path_text)
        return torch.cat([path_emb] + step_embs, dim=1)  # [1, 768*(num_steps+1)]
    else:
        raise ValueError(f"Unknown level {level}")

# -------------------- Path Selection --------------------
def select_path(embeddings_all, question_emb):
    """
    embeddings_all: list of tensors [num_paths, embedding_dim]
    question_emb: tensor [1, embedding_dim]
    """
    sims = []
    for emb in embeddings_all:
        # Adjust dimension
        if emb.size(1) != question_emb.size(1):
            # repeat question embedding to match
            q_repeat = question_emb.repeat(1, emb.size(1)//question_emb.size(1))
        else:
            q_repeat = question_emb
        sim = cosine_similarity(emb.detach().cpu().numpy(), q_repeat.detach().cpu().numpy())
        sims.append(sim.mean())  # average similarity
    best_idx = sims.index(max(sims))
    return best_idx

# -------------------- Main Experiment --------------------
def run_experiment(data_path):
    levels = ["Path-only", "Step-only", "Path+Step", "Full"]

    # Load dataset
    with open(data_path, "r") as f:
        data = json.load(f)

    results = []
    for level in levels:
        correct, total = 0, 0
        print(f"\n=== Running {level} experiment ===")
        for q in tqdm(data):
            question = q["question"]
            question_emb = embed_text(question)

            embeddings_all = []
            for p in q["paths"]:
                embeddings_all.append(get_path_embedding(p["text"], level))

            best_idx = select_path(embeddings_all, question_emb)
            pred = q["paths"][best_idx]["pred_answer"]
            is_correct = q["paths"][best_idx]["is_correct"]

            # 打印每条 question 选择的 path 和预测答案
            print(f"QID: {q['qid']}, Selected Path idx: {best_idx}, Pred: {pred}, Correct: {is_correct}")

            correct += is_correct
            total += 1

        acc = correct / total
        results.append({"Experiment": level, "Accuracy": acc, "Correct": correct, "Total": total})

    # 输出精度表格
    df = pd.DataFrame(results)
    print("\n=== Summary Table ===")
    print(df)

if __name__ == "__main__":
    data_path = "/home2/zzl/C-CoT/xiaorong/gsm8k_500_cot_qwen3.json"  # 替换为你的数据集路径
    run_experiment(data_path)
