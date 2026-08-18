# eval_c_cot.py
import json
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer as LlamaTokenizer
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter
import argparse
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型路径
LLAMA_PATH = "${MODEL_DIR}/Llama-2-7b-chat-hf"
BERT_BASE = "${MODEL_DIR}/bert-base-uncased"

# 模型映射：不同消融对应不同训练好的编码器
MODEL_MAP = {
    "full": "${PROJECT_ROOT}/baseline/xiaorong/models/f_theta_full",
    "wo_contrastive": "${PROJECT_ROOT}/baseline/xiaorong/models/f_theta_wo_con",
    "wo_multigranular": "${PROJECT_ROOT}/baseline/xiaorong/modelsf_theta_wo_mg",
    "wo_consistency": "${PROJECT_ROOT}/baseline/xiaorong/models/f_theta_wo_cs",
    "wo_dynamic_proto": "${PROJECT_ROOT}/baseline/xiaorong/models/f_theta_full",  # 动态原型是推理策略
}

def load_strategyqa(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    ground_truths = []
    for item in data:
        questions.append(item['question'])
        # 将 bool 转换为 "yes" / "no"
        gt = "yes" if item['answer'] else "no"
        ground_truths.append(gt)
    return questions, ground_truths

# ===== CoT 生成器（使用 Llama-2-7b-chat）=====
class LlamaCoTGenerator:
    def __init__(self, model_path, num_paths=5):
        print("Loading Llama-2-7b-chat...")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(DEVICE)
        self.model.eval()
        self.num_paths = num_paths
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, question):
        prompt = (
            "[INST] <<SYS>>\n"
            "You are a logical and helpful assistant. "
            "Answer the following question by reasoning step by step. "
            "At the end of your response, clearly state your final answer as '#### yes' or '#### no'.\n"
            "<</SYS>>\n\n"
            f"Question: {question}\n"
            "Answer: [/INST]"
        )
        inputs = self.tokenizer([prompt] * self.num_paths, return_tensors="pt", padding=True).to(DEVICE)
        # ... rest unchanged ...
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        cot_paths = []
        for i in range(self.num_paths):
            text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            if "Reasoning:" in text:
                cot = text.split("Reasoning:")[-1].strip()
            else:
                cot = text
            # 提取最终答案（StrategyQA 是 yes/no）
            if "yes" in cot.lower():
                cot += "\n#### yes"
            elif "no" in cot.lower():
                cot += "\n#### no"
            else:
                cot += "\n#### unknown"
            cot_paths.append(cot)
        return cot_paths

# ===== BERT 编码器 =====
class BERTEncoder:
    def __init__(self, model_path):
        print(f"Loading BERT encoder from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(DEVICE)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

def parse_answer(cot):
    if "#### yes" in cot.lower():
        return "yes"
    elif "#### no" in cot.lower():
        return "no"
    else:
        return "unknown"

def build_dynamic_prototype_and_select(cot_paths, encoder, ablation_config):
    answers = [parse_answer(p) for p in cot_paths]
    valid_answers = [a for a in answers if a in ["yes", "no"]]
    if not valid_answers:
        return "", "unknown"

    majority_answer, _ = Counter(valid_answers).most_common(1)[0]
    support_paths = [p for p, a in zip(cot_paths, answers) if a == majority_answer]
    if not support_paths:
        support_paths = cot_paths

    all_embs = encoder.encode(cot_paths)
    support_embs = encoder.encode(support_paths)

    if not ablation_config.get('wo_dynamic_proto', False):
        prototype = support_embs.mean(dim=0)
    else:
        prototype = torch.zeros(768, device=DEVICE)

    sims = F.cosine_similarity(all_embs, prototype.unsqueeze(0), dim=1)
    best_idx = sims.argmax().item()
    return cot_paths[best_idx], parse_answer(cot_paths[best_idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "wo_dynamic_proto", "wo_contrastive", "wo_multigranular", "wo_consistency"])
    args = parser.parse_args()

    # 设置消融配置
    ablation_config = {
        'wo_dynamic_proto': args.ablation == "wo_dynamic_proto",
        'wo_contrastive': False,
        'wo_multigranular': False,
        'wo_consistency': False,
    }

    # 加载数据
    questions, ground_truths = load_strategyqa("${PROJECT_ROOT}/database/StrategyQA/strategyqa_train_filtered.json")
        # 只取前 500 条用于评测
    questions = questions[:500]
    ground_truths = ground_truths[:500]
    # 初始化 CoT 生成器（可选：如果已有预生成 CoT，可跳过）
    cot_generator = LlamaCoTGenerator(LLAMA_PATH, num_paths=5)
    
    # 加载编码器
    encoder_path = MODEL_MAP[args.ablation]
    if not os.path.exists(encoder_path):
        print(f"Warning: {encoder_path} not found, using base BERT")
        encoder_path = BERT_BASE
    encoder = BERTEncoder(encoder_path)

    # 推理
    predictions = []
    for i in tqdm(range(len(questions)), desc="Evaluating"):
        q = questions[i]
        try:
            cot_paths = cot_generator.generate(q)
            _, pred = build_dynamic_prototype_and_select(cot_paths, encoder, ablation_config)
        except Exception as e:
            print(f"Error on question {i}: {e}")
            pred = "unknown"
        predictions.append(pred)

    # 计算准确率
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    acc = correct / len(predictions)
    print(f"\n🎯 Final Accuracy ({args.ablation}): {acc:.4f}")

if __name__ == "__main__":
    main()