import json
import re
import subprocess
from collections import defaultdict
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer

from config import *

# 读取原始数据
def read_raw_data(path, k=NUM_EXAMPLES):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= k: break
            data.append(json.loads(line))
    return data

# 生成多路径CoT（基于Qwen）
def generate_cots(raw_data):
    print("生成多路径CoT...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_DIR, device_map="auto", trust_remote_code=True
    ).eval()

    results = []
    for ex in tqdm(raw_data):
        context = ex["context"]
        question = ex["question"]
        prompt = f"""Context: {context}
Q: {question}
Let's reason step by step using the facts and rules, then answer "yes" or "no".
Reasoning:
"""
        cots = []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            for _ in range(N_SAMPLES):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P
                )
                cot = tokenizer.decode(outputs[0], skip_special_tokens=True)
                cot = cot[len(prompt):].strip()  # 提取推理部分
                cots.append(cot)
        results.append({
            "raw_example": ex,
            "paths": cots
        })
    return results

# 逻辑验证（调用pyDatalog）
def logic_verify(cot_text):
    try:
        res = subprocess.run(
            ["python3", PYDATALOG_PATH, cot_text],
            capture_output=True, text=True, timeout=10
        )
        return 1 if "true" in res.stdout.lower() else 0
    except:
        return 0

# 解析CoT中的预测标签
YES_PAT = re.compile(r"\b(answer[:：]?\s*)?(yes|true)\b", re.I)
NO_PAT = re.compile(r"\b(answer[:：]?\s*)?(no|false)\b", re.I)
def parse_pred_label(cot_text):
    cot_text = cot_text.lower()
    if YES_PAT.search(cot_text):
        return 1
    if NO_PAT.search(cot_text):
        return 0
    return None

# 构建题级数据集（聚合多路径）
def build_question_level_data(cot_data):
    print("构建题级数据集...")
    data = []
    for item in cot_data:
        ex = item["raw_example"]
        qid = ex["id"]
        paths = []
        for cot in item["paths"]:
            pred_label = parse_pred_label(cot)
            is_valid = logic_verify(cot)
            paths.append({
                "text": cot,
                "is_valid": is_valid,
                "pred_label": pred_label
            })
        data.append({
            "qid": qid,
            "context": ex["context"],
            "question": ex["question"],
            "gold_label": ex["label"],
            "paths": paths
        })
    return data

# 数据集类
class CCotDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        # 编码输入（context + question）
        inp_text = f"Context: {item['context']}\nQuestion: {item['question']}"
        inp_enc = self.encode(inp_text)
        
        # 编码所有CoT路径
        path_texts = [p["text"] for p in item["paths"]]
        path_encs = [self.encode(t) for t in path_texts]
        
        return {
            "input_ids": inp_enc["input_ids"].squeeze(0),
            "attention_mask": inp_enc["attention_mask"].squeeze(0),
            "gold_label": torch.tensor(item["gold_label"], dtype=torch.long),
            "path_input_ids": torch.stack([e["input_ids"].squeeze(0) for e in path_encs]),
            "path_attn_mask": torch.stack([e["attention_mask"].squeeze(0) for e in path_encs]),
            "path_valids": torch.tensor([p["is_valid"] for p in item["paths"]], dtype=torch.float),
            "path_preds": torch.tensor([p["pred_label"] if p["pred_label"] is not None else -1 
                                       for p in item["paths"]], dtype=torch.long)
        }