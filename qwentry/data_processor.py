import json
import re
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import BertTokenizer

from config import *

# # 读取原始数据（实时生成CoT时用，复用预生成时可忽略）
# def read_raw_data(path, k=NUM_EXAMPLES):
#     data = []
#     with open(path, "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             if i >= k: break
#             data.append(json.loads(line))
#     return data

# # 实时生成多路径CoT（复用预生成时无需调用）
# def generate_cots(raw_data):
#     print("生成多路径CoT...")
#     from modelscope import AutoModelForCausalLM, AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         QWEN_DIR, device_map="auto", trust_remote_code=True
#     ).eval()

#     results = []
#     for ex in tqdm(raw_data):
#         context = ex["context"]
#         question = ex["question"]
#         prompt = f"""Context: {context}
# Q: {question}
# Let's reason step by step using the facts and rules, then answer "yes" or "no".
# Reasoning:
# """
#         cots = []
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             for _ in range(N_SAMPLES):
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=MAX_NEW_TOKENS,
#                     do_sample=True,
#                     temperature=TEMPERATURE,
#                     top_p=TOP_P
#                 )
#                 cot = tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 cot = cot[len(prompt):].strip()
#                 cots.append(cot)
#         results.append({
#             "raw_example": ex,
#             "paths": cots
#         })
#     return results

# 读取预生成CoT（核心：读取is_correct标签）
def read_pregen_cots():
    cot_dict = defaultdict(list)
    with open(PREGEN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["raw_example"]["id"]  # 按问题ID聚合多路径
            cot_dict[qid].append({
                "raw_example": item["raw_example"],
                "cot_text": item["cot"],
                "is_correct": item.get("is_correct", 0),  # 读取is_correct，缺失默认0
                "pred_label": item.get("pred_label", -1)  # pred_label缺失默认-1
            })
    
    # 转换为与generate_cots一致的输出格式
    cot_data = []
    for qid, items in cot_dict.items():
        raw_example = items[0]["raw_example"]
        paths = [
            {
                "text": item["cot_text"],
                "is_correct": item["is_correct"],
                "pred_label": item["pred_label"]
            } 
            for item in items
        ]
        cot_data.append({
            "raw_example": raw_example,
            "paths": paths
        })
    return cot_data

# 解析CoT中的预测标签（实时生成时用，复用预生成时可忽略）
YES_PAT = re.compile(r"\b(answer[:：]?\s*)?(yes|true)\b", re.I)
NO_PAT = re.compile(r"\b(answer[:：]?\s*)?(no|false)\b", re.I)
def parse_pred_label(cot_text):
    cot_text = cot_text.lower()
    if YES_PAT.search(cot_text):
        return 1
    if NO_PAT.search(cot_text):
        return 0
    return -1

# 构建题级数据集（聚合多路径+保留is_correct）
def build_question_level_data(cot_data):
    print("构建题级数据集...")
    data = []
    for item in cot_data:
        ex = item["raw_example"]
        qid = ex["id"]
        paths = []
        for cot in item["paths"]:
            paths.append({
                "text": cot["text"],
                "is_correct": cot["is_correct"],  # 保留is_correct
                "pred_label": cot["pred_label"]   # 保留pred_label
            })
        data.append({
            "qid": qid,
            "context": ex["context"],
            "question": ex["question"],
            "gold_label": ex["label"],
            "paths": paths
        })
    return data

# 数据集类（完全修复版，确保所有方法正确缩进）
class CCotDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()  # 显式调用父类初始化
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        # 确保分词器加载成功
        assert self.tokenizer is not None, "BERT分词器加载失败，请检查BERT_MODEL路径"

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
        try:
            item = self.data[idx]
            
            # 编码问题
            inp_text = f"Context: {item['context']}\nQuestion: {item['question']}"
            inp_enc = self.encode(inp_text)
            
            # 编码CoT路径
            path_texts = [p["text"] for p in item["paths"]]
            path_encs = [self.encode(t) for t in path_texts]
            
            # 处理is_correct（确保无None值）
            path_is_correct = []
            for p in item["paths"]:
                val = p.get("is_correct")
                if val is None or not isinstance(val, (int, float)):
                    path_is_correct.append(0.0)
                else:
                    path_is_correct.append(float(val))
            path_is_correct = torch.tensor(path_is_correct, dtype=torch.float)
            
            # 处理pred_label（确保无None值）
            path_preds = []
            for p in item["paths"]:
                val = p.get("pred_label")
                if val is None or not isinstance(val, (int, float)):
                    path_preds.append(-1)
                else:
                    path_preds.append(int(val))
            path_preds = torch.tensor(path_preds, dtype=torch.long)
            
            # 构建返回字典
            return {
                "input_ids": inp_enc["input_ids"].squeeze(0),
                "attention_mask": inp_enc["attention_mask"].squeeze(0),
                "gold_label": torch.tensor(item["gold_label"], dtype=torch.long),
                "path_input_ids": torch.stack([e["input_ids"].squeeze(0) for e in path_encs]),
                "path_attn_mask": torch.stack([e["attention_mask"].squeeze(0) for e in path_encs]),
                "path_is_correct": path_is_correct,
                "path_preds": path_preds
            }
        except Exception as e:
            # 打印错误信息和当前索引，便于调试
            print(f"处理样本索引 {idx} 时出错: {str(e)}")
            raise  # 重新抛出异常，终止程序
