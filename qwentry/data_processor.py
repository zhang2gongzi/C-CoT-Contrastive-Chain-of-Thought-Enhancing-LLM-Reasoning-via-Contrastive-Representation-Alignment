# qwentry/data_processor.py (REPLACEMENT)
import json
import re
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import BertTokenizer

from config import *

# --------------- 读取预生成 CoT（保留 is_correct 与 pred_label） ---------------
def read_pregen_cots():
    cot_dict = defaultdict(list)
    with open(PREGEN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["raw_example"]["id"]
            cot_dict[qid].append({
                "raw_example": item["raw_example"],
                "cot_text": item.get("cot", item.get("paths", "")),
                "is_correct": item.get("is_correct", 0),
                "pred_label": item.get("pred_label", -1)
            })

    cot_data = []
    for qid, items in cot_dict.items():
        raw_example = items[0]["raw_example"]
        paths = [
            {
                "text": it["cot_text"],
                "is_correct": it["is_correct"],
                "pred_label": it["pred_label"]
            }
            for it in items
        ]
        cot_data.append({
            "raw_example": raw_example,
            "paths": paths
        })
    return cot_data

# ----------------- 将 cot_data 聚合为题级数据结构 -----------------
def build_question_level_data(cot_data):
    data = []
    for item in cot_data:
        ex = item["raw_example"]
        qid = ex["id"]
        paths = []
        for p in item["paths"]:
            paths.append({
                "text": p["text"],
                "is_correct": p.get("is_correct", 0),
                "pred_label": p.get("pred_label", -1)
            })
        data.append({
            "qid": qid,
            "context": ex.get("context", ""),
            "question": ex.get("question", ""),
            "gold_label": ex.get("label", 0),
            "paths": paths
        })
    return data

# ----------------- Dataset：输出 multi-granularity 的 token ids -----------------
class CCotDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        assert self.tokenizer is not None, "BERT tokenizer load failed"
        self.max_steps = MAX_STEPS
        self.step_max_len = STEP_MAX_LEN
        self.n_samples = N_SAMPLES
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.data)

    def split_into_steps(self, text):
        # 优先按换行或 Step 标识分割；回退到句子分割
        steps = [s.strip() for s in re.split(r'\n+|Step\s*\d+[:：]?', text) if s.strip()]
        if not steps:
            steps = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
        # 截断
        return steps[: self.max_steps]

    def encode(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_len
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def encode_step(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.step_max_len,
            return_tensors="pt"
        )

    def _pad_or_truncate_paths(self, path_texts):
        # 确保每个样本的 path 个数等于 self.n_samples（截断或 pad 空文本）
        if len(path_texts) > self.n_samples:
            return path_texts[:self.n_samples]
        else:
            return path_texts + [""] * (self.n_samples - len(path_texts))

    def __getitem__(self, idx):
        item = self.data[idx]
        # input（context + question）
        inp_text = f"Context: {item['context']}\nQuestion: {item['question']}"
        inp_enc = self.encode(inp_text, max_length=self.max_len)

        # path texts 处理并 pad/trunc
        path_texts = [p["text"] for p in item["paths"]]
        path_texts = self._pad_or_truncate_paths(path_texts)
        # path-level encodings
        path_encs = [self.encode(t, max_length=self.max_len) for t in path_texts]

        # step-level encodings：shape -> [P, S, Ls]
        path_step_input_ids_list = []
        path_step_attn_list = []
        for t in path_texts:
            steps = self.split_into_steps(t)
            step_encs = [self.encode_step(s) for s in steps]
            # pad steps 到 self.max_steps
            while len(step_encs) < self.max_steps:
                step_encs.append(self.encode_step(""))
            # stack each step encs -> [S, Ls]
            step_input_ids = torch.stack([e["input_ids"].squeeze(0) for e in step_encs])
            step_attn = torch.stack([e["attention_mask"].squeeze(0) for e in step_encs])
            path_step_input_ids_list.append(step_input_ids)
            path_step_attn_list.append(step_attn)
        # stack paths -> [P, S, Ls]
        path_step_input_ids = torch.stack(path_step_input_ids_list)
        path_step_attn_mask = torch.stack(path_step_attn_list)

        # path-level masks/ids -> [P, L]
        path_input_ids = torch.stack([e["input_ids"].squeeze(0) for e in path_encs])
        path_attn_mask = torch.stack([e["attention_mask"].squeeze(0) for e in path_encs])

        # path_is_correct and path_preds -> 保证长度 = P
        path_is_correct = []
        path_preds = []
        padded_paths = item["paths"][:self.n_samples] + [{"is_correct": 0, "pred_label": -1}] * max(0, self.n_samples - len(item["paths"]))
        for p in padded_paths:
            val = p.get("is_correct", 0)
            path_is_correct.append(float(val) if isinstance(val, (int, float)) else 0.0)
            pl = p.get("pred_label", -1)
            path_preds.append(int(pl) if isinstance(pl, (int, float)) else -1)
        path_is_correct = torch.tensor(path_is_correct, dtype=torch.float)  # [P]
        path_preds = torch.tensor(path_preds, dtype=torch.long)            # [P]

        return {
            "input_ids": inp_enc["input_ids"].squeeze(0),
            "attention_mask": inp_enc["attention_mask"].squeeze(0),
            "gold_label": torch.tensor(item["gold_label"], dtype=torch.long),
            "path_input_ids": path_input_ids,              # [P, L]
            "path_attn_mask": path_attn_mask,              # [P, L]
            "path_is_correct": path_is_correct,            # [P]
            "path_preds": path_preds,                      # [P]
            "path_step_input_ids": path_step_input_ids,    # [P, S, Ls]
            "path_step_attn_mask": path_step_attn_mask     # [P, S, Ls]
        }
