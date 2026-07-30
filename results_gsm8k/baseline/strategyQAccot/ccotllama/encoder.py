# encoder.py
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class MultiGranularityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ✅ 使用您的本地 bert-base-uncased
        local_bert_path = "/home2/zzl/model/bert-base-uncased"
        print(f"🚀 正在加载本地 BERT 模型: {local_bert_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(local_bert_path)
        self.bert = AutoModel.from_pretrained(local_bert_path)
        self.config = config

    def forward(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        cls_rep = outputs.last_hidden_state[:, 0, :]  # [B, H]
        return cls_rep, cls_rep, cls_rep