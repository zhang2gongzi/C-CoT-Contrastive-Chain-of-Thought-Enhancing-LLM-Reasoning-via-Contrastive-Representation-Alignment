import torch
import torch.nn as nn
from transformers import BertModel
from config import *

class CotEncoder(nn.Module):
    """仅输出序列级特征的编码器（适配纯InfoNCE）"""
    def __init__(self):
        super().__init__()
        # 加载预训练BERT（冻结底层或全量微调，这里用全量微调）
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        # InfoNCE投影层（将BERT输出映射到低维空间，提升对比效果）
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, PROJECT_DIM)
        )

    def forward(self, input_ids, attention_mask):
        """
        前向传播：输入token化结果，输出序列级特征
        Args:
            input_ids: [batch_size, MAX_SEQ_LEN]
            attention_mask: [batch_size, MAX_SEQ_LEN]
        Returns:
            proj_feat: [batch_size, PROJECT_DIM] → 用于InfoNCE损失计算
            raw_feat: [batch_size, 768] → BERT原始[CLS]特征（用于评估）
        """
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        raw_feat = bert_out.last_hidden_state[:, 0, :]  # 提取[CLS]特征（序列级）
        proj_feat = self.projection(raw_feat)  # 投影到低维空间
        return proj_feat, raw_feat