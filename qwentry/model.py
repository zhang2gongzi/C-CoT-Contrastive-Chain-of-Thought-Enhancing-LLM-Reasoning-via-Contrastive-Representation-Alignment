import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from config import *

# 编码器与投影层
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, PROJ_DIM)
        )
        self.clf_head = nn.Linear(self.bert.config.hidden_size + PROJ_DIM, 2)

    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS]向量
        z = F.normalize(self.proj(cls), dim=-1)   # 对比空间投影
        return cls, z

    def forward(self, input_ids, attention_mask, path_input_ids=None, path_attn_mask=None):
        # 编码输入
        cls_q, z_q = self.encode(input_ids, attention_mask)  # [B, H], [B, D]
        
        outputs = {"cls_q": cls_q, "z_q": z_q}
        
        if path_input_ids is not None:
            # 编码CoT路径（展平处理）
            B, P, L = path_input_ids.shape
            flat_ids = path_input_ids.view(B * P, L)
            flat_mask = path_attn_mask.view(B * P, L)
            cls_p, z_p = self.encode(flat_ids, flat_mask)  # [B*P, H], [B*P, D]
            outputs.update({
                "cls_p": cls_p.view(B, P, -1),  # [B, P, H]
                "z_p": z_p.view(B, P, -1)      # [B, P, D]
            })
        return outputs

# 对比损失（InfoNCE + 逻辑感知权重）
def info_nce_loss(z_q, z_p, valids, temperature=0.07):
    B, P, D = z_p.shape
    # 正样本掩码（逻辑有效的CoT）
    pos_mask = (valids > 0.5).float()  # [B, P]
    pos_count = pos_mask.sum(dim=1, keepdim=True)  # [B, 1]
    
    # 计算正样本中心（有效CoT的平均）
    pos_centers = (z_p * pos_mask.unsqueeze(-1)).sum(dim=1) / pos_count.clamp(min=1e-6)  # [B, D]
    
    # 负样本：所有无效CoT + 其他样本的CoT
    neg_mask = 1 - pos_mask  # [B, P]
    neg_z = []
    for b in range(B):
        # 自身无效CoT
        neg_self = z_p[b] * neg_mask[b].unsqueeze(-1)  # [P, D]
        # 其他样本的所有CoT
        neg_other = z_p[torch.arange(B) != b].view(-1, D)  # [(B-1)*P, D]
        neg_z.append(torch.cat([neg_self, neg_other], dim=0))  # [P + (B-1)*P, D]
    neg_z = torch.stack(neg_z, dim=0)  # [B, (B*P), D]
    
    # 计算相似度
    pos_sim = torch.sum(z_q * pos_centers, dim=1) / temperature  # [B]
    neg_sim = torch.bmm(z_q.unsqueeze(1), neg_z.transpose(1, 2)).squeeze(1)  # [B, (B*P)]
    
    # 拼接正负样本，计算交叉熵
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1 + B*P]
    labels = torch.zeros(B, dtype=torch.long).to(DEVICE)
    return F.cross_entropy(logits, labels)

# 分类损失（辅助准确率优化）
def classification_loss(cls_q, z_p, gold_label):
    B, P, _ = z_p.shape
    # 取有效CoT的平均表征
    z_p_avg = z_p.mean(dim=1)  # [B, D]
    # 拼接输入表征与CoT表征
    feat = torch.cat([cls_q, z_p_avg], dim=1)  # [B, H+D]
    logits = model.clf_head(feat)  # [B, 2]
    return F.cross_entropy(logits, gold_label)                     