import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from config import *

# 编码器与投影层（多粒度表征提取）
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        # 对比空间投影层
        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, PROJ_DIM)
        )
        # 分类头（辅助准确率优化）
        self.clf_head = nn.Linear(self.bert.config.hidden_size + PROJ_DIM, 2)

    def encode(self, input_ids, attention_mask):
        # 提取Token-level表征（last_hidden_state）和Sequence-level表征（[CLS]）
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # Sequence-level：整体语义
        z = F.normalize(self.proj(cls), dim=-1)   # 对比空间投影
        return cls, z

    def forward(self, input_ids, attention_mask, path_input_ids=None, path_attn_mask=None):
        # 编码问题（Context + Question）
        cls_q, z_q = self.encode(input_ids, attention_mask)  # [B, H], [B, D]
        
        outputs = {"cls_q": cls_q, "z_q": z_q, "clf_head": self.clf_head}
        
        # 编码CoT路径（多路径处理）
        if path_input_ids is not None:
            B, P, L = path_input_ids.shape
            flat_ids = path_input_ids.view(B * P, L)  # 展平批量和路径数
            flat_mask = path_attn_mask.view(B * P, L)
            cls_p, z_p = self.encode(flat_ids, flat_mask)  # [B*P, H], [B*P, D]
            # 恢复批量维度
            outputs.update({
                "cls_p": cls_p.view(B, P, -1),  # [B, P, H]：路径级Sequence表征
                "z_p": z_p.view(B, P, -1)      # [B, P, D]：路径级对比表征
            })
        return outputs

# # 对比损失（InfoNCE + 复用is_correct筛选正样本）
# def info_nce_loss(z_q, z_p, path_is_correct, temperature=0.07):
#     B, P, D = z_p.shape
    
#     # 正样本掩码：直接用数据中的is_correct（1=正确，0=错误）
#     pos_mask = path_is_correct  # [B, P]
#     pos_count = pos_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # 避免除零
    
#     # 正样本中心：正确CoT的平均表征（对齐目标）
#     pos_centers = (z_p * pos_mask.unsqueeze(-1)).sum(dim=1) / pos_count  # [B, D]
    
#     # 负样本构建：错误CoT + 其他问题的CoT
#     neg_z_list = []
#     for b in range(B):
#         neg_self = z_p[b] * (1 - pos_mask[b]).unsqueeze(-1)  # 自身错误CoT
#         neg_other = z_p[torch.arange(B) != b].view(-1, D)    # 其他问题的所有CoT
#         neg_z_list.append(torch.cat([neg_self, neg_other], dim=0))
#     neg_z = torch.stack(neg_z_list, dim=0)  # [B, B*P, D]
    
#     # InfoNCE核心计算：拉近正样本，推远负样本
#     pos_sim = torch.sum(z_q * pos_centers, dim=1) / temperature  # 正相似度
#     neg_sim = torch.bmm(z_q.unsqueeze(1), neg_z.transpose(1, 2)).squeeze(1)  # 负相似度
#     logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+B*P]
#     labels = torch.zeros(B, dtype=torch.long).to(DEVICE)
#     return F.cross_entropy(logits, labels)
# 消融后代码（仅用InfoNCE，删除逻辑加权）

# 消融后代码（仅用InfoNCE，随机选择路径作为正样本）
def info_nce_loss(z_q, z_p, path_is_correct=None, temperature=0.07):  # 忽略path_is_correct参数
    B, P, D = z_p.shape
    device = z_q.device
    
    # 正样本：随机选择1条路径作为正例（不再依赖逻辑正确性）
    pos_indices = torch.randint(0, P, (B,), device=device)  # 每个样本随机选1条路径
    pos_centers = z_p[torch.arange(B), pos_indices]  # [B, D]：直接取随机选中的路径表征
    
    # 负样本：同题其他路径 + 其他问题的所有路径
    neg_z_list = []
    for b in range(B):
        # 同题其他路径（排除当前随机选中的正例）
        mask = torch.ones(P, dtype=torch.bool, device=device)
        mask[pos_indices[b]] = False
        neg_self = z_p[b][mask]  # [P-1, D]
        # 其他问题的所有路径
        neg_other = z_p[torch.arange(B) != b].view(-1, D)  # [(B-1)*P, D]
        neg_z_list.append(torch.cat([neg_self, neg_other], dim=0))
    neg_z = torch.stack(neg_z_list, dim=0)  # [B, (P-1)+(B-1)*P, D]
    
    # 标准InfoNCE计算（无逻辑权重）
    pos_sim = torch.sum(z_q * pos_centers, dim=1) / temperature
    neg_sim = torch.bmm(z_q.unsqueeze(1), neg_z.transpose(1, 2)).squeeze(1)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(B, dtype=torch.long).to(device)
    return F.cross_entropy(logits, labels)
# 分类损失（辅助优化）
def classification_loss(cls_q, z_p, clf_head, gold_label):
    B, P, _ = z_p.shape
    z_p_avg = z_p.mean(dim=1)  # 所有CoT的平均对比表征
    feat = torch.cat([cls_q, z_p_avg], dim=1)  # 融合问题和CoT表征
    logits = clf_head(feat)  # [B, 2]
    return F.cross_entropy(logits, gold_label)
