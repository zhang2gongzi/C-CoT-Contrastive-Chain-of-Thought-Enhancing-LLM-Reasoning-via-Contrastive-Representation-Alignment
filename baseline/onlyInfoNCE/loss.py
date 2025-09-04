import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

def info_nce_loss(embeddings, labels):
    """
    InfoNCE损失计算
    
    参数:
        embeddings: 序列嵌入 [batch_size, hidden_size]
        labels: 每个样本的标签（问题ID） [batch_size]
        
    返回:
        loss: InfoNCE损失值
    """
    # 对嵌入进行L2归一化
    embeddings = F.normalize(embeddings, dim=1)
    
    # 计算相似度矩阵 [batch_size, batch_size]
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / TEMP
    
    # 创建掩码：相同标签的样本为正样本
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    # 对角线为自相似，需要排除
    mask = mask - torch.eye(mask.size(0), device=DEVICE)
    
    # 计算每个样本的正样本数量
    positive_counts = mask.sum(dim=1)
    
    # 对于没有正样本的情况，设置一个小值避免除以零
    positive_counts = torch.clamp(positive_counts, min=1e-6)
    
    # 计算logits和labels
    logits = similarity_matrix
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()  # 数值稳定
    
    # 计算分母（所有样本的指数和）
    exp_logits = torch.exp(logits)
    sum_exp = exp_logits.sum(dim=1)
    
    # 计算分子（正样本的指数和）
    positive_exp = (exp_logits * mask).sum(dim=1)
    
    # 计算损失
    loss = -torch.log(positive_exp / sum_exp)
    loss = loss.mean()  # 平均到每个样本
    
    return loss
