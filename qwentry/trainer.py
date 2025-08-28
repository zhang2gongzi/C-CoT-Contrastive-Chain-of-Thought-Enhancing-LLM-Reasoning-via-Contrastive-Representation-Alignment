import torch
import torch.optim as optim
from tqdm import tqdm

from config import *
from model import info_nce_loss, classification_loss

# 训练函数
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        # 加载批量数据
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        path_ids = batch["path_input_ids"].to(DEVICE)
        path_mask = batch["path_attn_mask"].to(DEVICE)
        gold_label = batch["gold_label"].to(DEVICE)
        path_is_correct = batch["path_is_correct"].to(DEVICE)  # 传入is_correct
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            path_input_ids=path_ids,
            path_attn_mask=path_mask
        )
        cls_q = outputs["cls_q"]
        z_q = outputs["z_q"]
        z_p = outputs["z_p"]
        clf_head = outputs["clf_head"]
        
        # 计算损失（对比损失 + 分类损失）
        contrast_loss = info_nce_loss(z_q, z_p, path_is_correct)
        clf_loss = classification_loss(cls_q, z_p, clf_head, gold_label)
        loss = contrast_loss + 0.5 * clf_loss  # 加权融合
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader):
    model.eval()
    baseline_correct = 0
    logic_correct = 0
    total = 0
    passed = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 加载数据
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            path_ids = batch["path_input_ids"].to(DEVICE)
            path_mask = batch["path_attn_mask"].to(DEVICE)
            path_is_correct = batch["path_is_correct"].cpu().numpy()  # 正确CoT标记
            gold_labels = batch["gold_label"].to(DEVICE)  # [B]
            path_preds = batch["path_preds"].cpu().numpy()  # [B, P]
            
            # 逐个样本处理
            for b in range(len(gold_labels)):
                gold_label = gold_labels[b].item()
                correct_mask = path_is_correct[b]  # 当前样本的正确CoT掩码
                preds = path_preds[b]              # 当前样本的CoT预测
                total += 1
                
                # 模型预测：仅用正确的CoT投票
                valid_preds = [p for p, c in zip(preds, correct_mask) if c == 1 and p != -1]
                if valid_preds:
                    passed += 1
                    pred = max(set(valid_preds), key=valid_preds.count)
                    if pred == gold_label:
                        logic_correct += 1
                
                # 基线预测：所有CoT投票
                all_preds = [p for p in preds if p != -1]
                if all_preds:
                    baseline_pred = max(set(all_preds), key=all_preds.count)
                    if baseline_pred == gold_label:
                        baseline_correct += 1
    
    # 计算指标
    baseline_acc = baseline_correct / total if total > 0 else 0.0
    logic_acc = logic_correct / passed if passed > 0 else 0.0
    pass_rate = passed / total if total > 0 else 0.0
    return baseline_acc, logic_acc, pass_rate