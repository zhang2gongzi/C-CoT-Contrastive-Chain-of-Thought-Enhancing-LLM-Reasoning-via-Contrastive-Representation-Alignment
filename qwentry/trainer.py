import torch
import torch.optim as optim
from tqdm import tqdm

from config import *
from data_processor import *
from model import Encoder, info_nce_loss, classification_loss

# 训练函数
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        # 加载数据
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        path_ids = batch["path_input_ids"].to(DEVICE)
        path_mask = batch["path_attn_mask"].to(DEVICE)
        valids = batch["path_valids"].to(DEVICE)
        gold_label = batch["gold_label"].to(DEVICE)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            path_input_ids=path_ids,
            path_attn_mask=path_mask
        )
        
        # 计算损失
        contrast_loss = info_nce_loss(outputs["z_q"], outputs["z_p"], valids)
        clf_loss = classification_loss(outputs["cls_q"], outputs["z_p"], gold_label)
        loss = contrast_loss + 0.5 * clf_loss  # 加权总和
        
        # 反向传播
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
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            path_ids = batch["path_input_ids"].to(DEVICE)
            path_mask = batch["path_attn_mask"].to(DEVICE)
            valids = batch["path_valids"].to(DEVICE)
            gold_label = batch["gold_label"].item()
            path_preds = batch["path_preds"].numpy()
            
            # 模型预测（取多数有效CoT的结果）
            valid_preds = [p for p, v in zip(path_preds, valids.numpy()) if v == 1 and p != -1]
            if valid_preds:
                pred = max(set(valid_preds), key=valid_preds.count)
                passed += 1
                if pred == gold_label:
                    logic_correct += 1
            
            # 基线预测（取所有CoT的多数结果）
            all_preds = [p for p in path_preds if p != -1]
            if all_preds:
                baseline_pred = max(set(all_preds), key=all_preds.count)
                if baseline_pred == gold_label:
                    baseline_correct += 1
            
            total += 1
    
    baseline_acc = baseline_correct / total if total > 0 else 0
    logic_acc = logic_correct / passed if passed > 0 else 0
    pass_rate = passed / total if total > 0 else 0
    return baseline_acc, logic_acc, pass_rate