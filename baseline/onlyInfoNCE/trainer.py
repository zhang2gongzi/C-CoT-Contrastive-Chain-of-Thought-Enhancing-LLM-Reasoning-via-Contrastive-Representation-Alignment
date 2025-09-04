import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from config import *

def info_nce_loss(proj_feats, q_labels):
    """
    修复版InfoNCE损失：处理正样本数量不匹配问题，确保维度正确
    Args:
        proj_feats: [batch_size, PROJECT_DIM] → 投影后的序列特征
        q_labels: [batch_size] → 每个推理链对应的问题标签
    Returns:
        loss: 标量 → InfoNCE损失值
    """
    batch_size = proj_feats.shape[0]
    temp = INFONCE_TEMP

    # 1. 计算余弦相似度矩阵 [batch_size, batch_size]
    sim_matrix = F.cosine_similarity(
        proj_feats.unsqueeze(1),  # [batch, 1, dim]
        proj_feats.unsqueeze(0),  # [1, batch, dim]
        dim=-1
    )

    # 2. 屏蔽自身相似度（对角线设为-∞，避免自对比）
    self_mask = torch.eye(batch_size, dtype=torch.bool).to(DEVICE)
    sim_matrix = sim_matrix.masked_fill(self_mask, -float('inf'))

    # 3. 构建正样本掩码（同问题=正样本，排除自身）
    pos_mask = (q_labels.unsqueeze(1) == q_labels.unsqueeze(0)).to(DEVICE)
    pos_mask = pos_mask & (~self_mask)  # 等价于masked_fill，更直观

    # 4. 关键修复：计算每个样本的正样本数量，避免view维度错误
    pos_count_per_sample = pos_mask.sum(dim=1)  # [batch_size]：每个样本的正样本数
    valid_samples = pos_count_per_sample > 0  # 筛选有正样本的样本（避免无正样本导致的NaN）

    # 5. 只保留有正样本的样本进行损失计算（减少无效计算）
    valid_proj = proj_feats[valid_samples]
    valid_sim = sim_matrix[valid_samples]
    valid_pos_mask = pos_mask[valid_samples]
    valid_batch_size = valid_proj.shape[0]

    # 6. 极端情况：当前batch无有效样本（理论上不会发生，因数据已过滤每个问题≥2条链）
    if valid_batch_size == 0:
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)

    # 7. 计算InfoNCE损失（无view操作，直接按样本聚合）
    exp_sim = torch.exp(valid_sim / temp)
    # 每个有效样本的正样本exp和：按行求和（仅正样本位置）
    pos_exp_sum = (exp_sim * valid_pos_mask.float()).sum(dim=1)  # [valid_batch_size]
    # 每个有效样本的总exp和：所有负样本+正样本的exp和
    total_exp_sum = exp_sim.sum(dim=1)  # [valid_batch_size]
    # 计算损失（避免log(0)，加微小值）
    per_sample_loss = -torch.log(pos_exp_sum / (total_exp_sum + 1e-12))  # 1e-12防止分母为0

    return per_sample_loss.mean()  # 平均到有效样本

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch):
    """训练单轮：含日志打印"""
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # 加载数据到设备
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        q_labels = batch["q_label"].to(DEVICE)

        # 前向传播：获取投影特征（用于InfoNCE）
        proj_feat, _ = model(input_ids, attention_mask)

        # 计算损失
        loss = info_nce_loss(proj_feat, q_labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防爆炸
        optimizer.step()
        scheduler.step()

        # 累计损失
        total_loss += loss.item()

        # 打印训练日志（按LOG_INTERVAL）
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch+1:2d} | Batch {batch_idx+1:3d}/{len(train_loader)} | Loss: {avg_loss:.4f}")

    return total_loss / len(train_loader)  # 本轮平均损失

def evaluate(model, val_loader):
    """评估：计算损失和推理链正确率（仅用于验证特征质量）"""
    model.eval()
    total_loss = 0.0
    total_correct_cot = 0
    total_cot = 0

    with torch.no_grad():  # 关闭梯度，加速评估
        for batch in val_loader:
            # 加载数据
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            q_labels = batch["q_label"].to(DEVICE)
            cot_correct = batch["cot_correct"].to(DEVICE)  # 推理链是否正确的标签

            # 前向传播
            proj_feat, _ = model(input_ids, attention_mask)

            # 计算评估损失
            loss = info_nce_loss(proj_feat, q_labels)
            total_loss += loss.item()

            # 统计推理链正确率（仅参考，非训练目标）
            total_correct_cot += cot_correct.sum().item()
            total_cot += cot_correct.shape[0]

    # 计算评估指标
    avg_loss = total_loss / len(val_loader)
    cot_acc = total_correct_cot / total_cot if total_cot > 0 else 0.0
    print(f"Val Loss: {avg_loss:.4f} | Cot Correct Rate: {cot_acc:.4f}")
    return avg_loss, cot_acc

def train_model(model, train_loader, val_loader):
    """完整训练流程：含优化器、学习率调度、模型保存"""
    # 初始化优化器（AdamW适合BERT类模型）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # 初始化学习率调度器（带预热，避免初期学习率过高）
    total_training_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # 训练循环：跟踪最优验证损失（用于保存最佳模型）
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        # 训练单轮
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        # 验证单轮
        val_loss, val_cot_acc = evaluate(model, val_loader)

        # 保存最佳模型（按验证损失）
        if val_loss < best_val_loss and SAVE_MODEL:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss
            }, MODEL_SAVE_PATH)
            print(f"✅ 保存最佳模型到 {MODEL_SAVE_PATH}（Val Loss: {best_val_loss:.4f}）")

    print(f"\n训练完成！最佳验证损失：{best_val_loss:.4f}")