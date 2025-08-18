import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import subprocess


# ====================== 1. 工具函数：加载JSONL数据集 ======================
def load_jsonl_dataset(file_path):
    """加载JSON Lines格式数据集"""
    dataset = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在：{file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                sample = json.loads(line)
                dataset.append(sample)
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败，跳过该样本：{e}")
    print(f"成功加载数据集：{file_path}，共 {len(dataset)} 个有效样本")
    return dataset


# ====================== 2. 数据集类：适配C-CoT模型输入 ======================
class CoTDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len=512):
        """
        Args:
            raw_data: 加载后的JSONL样本列表（load_jsonl_dataset的输出）
            tokenizer: BertTokenizer实例
            max_len: 文本最大长度（BERT-base最大支持512）
        """
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        
        # 1. 构造模型输入文本（背景 + 问题）
        context = sample["raw_example"]["context"]
        question = sample["raw_example"]["question"]
        input_text = f"Context: {context}\nQuestion: {question}"  # 让模型明确任务
        
        # 2. 构造推理文本（使用数据集中的cot字段）
        reasoning_text = sample["cot"].strip()
        # 处理空推理文本（避免编码错误）
        if not reasoning_text:
            reasoning_text = "No reasoning provided."
        
        # 3. 标签（结论的真实值：1=正确，0=错误）
        label = sample["gold_label"]
        # 推理正确性标记（可选，用于分析）
        is_correct = sample["is_correct"]

        # 4. BERT编码（输入文本 + 推理文本）
        def encode_text(text):
            return self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
        
        input_enc = encode_text(input_text)
        reason_enc = encode_text(reasoning_text)

        return {
            # 输入文本相关张量
            "input_ids": input_enc["input_ids"].squeeze(0),  # [max_len]
            "attention_mask": input_enc["attention_mask"].squeeze(0),  # [max_len]
            # 推理文本相关张量
            "reason_ids": reason_enc["input_ids"].squeeze(0),  # [max_len]
            "reason_mask": reason_enc["attention_mask"].squeeze(0),  # [max_len]
            # 标签与原始文本
            "label": torch.tensor(label, dtype=torch.long),
            "is_correct": torch.tensor(is_correct, dtype=torch.long),
            "reasoning_text": reasoning_text  # 用于逻辑验证
        }


# ====================== 3. 模型定义：BERT对比学习编码器 ======================
class CoTModel(nn.Module):
    def __init__(self, model_name="/home2/zzl/model/bert-base-uncased", hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(hidden_size, hidden_size)  # 对齐嵌入维度
        self.dropout = nn.Dropout(0.1)  # 防止过拟合

    def encode(self, input_ids, attention_mask):
        """对文本进行编码，返回[CLS]位置的语义嵌入"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        cls_emb = self.dropout(cls_emb)
        return self.fc(cls_emb)  # [batch_size, hidden_size]


# ====================== 4. 损失函数：带逻辑掩码的InfoNCE ======================
def info_nce_loss(query, keys, temperature=0.07, logic_mask=None):
    """
    Args:
        query: 输入文本嵌入 [batch_size, hidden_size]
        keys: 推理文本嵌入 [batch_size, hidden_size]
        temperature: 温度参数（控制相似度分布的平滑度）
        logic_mask: 逻辑验证掩码 [batch_size]（1=逻辑有效，0=无效）
    Returns:
        带权重的InfoNCE损失
    """
    # 归一化嵌入（提升对比学习效果）
    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)

    # 计算相似度矩阵
    logits = torch.matmul(query, keys.T) / temperature  # [batch_size, batch_size]
    labels = torch.arange(query.size(0)).long().to(query.device)  # 正样本标签（对角线）

    # 逻辑掩码加权（有效推理样本权重更高）
    if logic_mask is not None:
        weight = logic_mask.float() + 0.5  # 有效=1.5，无效=0.5
        base_loss = F.cross_entropy(logits, labels, reduction="none")
        weighted_loss = (base_loss * weight).mean()
        return weighted_loss
    else:
        return F.cross_entropy(logits, labels)


# ====================== 5. 逻辑验证：调用外部pyDatalog脚本 ======================
def logic_verify(reasoning_text):
    """
    调用pyDatalog脚本验证推理文本的逻辑一致性
    Returns: 1=逻辑有效，0=无效/验证失败
    """
    script_path = "/home2/zzl/ChatLogic/pyDatalog_processing.py"
    if not os.path.exists(script_path):
        print(f"警告：逻辑验证脚本不存在：{script_path}")
        return 0
    
    try:
        # 调用外部脚本（传递推理文本作为参数）
        result = subprocess.run(
            ["python3", script_path, reasoning_text],
            capture_output=True,
            text=True,
            timeout=10  # 超时保护（避免脚本卡住）
        )
        # 检查脚本输出（假设输出包含"True"表示逻辑有效）
        if "True" in result.stdout and "False" not in result.stdout:
            return 1
        else:
            # 打印无效推理的详情（便于调试）
            print(f"逻辑无效的推理文本：{reasoning_text[:50]}...")
            print(f"脚本输出：{result.stdout.strip() or '无输出'}")
            return 0
    except subprocess.TimeoutExpired:
        print(f"警告：逻辑验证超时（推理文本：{reasoning_text[:30]}...）")
        return 0
    except Exception as e:
        print(f"逻辑验证报错：{str(e)}（推理文本：{reasoning_text[:30]}...）")
        return 0


# ====================== 6. 训练函数：单轮epoch训练 ======================
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()  # 切换训练模式
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        # 1. 数据移至设备（CPU/GPU）
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["attention_mask"].to(device)
        reason_ids = batch["reason_ids"].to(device)
        reason_mask = batch["reason_mask"].to(device)

        # 2. 生成逻辑掩码（对当前batch的推理文本做验证）
        logic_mask = []
        for reasoning_text in batch["reasoning_text"]:
            logic_mask.append(logic_verify(reasoning_text))
        logic_mask = torch.tensor(logic_mask, dtype=torch.float, device=device)

        # 3. 模型前向传播
        optimizer.zero_grad()  # 清空梯度
        input_emb = model.encode(input_ids, input_mask)
        reason_emb = model.encode(reason_ids, reason_mask)
        loss = info_nce_loss(input_emb, reason_emb, logic_mask=logic_mask)

        # 4. 反向传播与参数更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪（防止梯度爆炸）
        optimizer.step()

        # 5. 累计损失
        total_loss += loss.item()
        total_batches += 1

    # 计算平均损失
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


# ====================== 7. 验证函数：评估模型性能 ======================
def validate(model, dataloader, device):
    model.eval()  # 切换评估模式
    total_samples = 0
    correct_preds = 0  # 模型预测正确数
    logic_pass_count = 0  # 逻辑验证通过数
    correct_reason_count = 0  # 推理过程本身正确（is_correct=1）的数量

    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for batch in dataloader:
            # 1. 数据移至设备
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["attention_mask"].to(device)
            reason_ids = batch["reason_ids"].to(device)
            reason_mask = batch["reason_mask"].to(device)
            is_correct = batch["is_correct"].to(device)

            # 2. 模型编码与相似度计算
            input_emb = model.encode(input_ids, input_mask)
            reason_emb = model.encode(reason_ids, reason_mask)
            similarity = torch.matmul(input_emb, reason_emb.T)  # [batch, batch]
            predictions = torch.argmax(similarity, dim=1)  # 预测匹配的推理文本
            true_labels = torch.arange(len(predictions)).to(device)  # 真实匹配（对角线）

            # 3. 统计预测准确率
            correct_preds += (predictions == true_labels).sum().item()
            total_samples += len(predictions)

            # 4. 统计逻辑验证通过率
            for reasoning_text in batch["reasoning_text"]:
                if logic_verify(reasoning_text) == 1:
                    logic_pass_count += 1

            # 5. 统计本身正确的推理过程数
            correct_reason_count += is_correct.sum().item()

    # 计算核心指标
    baseline_acc = correct_preds / total_samples if total_samples > 0 else 0.0
    logic_pass_rate = logic_pass_count / total_samples if total_samples > 0 else 0.0
    correct_reason_rate = correct_reason_count / total_samples if total_samples > 0 else 0.0

    # 打印验证结果
    print("\n" + "="*60)
    print(f"验证结果汇总（样本总数：{total_samples}）")
    print(f"1. Baseline准确率（输入-推理匹配度）：{baseline_acc:.4f}")
    print(f"2. 逻辑验证通过率：{logic_pass_rate:.4f}")
    print(f"3. 推理过程本身正确率（is_correct）：{correct_reason_rate:.4f}")
    print("="*60 + "\n")

    return baseline_acc, logic_pass_rate, correct_reason_rate


# ====================== 8. 主函数：全流程执行 ======================
if __name__ == "__main__":
    # ---------------------- 配置参数 ----------------------
    DATASET_PATH = "/home2/zzl/C-CoT/test_C-CoT/first10_flat_labeled_onlysaveclearly.jsonl"
    BERT_MODEL_PATH = "/home2/zzl/model/bert-base-uncased"
    MAX_LEN = 512  # 适配长背景文本
    BATCH_SIZE = 2  # 根据GPU显存调整（12GB显存可设为8）
    EPOCHS = 5  # 训练轮次
    LEARNING_RATE = 5e-5  # BERT微调常用学习率

    # ---------------------- 设备初始化 ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}（GPU型号：{torch.cuda.get_device_name(0)}）" if torch.cuda.is_available() else "使用设备：CPU")

    # ---------------------- 1. 加载数据集 ----------------------
    raw_data = load_jsonl_dataset(DATASET_PATH)
    
    # 划分训练集/验证集（8:2，若样本少可全用）
    train_size = int(0.8 * len(raw_data))
    train_raw = raw_data[:train_size]
    val_raw = raw_data[train_size:]

    # ---------------------- 2. 初始化Tokenizer与Dataset ----------------------
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    train_dataset = CoTDataset(train_raw, tokenizer, MAX_LEN)
    val_dataset = CoTDataset(val_raw, tokenizer, MAX_LEN)

    # 创建DataLoader（多线程加载）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # 加速GPU数据传输
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ---------------------- 3. 初始化模型与优化器 ----------------------
    model = CoTModel(model_name=BERT_MODEL_PATH).to(device)
    # 使用PyTorch原生AdamW（解决transformers优化器 deprecated警告）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01  # 权重衰减（防止过拟合）
    )
    # 学习率调度器（可选，后期降低学习率）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------------- 4. 训练与验证循环 ----------------------
    best_val_acc = 0.0  # 记录最佳验证准确率
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        
        # 训练
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        print(f"训练损失：{train_loss:.4f}")
        
        # 验证（每个epoch后验证一次）
        val_acc, val_logic_rate, val_reason_rate = validate(model, val_dataloader, device)
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cot_model.pth")
            print(f"保存最佳模型（验证准确率：{best_val_acc:.4f}）")

    # ---------------------- 5. 最终评估 ----------------------
    print(f"\n训练完成！最佳验证准确率：{best_val_acc:.4f}")
    # 加载最佳模型做最终验证
    model.load_state_dict(torch.load("best_cot_model.pth"))
    final_acc, final_logic_rate, final_reason_rate = validate(model, val_dataloader, device)
    print(f"最终验证结果：准确率={final_acc:.4f}，逻辑通过率={final_logic_rate:.4f}")