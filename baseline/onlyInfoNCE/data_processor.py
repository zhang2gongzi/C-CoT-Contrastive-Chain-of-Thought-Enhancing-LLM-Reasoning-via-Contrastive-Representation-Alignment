import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import *

# 初始化BERT分词器（与模型保持一致）
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

def read_pregen_cots(file_path=DATA_PATH):
    """读取JSON Lines格式数据，过滤无效推理链"""
    cot_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                # 解析单行JSON，捕获格式错误
                try:
                    item = json.loads(line)
                    # 验证核心字段（匹配你数据的结构）
                    required = ["raw_example", "cot", "gold_label", "is_correct"]
                    if not all(k in item for k in required):
                        print(f"警告：第{line_num}行缺少核心字段（如raw_example/cot），跳过")
                        continue
                    # 提取关键信息，简化数据结构
                    processed = {
                        "qid": item["raw_example"]["id"],  # 问题唯一ID（如NonNegationRule-D2-20521）
                        "question": item["raw_example"]["question"],  # 问题文本（如Harry is quiet.）
                        "context": item["raw_example"]["context"],  # 推理上下文
                        "cot_text": item["cot"],  # 推理链文本
                        "gold_label": item["gold_label"],  # 问题真实标签（1/0）
                        "cot_correct": item["is_correct"]  # 该推理链是否正确（1/0）
                    }
                    cot_data.append(processed)
                except json.JSONDecodeError as e:
                    print(f"错误：第{line_num}行JSON格式错误 -> {str(e)}，跳过")
        print(f"成功加载 {len(cot_data)} 条有效推理链（共处理{line_num}行）")
        return cot_data
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件不存在：{file_path}，请检查路径")
    except Exception as e:
        raise RuntimeError(f"读取数据失败：{str(e)}")

def group_by_question(cot_data):
    """按问题ID分组（InfoNCE需同问题的不同推理链做对比）"""
    q_group = {}
    for item in cot_data:
        qid = item["qid"]
        if qid not in q_group:
            q_group[qid] = {
                "question": item["question"],
                "gold_label": item["gold_label"],
                "cots": []  # 存储该问题的所有推理链
            }
        q_group[qid]["cots"].append({
            "cot_text": item["cot_text"],
            "cot_correct": item["cot_correct"]
        })
    # 过滤：仅保留至少2条推理链的问题（对比学习需要正负样本）
    valid_q = {k: v for k, v in q_group.items() if len(v["cots"]) >= 2}
    print(f"按问题分组后：有效问题{len(valid_q)}个（每个问题≥2条推理链）")
    return list(valid_q.values())  # 转为列表，便于后续处理

class InfoNCEDataset(Dataset):
    """适配仅InfoNCE损失的数据集，输出token化结果+标签"""
    def __init__(self, grouped_data):
        self.data = grouped_data
        self.flattened = self._flatten_data()  # 展平为单推理链样本

    def _flatten_data(self):
        """将“问题-多推理链”结构展平为“单推理链-标签”结构"""
        flattened = []
        for q in self.data:
            q_label = q["gold_label"]
            for cot in q["cots"]:
                flattened.append({
                    "cot_text": cot["cot_text"],
                    "q_label": q_label,
                    "cot_correct": cot["cot_correct"]
                })
        return flattened

    def __len__(self):
        return len(self.flattened)

    def __getitem__(self, idx):
        item = self.flattened[idx]
        # 推理链文本token化（适配BERT输入格式）
        tokenized = tokenizer(
            item["cot_text"],
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 移除batch维度（DataLoader会自动加），转为tensor
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),  # [MAX_SEQ_LEN]
            "attention_mask": tokenized["attention_mask"].squeeze(0),  # [MAX_SEQ_LEN]
            "q_label": torch.tensor(item["q_label"], dtype=torch.long),  # 问题标签（对比用）
            "cot_correct": torch.tensor(item["cot_correct"], dtype=torch.long)  # 推理链正确性（评估用）
        }

def get_dataloaders():
    """构建训练/验证/测试加载器（8:1:1划分）"""
    # 1. 读取并处理数据
    raw_data = read_pregen_cots()
    grouped_data = group_by_question(raw_data)
    total_q = len(grouped_data)
    
    # 2. 划分数据集（按问题划分，避免数据泄露）
    train_q = int(total_q * 0.8)
    val_q = int(total_q * 0.1)
    test_q = total_q - train_q - val_q

    train_data = grouped_data[:train_q]
    val_data = grouped_data[train_q:train_q+val_q]
    test_data = grouped_data[train_q+val_q:]

    # 3. 构建数据集和加载器
    train_set = InfoNCEDataset(train_data)
    val_set = InfoNCEDataset(val_data)
    test_set = InfoNCEDataset(test_data)

    # 4. 生成DataLoader
    def _build_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True  # 加速GPU数据传输
        )

    train_loader = _build_loader(train_set, shuffle=True)
    val_loader = _build_loader(val_set, shuffle=False)
    test_loader = _build_loader(test_set, shuffle=False)

    # 打印数据统计
    print(f"\n数据加载器统计：")
    print(f"- 训练集：{len(train_set)}条推理链 | {len(train_loader)}个batch")
    print(f"- 验证集：{len(val_set)}条推理链 | {len(val_loader)}个batch")
    print(f"- 测试集：{len(test_set)}条推理链 | {len(test_loader)}个batch")
    return train_loader, val_loader, test_loader