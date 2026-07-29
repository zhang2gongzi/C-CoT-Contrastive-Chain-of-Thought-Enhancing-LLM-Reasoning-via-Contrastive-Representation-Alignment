#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级数据划分脚本：直接读取 ckpt_generated.json，按 8:1:1 切分 train/val/test
完全适配你的数据格式：{"question", "ground_truth", "paths": [{"reasoning", "predicted_answer"}]}
"""
import json
import os
from sklearn.model_selection import train_test_split

# === 配置区（按你的实际路径调整）===
CKPT_PATH = "/home2/zzl/C-CoT/baseline/ORM/commqa/data/csqa_orm_1k/ckpt_generated.json"
OUTPUT_DIR = "/home2/zzl/C-CoT/baseline/ORM/commqa/data/csqa_orm_1k"
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test
SEED = 42
# ================================

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"未找到 checkpoint 文件: {CKPT_PATH}")

with open(CKPT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📦 成功加载 {len(data)} 条数据，格式验证通过...")

# 固定随机种子划分，保证可复现
train_r, val_r, test_r = SPLIT_RATIO
train_val, test_data = train_test_split(data, test_size=test_r, random_state=SEED)
val_size = val_r / (train_r + val_r)
train_data, val_data = train_test_split(train_val, test_size=val_size, random_state=SEED)

# 保存结果
os.makedirs(OUTPUT_DIR, exist_ok=True)
for name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
    out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存 {name}.json ({len(split_data)} 条)")

print("🎉 划分完成！目录结构如下：")
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.json'):
        print(f"  📄 {f}")