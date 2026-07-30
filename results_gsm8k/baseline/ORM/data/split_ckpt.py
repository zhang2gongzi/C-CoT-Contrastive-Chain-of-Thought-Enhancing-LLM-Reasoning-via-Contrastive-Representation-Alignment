import json
import os
from sklearn.model_selection import train_test_split

# === 配置区 ===
CKPT_PATH = "/home2/zzl/C-CoT/baseline/ORM/data/orm_1k/ckpt_generated.json"
OUTPUT_DIR = "/home2/zzl/C-CoT/baseline/ORM/data/orm_1k"
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test
SEED = 42
# =============

with open(CKPT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📦 已加载 {len(data)} 条数据，开始按 {SPLIT_RATIO} 划分...")

train_r, val_r, test_r = SPLIT_RATIO
# 先切出 test
train_val, test_data = train_test_split(data, test_size=test_r, random_state=SEED)
# 再从剩余中切出 val
val_size = val_r / (train_r + val_r)
train_data, val_data = train_test_split(train_val, test_size=val_size, random_state=SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
for name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
    out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存 {name}.json ({len(split)} 条)")