"""统计推理路径的BERT token长度分布"""
import json
from transformers import BertTokenizer

# ============ 配置 ============
# 修改这个路径为你的数据文件路径
data_path = "/home2/zzl/C-CoT/database/Icanuse/qwen/strategyqa_generated.json"  # 或 "data/gsm8k_generated.json" 等

# ============ 加载数据 ============
print(f"加载数据: {data_path}")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"问题数量: {len(data)}")

# ============ 加载tokenizer ============
print("加载BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("/home2/zzl/model/bert-base-uncased")

# ============ 统计长度 ============
lengths = []
total_paths = 0

for sample in data:
    if "paths" in sample:
        for path in sample["paths"]:
            if "reasoning" in path:
                tokens = tokenizer(path["reasoning"])["input_ids"]
                lengths.append(len(tokens))
                total_paths += 1

# ============ 输出结果 ============
print(f"\n{'='*50}")
print(f"Token长度统计")
print(f"{'='*50}")
print(f"总路径数: {total_paths}")
print(f"\n--- 超限统计 ---")
print(f"超512数量: {sum(l > 512 for l in lengths)}")
print(f"超512比例: {sum(l > 512 for l in lengths) / len(lengths) * 100:.2f}%")
print(f"超384数量: {sum(l > 384 for l in lengths)} ({sum(l > 384 for l in lengths) / len(lengths) * 100:.2f}%)")
print(f"超256数量: {sum(l > 256 for l in lengths)} ({sum(l > 256 for l in lengths) / len(lengths) * 100:.2f}%)")

print(f"\n--- 基本统计 ---")
print(f"平均长度: {sum(lengths)/len(lengths):.1f}")
print(f"中位数: {sorted(lengths)[len(lengths)//2]}")
print(f"最大长度: {max(lengths)}")
print(f"最小长度: {min(lengths)}")

print(f"\n--- 长度分布 ---")
for threshold in [128, 256, 384, 512, 768, 1024]:
    count = sum(l <= threshold for l in lengths)
    print(f"<= {threshold}: {count} ({count/len(lengths)*100:.1f}%)")
