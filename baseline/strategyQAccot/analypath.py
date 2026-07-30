import json
import os
from collections import Counter

# 配置路径
OUTPUT_DIR = "/home2/zzl/c_cot_outputs"
RESULT_FILE = "/home2/zzl/c_cot_outputs/multi_paths_train_filtered_first_200.json"
file_path = os.path.join(OUTPUT_DIR, RESULT_FILE)

# 加载结果
with open(file_path, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"共加载 {len(results)} 条问题的推理路径")

# 初始化统计变量
total_paths = 0
correct_paths = 0
questions_consistent_correct = 0  # 自一致性正确的问题数

# 统计循环
for item in results:
    gold_answer = item["gold_answer"]
    predicted_answers = []

    for path in item["paths"]:
        pred_answer = path["answer"].lower().strip()
        gold = gold_answer.lower().strip()

        # 收集所有预测答案
        predicted_answers.append(pred_answer)

        # 单路径正确性
        if pred_answer == gold:
            correct_paths += 1
        total_paths += 1

    # 自一致性：多数投票
    vote_count = Counter(predicted_answers)
    majority_answer, count = vote_count.most_common(1)[0]
    if majority_answer == gold:
        questions_consistent_correct += 1

# 计算指标
per_path_accuracy = correct_paths / total_paths
self_consistency_accuracy = questions_consistent_correct / len(results)

# 输出结果
print("\n" + "="*50)
print("           推理路径正确率统计")
print("="*50)
print(f"样本数量: {len(results)} 个问题")
print(f"每问题生成路径数: {len(results[0]['paths'])} 条")
print(f"总路径数: {total_paths}")
print(f"单路径准确率: {per_path_accuracy:.4f} ({correct_paths}/{total_paths})")
print(f"自一致性准确率（多数投票）: {self_consistency_accuracy:.4f} ({questions_consistent_correct}/{len(results)})")
print("="*50)