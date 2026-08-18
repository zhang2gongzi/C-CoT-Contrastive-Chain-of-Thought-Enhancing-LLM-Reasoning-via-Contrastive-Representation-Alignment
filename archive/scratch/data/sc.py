import json
from collections import Counter
from typing import List, Dict, Any

def calculate_self_consistency_accuracy(json_data: List[Dict[str, Any]], 
                                       tie_breaking_strategy: str = "lexicographical") -> Dict[str, Any]:
    """
    计算self-consistency方法的准确度
    
    参数:
        json_data: JSON数据列表，每个元素包含'answer'和'paths'字段
        tie_breaking_strategy: 平票处理策略 ('lexicographical', 'first', 'random')
    
    返回:
        包含准确度统计的字典
    """
    total_samples = len(json_data)
    correct_samples = 0
    results = []
    
    for sample in json_data:
        qid = sample["qid"]
        gold_answer = sample["answer"].strip().lower()
        predictions = [path["final_answer"].strip().lower() for path in sample["paths"]]
        
        # 统计投票结果
        vote_counts = Counter(predictions)
        max_votes = max(vote_counts.values())
        winners = [ans for ans, count in vote_counts.items() if count == max_votes]
        
        # 处理平票
        if len(winners) > 1:
            if tie_breaking_strategy == "lexicographical":
                final_prediction = min(winners)  # 字典序最小
            elif tie_breaking_strategy == "first":
                final_prediction = predictions[0]  # 选择第一个出现的答案
            else:  # random
                import random
                final_prediction = random.choice(winners)
            is_tie = True
        else:
            final_prediction = winners[0]
            is_tie = False
        
        # 判断是否正确
        is_correct = (final_prediction == gold_answer)
        if is_correct:
            correct_samples += 1
        
        # 记录详细结果
        results.append({
            "qid": qid,
            "gold_answer": gold_answer,
            "predictions": predictions,
            "vote_counts": dict(vote_counts),
            "final_prediction": final_prediction,
            "is_tie": is_tie,
            "is_correct": is_correct
        })
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct_samples": correct_samples,
        "total_samples": total_samples,
        "results": results
    }

def print_detailed_report(stats: Dict[str, Any]):
    """打印详细统计报告"""
    print("=" * 80)
    print(f"Self-Consistency Accuracy Report")
    print("=" * 80)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Correct Samples: {stats['correct_samples']}")
    print(f"Accuracy: {stats['accuracy']:.2%}")
    print("=" * 80)
    
    print("\nPer-sample Results:")
    for res in stats["results"]:
        status = "✓ CORRECT" if res["is_correct"] else "✗ INCORRECT"
        tie_note = " [TIE]" if res["is_tie"] else ""
        print(f"\nQID: {res['qid']} {status}{tie_note}")
        print(f"  Gold Answer: {res['gold_answer']}")
        print(f"  Predictions: {res['predictions']}")
        print(f"  Vote Counts: {res['vote_counts']}")
        print(f"  Final Prediction: {res['final_prediction']}")
    
    print("\n" + "=" * 80)

# ============ 使用示例 ============
if __name__ == "__main__":
    with open('${PROJECT_ROOT}/data/cot_train3_first500.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算准确度
    stats = calculate_self_consistency_accuracy(
        data, 
        tie_breaking_strategy="lexicographical"  # 可选: 'first', 'random'
    )
    
    # 打印报告
    # print_detailed_report(stats)
    
    # 如需保存结果到文件:
    # with open('self_consistency_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(stats, f, indent=2, ensure_ascii=False)