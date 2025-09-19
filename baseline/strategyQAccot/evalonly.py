# ========================================
# evaluate_only.py
# 直接加载训练好的 C-CoT 编码器，在测试集上评估准确率
# ========================================

import os
import torch
from ccotqwenfulltry import Config, StrategyQADataset, MultiGranularityEncoder, Evaluator

# 注意：请确保 c_cot_framework.py 包含 Evaluator 类
# 如果没有，请先添加我上一条消息中的 Evaluator 代码

def evaluate_trained_model():
    config = Config()
    print(f"🚀 使用设备: {config.device}")

    # -------------------------------
    # 1. 加载测试集
    # -------------------------------
    try:
        test_dataset = StrategyQADataset(config, split="test")
        print(f"✅ 测试集加载完成，共 {len(test_dataset)} 条")
    except Exception as e:
        print(f"❌ 测试集加载失败: {e}")
        return

    # -------------------------------
    # 2. 加载训练好的编码器
    # -------------------------------
    encoder = MultiGranularityEncoder(config)
    encoder_path = "./c_cot_outputs/best_c_cot_encoder.pt"

    if not os.path.exists(encoder_path):
        print(f"❌ 找不到训练好的模型: {encoder_path}")
        return

    try:
        encoder.load_state_dict(torch.load(encoder_path, map_location=config.device))
        print(f"✅ 成功加载训练好的编码器: {encoder_path}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # -------------------------------
    # 3. 初始化评估器并运行评估
    # -------------------------------
    evaluator = Evaluator(config, encoder)
    accuracy, results = evaluator.evaluate(test_dataset, encoder_path)

    print(f"\n🌟 最终准确率: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_trained_model()