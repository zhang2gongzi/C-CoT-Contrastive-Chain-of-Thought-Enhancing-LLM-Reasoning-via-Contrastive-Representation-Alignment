# main.py
from config import Config
from llama_client import LlamaClient
from strategyqa_dataset import StrategyQADataset
from path_generator import PathGenerator
from encoder import MultiGranularityEncoder
from trainer import CCoTTrainer
from evaluator import Evaluator
import os

def main():
    config = Config()
    print(f"🚀 使用模型: {config.model_name}")
    print(f"🚀 设备: {config.device}")

    # 加载 Llama
    llm = LlamaClient(config.model_path)

    # 训练：使用 train_filtered.json
    train_dataset = StrategyQADataset(config, "train_filtered")
    # 小样本调试用（可删）
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, list(range(200)))

    path_gen = PathGenerator(config, llm)
    print("🧠 开始生成训练路径...")
    train_data = path_gen.generate_multi_paths(train_subset)

    encoder = MultiGranularityEncoder(config).to(config.device)
    trainer = CCoTTrainer(config, encoder)
    print("🔥 开始训练 C-CoT 编码器...")
    trainer.train(train_data)

    # 评估：使用 strategyqa_test.json
    evaluator = Evaluator(config, encoder)
    acc, _ = evaluator.evaluate(
        os.path.join(config.output_dir, "best_c_cot_encoder.pt")
    )
    print(f"🌟 最终测试集准确率: {acc:.4f}")

if __name__ == "__main__":
    main()