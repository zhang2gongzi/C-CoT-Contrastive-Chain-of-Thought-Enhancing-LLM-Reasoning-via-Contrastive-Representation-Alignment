# strategyqa_dataset.py
import json
import os
from torch.utils.data import Dataset

class StrategyQADataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        if split == "train":
            file_name = "strategyqa_train.json"
        elif split == "train_filtered":
            file_name = "strategyqa_train_filtered.json"
        elif split == "test":
            file_name = "strategyqa_test.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        file_path = os.path.join(config.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ 加载 {split} 数据集: {file_path} ({len(self.data)} 条)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "gold_answer": str(item["answer"]).lower(),
        }