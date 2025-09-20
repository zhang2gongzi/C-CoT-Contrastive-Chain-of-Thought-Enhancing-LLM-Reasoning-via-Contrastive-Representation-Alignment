# evaluator.py
import torch
import os
import json
from tqdm import tqdm
from strategyqa_dataset import StrategyQADataset

class Evaluator:
    def __init__(self, config, encoder):
        self.config = config
        self.encoder = encoder

    def score_paths(self, question: str, paths: list, gold_answer: str):
        reasonings = [p["reasoning"] for p in paths]
        _, reps, _ = self.encoder(reasonings)
        cluster_center = reps.mean(dim=0)
        scores = -torch.norm(reps - cluster_center, dim=1)
        best_idx = torch.argmax(scores).item()
        pred = paths[best_idx]["answer"]

        return pred, {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": pred,
            "is_correct": pred == gold_answer,
            "sorted_paths": [{"reasoning": p["reasoning"], "answer": p["answer"]} for p in paths]
        }

    def evaluate(self, encoder_path=None):
        if encoder_path and os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.config.device))
            print(f"✅ 加载编码器: {encoder_path}")
        self.encoder.eval()

        test_dataset = StrategyQADataset(self.config, "test")  # ✅ 使用 test.json
        results = []
        correct = 0

        print("🚀 开始评估测试集...")
        for item in tqdm(test_dataset, total=len(test_dataset)):
            paths = []
            for _ in range(self.config.num_paths):
                r, a = PathGenerator(self.config, None).generate_single_path(item["question"])
                paths.append({"reasoning": r, "answer": a})

            pred, detail = self.score_paths(item["question"], paths, item["gold_answer"])
            results.append(detail)
            if detail["is_correct"]:
                correct += 1

        acc = correct / len(test_dataset)
        print(f"🎯 测试集准确率: {acc:.4f} ({correct}/{len(test_dataset)})")

        with open(os.path.join(self.config.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return acc, results