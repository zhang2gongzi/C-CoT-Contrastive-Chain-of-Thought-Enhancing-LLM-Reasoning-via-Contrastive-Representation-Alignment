import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig
)
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple, Dict

# ========================================
# 配置参数
# ========================================
class Config:
    def __init__(self):
        # 路径配置
        self.data_dir = "${PROJECT_ROOT}/database/StrategyQA"
        self.QWEN_DIR = "${MODEL_DIR}/../model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
        self.BERT_MODEL = "${MODEL_DIR}/bert-base-uncased"
        self.output_dir = "./c_cot_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # 模型参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8
        self.num_epochs = 10
        self.learning_rate = 1e-5
        self.tau = 0.1  # 温度参数
        self.lambda_weight = 0.3  # 自一致性损失权重
        self.num_paths = 4  # 每条问题生成的推理路径数量
        self.max_seq_length = 512  # BERT最大序列长度
        self.max_gen_length = 256  # 推理路径最大生成长度


# ========================================
# 模块1: 数据集加载与预处理
# ========================================
class StrategyQADataset(Dataset):
    def __init__(self, config, split="train_filtered"):
        self.config = config
        self.split = split
        self.data = self._load_data()

    def _load_data(self):
        split_file_map = {
            "train": "strategyqa_train.json",
            "train_filtered": "strategyqa_train_filtered.json",
            "train_paragraphs": "strategyqa_train_paragraphs.json",
            "test": "strategyqa_test.json"
        }

        if self.split not in split_file_map:
            raise ValueError(f"无效的split: {self.split}，可选值: {list(split_file_map.keys())}")

        data_path = os.path.join(self.config.data_dir, split_file_map[self.split])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            question = item.get("question", "").strip()
            answer_bool = item.get("answer", False)
            qid = item.get("qid", str(len(processed_data)))
            gold_answer = "yes" if answer_bool else "no"

            if not question:
                continue

            processed_data.append({
                "question": question,
                "gold_answer": gold_answer,
                "id": qid,
                "term": item.get("term", ""),
                "description": item.get("description", "")
            })

        print(f"加载{self.split}集完成，有效样本数: {len(processed_data)}/{len(data)}")
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ========================================
# 模块2: 多路径推理生成（使用 Qwen）
# ========================================
class PathGenerator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.QWEN_DIR, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.QWEN_DIR,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def generate_single_path(self, question: str, term: str = "", description: str = "") -> Tuple[str, str]:
        prompt_parts = [f"Question: {question}"]
        if term and description:
            prompt_parts.append(f"Term: {term}")
            prompt_parts.append(f"Description: {description}")
        prompt_parts.append("Let's think step by step to find the answer.")
        prompt_parts.append("Reasoning: ")
        prompt = "\n".join(prompt_parts)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_gen_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        ).strip()

        answer = self._extract_answer(generated_text)
        return generated_text, answer

    def _extract_answer(self, reasoning: str) -> str:
        reasoning_lower = reasoning.lower()
        if "answer:" in reasoning_lower:
            answer_part = reasoning_lower.split("answer:")[-1].strip()
            if answer_part.startswith("yes"):
                return "yes"
            elif answer_part.startswith("no"):
                return "no"

        yes_keywords = {"yes", "correct", "true", "affirmative"}
        no_keywords = {"no", "incorrect", "false", "negative"}

        last_sentence = reasoning_lower.split(".")[-1].strip() if "." in reasoning_lower else reasoning_lower
        words = last_sentence.split()

        for word in words:
            if word in yes_keywords:
                return "yes"
            elif word in no_keywords:
                return "no"

        yes_count = sum(1 for word in reasoning_lower.split() if word in yes_keywords)
        no_count = sum(1 for word in reasoning_lower.split() if word in no_keywords)
        return "yes" if yes_count > no_count else "no"

    def generate_multi_paths(self, dataset: Dataset) -> List[Dict]:
        results = []
        for item in tqdm(dataset, desc=f"生成{self.config.num_paths}条推理路径"):
            question = item["question"]
            gold_answer = item["gold_answer"]
            term = item.get("term", "")
            description = item.get("description", "")
            paths = []

            for _ in range(self.config.num_paths):
                reasoning, answer = self.generate_single_path(question, term, description)
                paths.append({
                    "reasoning": reasoning,
                    "answer": answer,
                    "is_positive": (answer == gold_answer)
                })

            results.append({
                "id": item["id"],
                "question": question,
                "gold_answer": gold_answer,
                "paths": paths
            })

        save_path = os.path.join(self.config.output_dir, "c_cot_multi_paths_first_200.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"多路径数据已保存至: {save_path}")
        return results


# ========================================
# 模块3: 多粒度表示编码器（BERT）
# ========================================
class MultiGranularityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.BERT_MODEL)
        self.bert.to(config.device)

        # 冻结前6层
        for i, layer in enumerate(self.bert.encoder.layer[:6]):
            for param in layer.parameters():
                param.requires_grad = False

        self.hidden_size = self.bert.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.config.device)

        outputs = self.bert(**inputs, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        attention_mask = inputs["attention_mask"]  # [B, L]

        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden)
        masked_hidden = last_hidden * mask
        seq_reps = torch.sum(masked_hidden, dim=1) / torch.sum(mask, dim=1)  # [B, H]

        token_reps = last_hidden  # [B, L, H]
        step_reps = seq_reps      # 简化：用句向量代替步骤向量

        return token_reps, step_reps, seq_reps


# ========================================
# 模块4: C-CoT 训练器（对比学习）
# ========================================
class CCoTTrainer:
    def __init__(self, config, encoder: MultiGranularityEncoder):
        self.config = config
        self.encoder = encoder
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=500
        )

    def info_nce_loss(self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        anchor = anchor.unsqueeze(0)
        pos_sim = F.cosine_similarity(anchor, positives, dim=1) / self.config.tau
        neg_sim = F.cosine_similarity(anchor, negatives, dim=1) / self.config.tau
        logits = torch.cat([pos_sim, neg_sim], dim=0)
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits.unsqueeze(0), labels)

    def self_consistency_loss(self, positives: torch.Tensor) -> torch.Tensor:
        if len(positives) < 2:
            return torch.tensor(0.0, device=positives.device)
        sim_matrix = F.cosine_similarity(positives.unsqueeze(1), positives.unsqueeze(0), dim=-1)
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_mean = sim_matrix[mask].mean()
        return 1 - sim_mean

    def train_epoch(self, multi_path_data: List[Dict]) -> float:
        self.encoder.train()
        total_loss = 0.0
        valid_count = 0

        for item in tqdm(multi_path_data, desc="Training"):
            pos_texts = [p["reasoning"] for p in item["paths"] if p["is_positive"]]
            neg_texts = [p["reasoning"] for p in item["paths"] if not p["is_positive"]]

            if len(pos_texts) < 2 or len(neg_texts) < 1:
                continue

            valid_count += 1
            _, _, pos_reps = self.encoder(pos_texts)
            _, _, neg_reps = self.encoder(neg_texts)

            epoch_loss = 0.0
            for anchor_rep in pos_reps:
                loss_nce = self.info_nce_loss(anchor_rep, pos_reps, neg_reps)
                loss_consist = self.self_consistency_loss(pos_reps)
                loss = loss_nce + self.config.lambda_weight * loss_consist
                epoch_loss += loss

            epoch_loss = epoch_loss / len(pos_reps)
            self.optimizer.zero_grad()
            epoch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += epoch_loss.item()

        return total_loss / valid_count if valid_count > 0 else 0.0

    def train(self, multi_path_data: List[Dict]):
        best_loss = float("inf")
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            epoch_loss = self.train_epoch(multi_path_data)
            print(f"Epoch Loss: {epoch_loss:.4f}")

            if 0 < epoch_loss < best_loss:
                best_loss = epoch_loss
                save_path = os.path.join(self.config.output_dir, "best_c_cot_encoder.pt")
                torch.save(self.encoder.state_dict(), save_path)
                print(f"✅ 最优模型已保存: {save_path} (Loss: {best_loss:.4f})")

# ========================================
# 模块5: C-CoT 评估器（使用训练好的编码器进行重排序）
# ========================================
class Evaluator:
    def __init__(self, config, encoder: MultiGranularityEncoder):
        self.config = config
        self.encoder = encoder
        self.encoder.eval()

    def score_paths(self, question: str, paths: List[Dict], gold_answer: str) -> Tuple[str, Dict]:
        """
        使用训练好的编码器对路径进行打分和排序
        策略：每条路径的得分 = 与“问题+路径”整体表示的相似度（可自定义）
        """
        # 这里简单使用句向量的 L2 距离作为“合理性”指标（越小越好）
        # 也可以引入“问题-路径”匹配度
        reasonings = [p["reasoning"] for p in paths]
        _, _, reps = self.encoder(reasonings)  # [K, H]

        # 方法：计算所有路径表示的中心（“正确簇”先验）
        cluster_center = reps.mean(dim=0)  # [H]
        scores = -torch.norm(reps - cluster_center, dim=1)  # 负距离，越大越好

        # 排序并选择最佳路径
        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
        best_idx = sorted_indices[0]
        predicted_answer = paths[best_idx]["answer"]

        # 统计信息
        details = {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == gold_answer,
            "path_scores": scores.cpu().tolist(),
            "sorted_paths": [
                {
                    "reasoning": paths[i]["reasoning"],
                    "answer": paths[i]["answer"],
                    "score": scores[i].item(),
                    "is_positive": paths[i]["is_positive"] if "is_positive" in paths[i] else None
                }
                for i in sorted_indices
            ]
        }

        return predicted_answer, details

    def evaluate(self, test_dataset: Dataset, encoder_path: str = None):
        if encoder_path and os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.config.device))
            print(f"✅ 加载训练好的编码器: {encoder_path}")
        else:
            print("⚠️ 未找到训练好的编码器，使用初始化权重")

        self.encoder.eval()
        path_generator = PathGenerator(self.config)
        results = []
        correct = 0
        total = 0

        print(f"🚀 开始在测试集上评估（生成{self.config.num_paths}条路径 + 重排序）...")

        for item in tqdm(test_dataset, desc="Evaluating"):
            question = item["question"]
            gold_answer = item["gold_answer"]
            paths = []

            # Step 1: 生成多条推理路径
            for _ in range(self.config.num_paths):
                reasoning, answer = path_generator.generate_single_path(
                    question, item.get("term", ""), item.get("description", "")
                )
                paths.append({"reasoning": reasoning, "answer": answer})

            # Step 2: 使用编码器打分 + 重排序 + 选最佳
            pred_answer, detail = self.score_paths(question, paths, gold_answer)
            results.append(detail)
            total += 1
            if detail["is_correct"]:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n✅ 评估完成！")
        print(f"📊 总样本数: {total}")
        print(f"✅ 正确数: {correct}")
        print(f"❌ 错误数: {total - correct}")
        print(f"🎯 准确率: {accuracy:.4f} ({correct}/{total})")

        # 保存结果
        result_file = os.path.join(self.config.output_dir, "c_cot_evaluation_results.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📝 详细结果已保存至: {result_file}")

        return accuracy, results
    
# ========================================
# 主函数：完整 C-CoT 流程（仅前200条）
# ========================================
def main():
    config = Config()
    print(f"🚀 使用设备: {config.device}")

    # -------------------------------
    # 阶段1: 训练 C-CoT 编码器（使用前200条训练数据）
    # -------------------------------
    try:
        print("🔧 阶段1: 加载训练数据（前200条）")
        train_dataset = StrategyQADataset(config, split="train_filtered")
        subset_train = Subset(train_dataset, list(range(200)))
        print(f"✅ 截取前200条用于训练")

    except Exception as e:
        print(f"❌ 训练数据加载失败: {e}")
        return

    try:
        generator = PathGenerator(config)
        print("🧠 正在生成多路径推理...")
        multi_path_data = generator.generate_multi_paths(subset_train)
        print(f"✅ 生成完成，共 {len(multi_path_data)} 个问题")
    except Exception as e:
        print(f"❌ 多路径生成失败: {e}")
        return

    try:
        encoder = MultiGranularityEncoder(config)
        trainer = CCoTTrainer(config, encoder)
        print("🔥 开始 C-CoT 对比学习训练...")
        trainer.train(multi_path_data)
        print("🎉 训练完成！")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return

    # -------------------------------
    # 阶段2: 在测试集上评估
    # -------------------------------
    try:
        print("\n🔍 阶段2: 加载测试集进行评估")
        test_dataset = StrategyQADataset(config, split="test")
        print(f"✅ 测试集大小: {len(test_dataset)}")

        # 加载训练好的编码器
        encoder_path = os.path.join(config.output_dir, "best_c_cot_encoder.pt")
        if not os.path.exists(encoder_path):
            print("⚠️ 警告：未找到训练好的编码器，将使用随机初始化")
            encoder_path = None

        evaluator = Evaluator(config, encoder)
        accuracy, eval_results = evaluator.evaluate(test_dataset, encoder_path)

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n🌟 最终准确率: {accuracy:.4f}")
    
if __name__ == "__main__":
    main()