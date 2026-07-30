import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig
)
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple, Dict

# 配置参数
class Config:
    def __init__(self):
        # 路径配置
        self.data_dir = "/home2/zzl/C-CoT/database/StrategyQA"
        self.QWEN_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
        self.BERT_MODEL = "/home2/zzl/model/bert-base-uncased"
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


# 模块1: 数据集加载与预处理
class StrategyQADataset(Dataset):
    def __init__(self, config, split="train"):
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


# 模块2: 多路径推理生成
class PathGenerator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.QWEN_DIR, trust_remote_code=True)

        # 关键：使用 eos_token 作为 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 4-bit量化
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

        save_path = os.path.join(self.config.output_dir, f"multi_paths_{dataset.split}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"多路径数据已保存至: {save_path}")
        return results


# 模块3: 多粒度表示编码器
class MultiGranularityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.BERT_MODEL)
        self.bert.to(config.device)

        # 冻结BERT前几层（例如前6层）
        for i, layer in enumerate(self.bert.encoder.layer[:6]):  # 冻结前6层
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

        # 1. Token级表示
        token_reps = last_hidden  # [B, L, H]

        # 2. 序列级表示：mean pooling
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden)
        masked_hidden = last_hidden * mask
        seq_reps = torch.sum(masked_hidden, dim=1) / torch.sum(mask, dim=1)  # [B, H]

        # 3. 步骤级表示：暂用 seq_reps 均值代替（可后续扩展）
        step_reps = seq_reps  # 简化处理，实际可按句分割后编码

        return token_reps, step_reps, seq_reps


# 模块4: 对比学习训练
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
        # anchor: [H], positives: [P, H], negatives: [N, H]
        anchor = anchor.unsqueeze(0)  # [1, H]

        pos_sim = F.cosine_similarity(anchor, positives, dim=1) / self.config.tau  # [P]
        neg_sim = F.cosine_similarity(anchor, negatives, dim=1) / self.config.tau  # [N]

        logits = torch.cat([pos_sim, neg_sim], dim=0)  # [P + N]
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)  # 第一个位置是正样本

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

        for item in tqdm(multi_path_data, desc="训练中"):
            pos_texts = [p["reasoning"] for p in item["paths"] if p["is_positive"]]
            neg_texts = [p["reasoning"] for p in item["paths"] if not p["is_positive"]]

            if len(pos_texts) < 2 or len(neg_texts) < 1:
                continue

            valid_count += 1
            _, _, pos_reps = self.encoder(pos_texts)  # [P, H]
            _, _, neg_reps = self.encoder(neg_texts)  # [N, H]

            epoch_loss = 0.0
            for anchor_rep in pos_reps:
                loss_nce = self.info_nce_loss(anchor_rep, pos_reps, neg_reps)
                loss_consist = self.self_consistency_loss(pos_reps)
                loss = loss_nce + self.config.lambda_weight * loss_consist
                epoch_loss += loss

            epoch_loss = epoch_loss / len(pos_reps)  # 平均每个 anchor 的 loss

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

            if epoch_loss < best_loss and epoch_loss > 0:
                best_loss = epoch_loss
                save_path = os.path.join(self.config.output_dir, "best_encoder.pt")
                torch.save(self.encoder.state_dict(), save_path)
                print(f"已保存最优模型（损失: {best_loss:.4f}）")


def main():
    config = Config()
    print(f"使用设备: {config.device}")

    try:
        # 加载完整数据集
        train_dataset = StrategyQADataset(config, split="train_filtered")
        print(f"原始数据集大小: {len(train_dataset)}")

        # ✅ 只取前200条数据
        subset_dataset = torch.utils.data.Subset(train_dataset, list(range(200)))
        print(f"已截取前200条数据进行推理路径生成")

    except Exception as e:
        print(f"数据集加载失败: {str(e)}")
        return

    try:
        # 初始化路径生成器
        path_generator = PathGenerator(config)

        # 存储结果
        results = []
        for item in tqdm(subset_dataset, desc=f"为前200条数据生成{config.num_paths}条推理路径"):
            item = item  # 因为是Subset，item已经是字典
            question = item["question"]
            gold_answer = item["gold_answer"]
            term = item.get("term", "")
            description = item.get("description", "")
            paths = []

            for _ in range(config.num_paths):
                reasoning, answer = path_generator.generate_single_path(question, term, description)
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

        # ✅ 保存前200条的结果
        save_path = os.path.join(config.output_dir, "multi_paths_train_filtered_first_200.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已生成前200条数据的推理路径，保存至: {save_path}")

    except Exception as e:
        print(f"多路径生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()