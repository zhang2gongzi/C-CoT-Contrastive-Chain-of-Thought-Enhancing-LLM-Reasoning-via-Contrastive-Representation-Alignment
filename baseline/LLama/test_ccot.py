import os
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import multiprocessing  # 新增：用于设置多进程启动方式

# 新增：设置多进程启动方式为spawn，解决CUDA多进程问题
multiprocessing.set_start_method('spawn', force=True)

# ==============================================================================
# 1. 全局配置
# ==============================================================================
CONFIG = {
    "llama_model_path": "/home2/zzl/model/Llama-2-7b-chat-hf",
    "trained_proj_path": "/home2/zzl/C-CoT/baseline/LLama/ccot_seq_model.pt",
    "train_data_path": "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first100_flat_labeled_depth5.jsonl",
    "test_data_path": "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first100_flat_labeled_depth5.jsonl",
    "save_classifier_path": "/home2/zzl/C-CoT/baseline/LLama/cot_correctness_classifier.pt",
    "max_len": 400,
    "batch_size": 4,
    "epochs": 10,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16
}

# ==============================================================================
# 2. 模型定义
# ==============================================================================
class LlamaEncoder(nn.Module):
    """复用训练好的嵌入模型（LLaMA基础模型+投影层）"""
    def __init__(self, config):
        super().__init__()
        self.llama = AutoModel.from_pretrained(
            config["llama_model_path"],
            torch_dtype=config["dtype"],
            device_map="auto",
            low_cpu_mem_usage=True,
            max_position_embeddings=4096
        )
        for param in self.llama.parameters():
            param.requires_grad = False
        
        self.hidden_size = self.llama.config.hidden_size
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config["dtype"]).to(config["device"])
        self.proj.load_state_dict(
            torch.load(config["trained_proj_path"], map_location=config["device"], weights_only=True)
        )
        for param in self.proj.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.llama(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        raw_emb = outputs.last_hidden_state[:, -1, :]
        proj_emb = self.proj(raw_emb)
        norm_emb = F.normalize(proj_emb, dim=-1)
        return norm_emb


class CotCorrectnessClassifier(nn.Module):
    """完整分类模型：嵌入模型 + 分类头"""
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.classifier_head = nn.Sequential(
            nn.Linear(encoder.hidden_size, encoder.hidden_size // 2, dtype=config["dtype"]),
            nn.ReLU(),
            nn.Linear(encoder.hidden_size // 2, 2, dtype=config["dtype"])
        ).to(config["device"])

    def forward(self, input_ids, attention_mask):
        emb = self.encoder(input_ids, attention_mask)
        logits = self.classifier_head(emb)
        return logits

# ==============================================================================
# 3. 数据集定义
# ==============================================================================
class CotClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, config):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = config["max_len"]
        self.device = config["device"]
        
        with jsonlines.open(data_path, 'r') as f:
            for obj in f:
                if not obj.get("cot") or not obj["cot"].strip() or obj["cot"] in ("'t", "''t"):
                    continue
                if "is_correct" not in obj:
                    continue
                input_text = f"{obj['raw_example']['question']} {obj['cot']}"
                self.samples.append({
                    "text": input_text,
                    "label": obj["is_correct"]
                })
        
        print(f"✅ 数据集加载完成：{data_path} | 有效样本数：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True
        )
        # 先不转移到GPU，在主进程中统一转移（避免子进程CUDA问题）
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }

# ==============================================================================
# 4. 工具函数
# ==============================================================================
def train_classifier(model, train_dataloader, config):
    optimizer = optim.AdamW(
        model.classifier_head.parameters(),
        lr=config["lr"],
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss().to(config["device"])

    print(f"\n=== 开始训练分类头 ===")
    print(f"设备：{config['device']} | 批次大小：{config['batch_size']} | 轮数：{config['epochs']}")

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            # 在主进程中将数据转移到GPU（解决子进程无法访问CUDA的问题）
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["label"].to(config["device"])

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.shape[0]
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.shape[0]

        avg_loss = total_loss / total_samples
        train_acc = (total_correct / total_samples) * 100
        print(f"Epoch {epoch+1}/{config['epochs']} | 平均损失：{avg_loss:.4f} | 训练准确率：{train_acc:.2f}%")

    torch.save(model.state_dict(), config["save_classifier_path"])
    print(f"\n✅ 分类模型保存完成：{config['save_classifier_path']}")
    return model


def evaluate_classifier(model, test_dataloader, config):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="评估测试集"):
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["label"].to(config["device"])

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.shape[0]

    test_acc = (total_correct / total_samples) * 100
    print(f"\n=== 测试集评估结果 ===")
    print(f"测试样本数：{total_samples} | 测试准确率：{test_acc:.2f}%")
    return test_acc


def predict_single_cot(question, cot, model_path, encoder, tokenizer, config):
    model = CotCorrectnessClassifier(encoder, config)
    model.load_state_dict(
        torch.load(model_path, map_location=config["device"], weights_only=True)
    )
    model.eval()

    input_text = f"{question} {cot}"
    enc = tokenizer(
        input_text,
        truncation=True,
        max_length=config["max_len"],
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True
    )
    input_ids = enc["input_ids"].to(config["device"])
    attention_mask = enc["attention_mask"].to(config["device"])

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    result_str = "正确" if pred == 1 else "错误"
    print(f"\n=== CoT正误判断结果 ===")
    print(f"问题：{question}")
    print(f"CoT推理：{cot[:100]}..." if len(cot) > 100 else f"CoT推理：{cot}")
    print(f"判断结果：{result_str}")
    print(f"置信度：{confidence:.4f}")
    return pred, confidence

# ==============================================================================
# 5. 主函数
# ==============================================================================
def main():
    print("=== 步骤1/5：加载Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llama_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer配置：pad_token={tokenizer.pad_token} | max_len={CONFIG['max_len']}")

    print("\n=== 步骤2/5：加载嵌入模型 ===")
    encoder = LlamaEncoder(CONFIG)
    print(f"嵌入模型加载完成：hidden_size={encoder.hidden_size} | 设备={CONFIG['device']}")

    print("\n=== 步骤3/5：加载数据集 ===")
    train_dataset = CotClassificationDataset(CONFIG["train_data_path"], tokenizer, CONFIG)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,  # 现在可以安全使用多进程了
        pin_memory=True  # 启用内存锁定，加速数据传输到GPU
    )
    
    test_dataset = CotClassificationDataset(CONFIG["test_data_path"], tokenizer, CONFIG)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("\n=== 步骤4/5：训练分类头 ===")
    classifier_model = CotCorrectnessClassifier(encoder, CONFIG)
    classifier_model = train_classifier(classifier_model, train_dataloader, CONFIG)
    evaluate_classifier(classifier_model, test_dataloader, CONFIG)

    print("\n=== 步骤5/5：单样本推理演示 ===")
    demo_question1 = "Harry is quiet."
    demo_cot1 = "1. Harry is strong (from context). 2. Rule: Strong people are smart (context). 3. Rule: All smart people are quiet (context). 4. So Harry is smart → Harry is quiet. Answer: yes."
    predict_single_cot(demo_question1, demo_cot1, CONFIG["save_classifier_path"], encoder, tokenizer, CONFIG)

    demo_question2 = "Harry is quiet."
    demo_cot2 = "1. Gary is quiet (from context). 2. Harry is big, so Harry is not quiet. Answer: no."
    predict_single_cot(demo_question2, demo_cot2, CONFIG["save_classifier_path"], encoder, tokenizer, CONFIG)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
    