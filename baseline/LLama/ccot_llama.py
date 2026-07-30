import os
import json
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# --------------------------
# 1. 配置参数（关键：移除[SEP]，严格控制长度）
# --------------------------
DATASET_PATH = "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first100_flat_labeled_depth5.jsonl"
MODEL_NAME = "/home2/zzl/model/Llama-2-7b-chat-hf"
SAVE_PATH = "/home2/zzl/C-CoT/baseline/LLama/ccot_seq_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # 保持小批量，避免显存溢出
LR = 1e-5
EPOCHS = 3
MAX_LEN = 400  # 关键：设为400（远小于LLaMA-2的4096上限，留足余量）
TEMPERATURE = 0.07
DTYPE = torch.float16


# --------------------------
# 2. 数据集类（核心修复：移除[SEP]，添加长度校验）
# --------------------------
class CoTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=400):
        self.question_samples = {}
        with jsonlines.open(path, 'r') as reader:
            for obj in reader:
                # 提取核心字段（与原代码一致）
                question_id = obj["raw_example"]["id"]
                question_text = obj["raw_example"]["question"]
                cot_text = obj["cot"]
                is_correct = obj["is_correct"]

                # 过滤无效CoT（增强过滤逻辑）
                if not cot_text.strip() or cot_text in ("'t", "''t"):
                    continue

                # 按问题ID分组
                if question_id not in self.question_samples:
                    self.question_samples[question_id] = {
                        "question": question_text,
                        "pos_cots": [],
                        "neg_cots": []
                    }
                if is_correct == 1:
                    self.question_samples[question_id]["pos_cots"].append(cot_text)
                else:
                    self.question_samples[question_id]["neg_cots"].append(cot_text)

        # 生成训练样本（关键：用空格分隔问题和CoT，移除[SEP]）
        self.train_samples = []
        self.max_len = max_len
        self.tokenizer = tokenizer  # 保存tokenizer用于长度校验

        for qid, data in self.question_samples.items():
            pos_cots = data["pos_cots"]
            neg_cots = data["neg_cots"]
            if len(pos_cots) == 0 or len(neg_cots) == 0:
                continue

            # 处理正样本
            for pos_cot in pos_cots:
                # 关键：用空格分隔，不新增特殊符号
                full_text = f"{data['question']} {pos_cot}"
                # 提前编码校验长度（避免后续训练报错）
                token_len = len(self.tokenizer.encode(full_text, truncation=False))
                if token_len > self.max_len:
                    print(f"⚠️ 样本{qid}（正）原长度{token_len}，将被截断至{self.max_len}")
                self.train_samples.append({
                    "qid": qid,
                    "text": full_text,
                    "label": 1
                })

            # 处理负样本
            for neg_cot in neg_cots:
                full_text = f"{data['question']} {neg_cot}"
                token_len = len(self.tokenizer.encode(full_text, truncation=False))
                if token_len > self.max_len:
                    print(f"⚠️ 样本{qid}（负）原长度{token_len}，将被截断至{self.max_len}")
                self.train_samples.append({
                    "qid": qid,
                    "text": full_text,
                    "label": 0
                })

        print(f"✅ 数据集加载完成：{len(self.question_samples)}个问题，{len(self.train_samples)}个训练样本")

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        sample = self.train_samples[idx]
        # 文本编码（严格截断，确保长度≤MAX_LEN）
        enc = self.tokenizer(
            sample["text"],
            truncation=True,  # 强制截断超长文本
            max_length=self.max_len,
            padding="max_length",  # 不足补全
            return_tensors="pt",
            add_special_tokens=True  # 使用LLaMA原生特殊符号（<s>开头，</s>结尾）
        )
        # 校验编码后长度（ debug 用，可删除）
        assert enc["input_ids"].shape[1] == self.max_len, f"编码后长度异常：{enc['input_ids'].shape[1]}"

        return {
            "qid": sample["qid"],
            "input_ids": enc["input_ids"].squeeze(0),  # [max_len]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [max_len]
            "label": sample["label"]
        }


# --------------------------
# 3. 模型封装（无修改，确保位置嵌入匹配）
# --------------------------
class LlamaEncoder(nn.Module):
    def __init__(self, model_name, dtype=torch.float16):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",  # 自动分配设备
            low_cpu_mem_usage=True,
            # 关键：显式指定模型最大序列长度（与Tokenizer一致）
            max_position_embeddings=4096
        )
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        # 冻结主体，训练投影层
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        # 取最后一层最后一个token的嵌入
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        seq_emb = last_hidden[:, -1, :]  # [B, H]
        seq_emb = self.proj(seq_emb)
        seq_emb = F.normalize(seq_emb, dim=-1)
        return seq_emb


# --------------------------
# 4. InfoNCE损失（无修改）
# --------------------------
def info_nce_loss(embeddings, labels, qids, temperature=0.07):
    B, H = embeddings.shape
    device = embeddings.device

    # 计算相似度矩阵
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    sim_matrix = sim_matrix - torch.eye(B, device=device) * 1e12  # 屏蔽自身

    # 构建正例掩码
    pos_mask = torch.zeros((B, B), device=device)
    for i in range(B):
        if labels[i] == 1:
            same_qid = (qids == qids[i])
            is_pos = (labels == 1)
            pos_mask[i] = (same_qid & is_pos).float()

    # 构建负例掩码
    neg_mask = 1 - pos_mask - torch.eye(B, device=device)
    neg_mask = neg_mask.clamp(0, 1)

    # 计算损失
    pos_score = (sim_matrix * pos_mask).sum(dim=1, keepdim=True)
    neg_score = (sim_matrix * neg_mask).view(B, -1)
    logits = torch.cat([pos_score, neg_score], dim=1)
    target = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, target)

    return loss


# --------------------------
# 5. 训练循环（添加CUDA调试开关）
# --------------------------
def train():
    # 步骤1：加载Tokenizer（关键：不新增[SEP]，用原生符号）
    print(">>> 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # LLaMA原生eos_token作为pad_token
    # 打印Tokenizer信息（ debug 用）
    print(f"Tokenizer信息：pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
    print(f"模型最大序列长度：{tokenizer.model_max_length}")

    # 步骤2：加载数据集
    print(">>> 加载数据集...")
    dataset = CoTDataset(DATASET_PATH, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 步骤3：加载模型
    print(">>> 加载LLaMA-2模型...")
    model = LlamaEncoder(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    if os.path.exists(SAVE_PATH):
        model.proj.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        print(f"✅ 加载已有权重：{SAVE_PATH}")

    # 步骤4：初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )
    grad_clip = torch.nn.utils.clip_grad_norm_
    max_grad_norm = 1.0

    # 步骤5：训练（关键：添加CUDA_BLOCKING，方便调试）
    print(">>> 开始训练...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} | Loss: ---")
        for batch in pbar:
            # 读取数据（添加blocking=True，确保CUDA错误定位）
            input_ids = batch["input_ids"].to(DEVICE, dtype=torch.long, non_blocking=False)
            attention_mask = batch["attention_mask"].to(DEVICE, dtype=torch.long, non_blocking=False)
            labels = batch["label"].to(DEVICE, dtype=torch.long, non_blocking=False)
            qids = batch["qid"]

            # 前向传播（捕获异常，打印详细信息）
            try:
                embeddings = model(input_ids, attention_mask)
            except Exception as e:
                print(f"\n❌ 前向传播错误：input_ids形状={input_ids.shape}, attention_mask形状={attention_mask.shape}")
                print(f"input_ids样本：{input_ids[0][:10]}...")  # 打印前10个Token，看是否异常
                raise e

            # 计算损失
            loss = info_nce_loss(embeddings, labels, qids, temperature=TEMPERATURE)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            grad_clip(model.parameters(), max_grad_norm)
            optimizer.step()

            # 累计损失
            total_loss += loss.item() * input_ids.size(0)
            pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

        # 保存模型
        epoch_avg_loss = total_loss / len(dataset)
        print(f"📊 Epoch {epoch+1} 完成 | 平均损失：{epoch_avg_loss:.4f}")
        torch.save(model.proj.state_dict(), SAVE_PATH)
        print(f"💾 模型保存至：{SAVE_PATH}\n")

    print(">>> 训练结束！")


# --------------------------
# 6. 主函数（添加CUDA调试环境变量）
# --------------------------
if __name__ == "__main__":
    # 固定随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # 关键：启用CUDA同步，确保错误定位准确（训练速度会变慢，调试完成后可注释）
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # 启用设备端断言，显示详细错误
    # 启动训练
    train()