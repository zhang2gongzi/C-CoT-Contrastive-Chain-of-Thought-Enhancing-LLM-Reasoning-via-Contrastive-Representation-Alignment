import os
import torch
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F


# ======================
# 数据集定义
# ======================
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256): # ✅ 降低默认 max_length
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row["question"]
        choices = row["choices"]
        label = row["answerKey"]  # e.g., "A", "B"

        text = f"Question: {question}\nChoices: {choices}\nAnswer: {label}"
        # 我们将在训练时只监督 "Answer: X" 中 X 的位置

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label_char": label  # 保留原始标签字符用于后续计算
        }


# ======================
# 安全加载 tokenizer（Qwen 专用）- ✅ 最终极修复版本
# ======================
def safe_tokenizer_load(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 打印初始状态
    print(f"🔍 初始状态: pad_token={repr(tokenizer.pad_token)}, pad_token_id={tokenizer.pad_token_id}")
    print(f"🔍 eos_token={repr(tokenizer.eos_token)}, eos_token_id={tokenizer.eos_token_id}")
    print(f"🔍 unk_token={repr(tokenizer.unk_token)}, unk_token_id={tokenizer.unk_token_id}")

    # Qwen 不允许添加新 token，所以我们只关注 pad_token_id
    # 尝试从 eos_token 或 unk_token 获取 ID
    pad_token_id = None
    if tokenizer.eos_token_id is not None:
        pad_token_id = tokenizer.eos_token_id
        print(f"✅ 使用 eos_token_id ({pad_token_id}) 作为 pad_token_id")
    elif tokenizer.unk_token_id is not None:
        pad_token_id = tokenizer.unk_token_id
        print(f"✅ 使用 unk_token_id ({pad_token_id}) 作为 pad_token_id")
    else:
        pad_token_id = 0  # 默认使用 0
        print(f"⚠️ 未找到 eos/unk_token_id，使用默认值 0 作为 pad_token_id")

    # 关键：不修改 tokenizer 的 pad_token 或 pad_token_id 属性
    # 而是在模型配置中设置，这样模型内部会使用这个 ID
    # 同时，我们在 collate_fn 中手动传递这个 ID
    print(f"✅ 最终决定使用的 pad_token_id: {pad_token_id}")
    return tokenizer, pad_token_id # 返回 pad_token_id


# ======================
# 找到 'Answer:' 后第一个 token 的位置，并获取其 label_id
# ======================
def find_answer_start_position(input_ids, attention_mask, tokenizer):
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # 解码整个序列
    decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

    answer_token = "Answer:"
    answer_ids = tokenizer.encode(answer_token, add_special_tokens=False)
    answer_len = len(answer_ids)

    start_positions = []

    for i, text in enumerate(decoded_texts):
        tokens = input_ids[i].tolist()
        pos = -1
        for j in range(len(tokens) - answer_len + 1):
            if tokens[j:j + answer_len] == answer_ids:
                pos = j + answer_len  # 紧跟在 "Answer:" 之后
                break
        start_positions.append(pos)

    return torch.tensor(start_positions, device=device)


# ======================
# 训练函数（含 Accuracy）
# ======================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer, pad_token_id_from_tokenizer = safe_tokenizer_load(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)

    # 设置 model.config.pad_token_id，以便模型内部使用
    model.config.pad_token_id = pad_token_id_from_tokenizer

    # 获取 A-E 的 token id
    choice_tokens = ["A", "B", "C", "D", "E"]
    choice_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in choice_tokens]
    print(f"✅ 识别的答案 token: {dict(zip(choice_tokens, choice_token_ids))}")

    df = pd.read_parquet(args.parquet_path)
    df = df.head(args.limit)
    print(f"📊 使用 {len(df)} 条数据进行训练")

    dataset = TextDataset(df, tokenizer, max_length=args.max_length) # ✅ 传入 max_length 参数
    # 注意：在 collate_fn 中，我们使用 pad_token_id_from_tokenizer
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'input_ids': torch.nn.utils.rnn.pad_sequence(
                [item['input_ids'] for item in x],
                batch_first=True,
                padding_value=pad_token_id_from_tokenizer  # ✅ 使用从 safe_tokenizer_load 获取的 ID
            ),
            'attention_mask': torch.nn.utils.rnn.pad_sequence(
                [item['attention_mask'] for item in x],
                batch_first=True,
                padding_value=0
            ),
            'label_char': [item['label_char'] for item in x]
        }
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.train()
    best_acc = 0.0

    for epoch in range(args.epochs):
        total_loss = 0
        total_acc = 0
        n_correct = 0
        n_total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_chars = batch["label_char"]

            # 找到 "Answer:" 之后的第一个 token 位置
            answer_start_positions = find_answer_start_position(input_ids, attention_mask, tokenizer)
            valid_mask = answer_start_positions >= 0
            if not valid_mask.any():
                continue

            input_ids = input_ids[valid_mask]
            attention_mask = attention_mask[valid_mask]
            answer_start_positions = answer_start_positions[valid_mask]
            label_chars = [c for c, m in zip(label_chars, valid_mask) if m]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

            # 取每个序列中 "Answer:" 后第一个 token 的 logits
            B = logits.size(0)
            pred_logits = logits[torch.arange(B), answer_start_positions]  # [B, V]

            # 只保留 A-E 选项的 logits
            pred_logits = pred_logits[:, choice_token_ids]  # [B, 5]
            pred_ids = pred_logits.argmax(dim=-1)  # [B]
            target_ids = torch.tensor([
                choice_tokens.index(c) for c in label_chars
            ], device=device)

            loss = F.cross_entropy(pred_logits, target_ids)
            acc = (pred_ids == target_ids).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            n_total += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}"
            })

        avg_loss = total_loss / n_total
        avg_acc = total_acc / n_total
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

        # 保存最佳模型
        if avg_acc > best_acc:
            best_acc = avg_acc
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir) # 保存 tokenizer 时，其 pad_token_id 仍为 None
            print(f"✅ 最佳模型已保存到 {args.output_dir} (Acc: {best_acc:.4f})")

    print(f"✅ 训练完成！最佳 Accuracy: {best_acc:.4f}")


# ======================
# 主入口
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径，如 Qwen/Qwen-1_8B")
    parser.add_argument("--parquet_path", type=str, required=True, help="训练数据 parquet 文件路径")
    parser.add_argument("--limit", type=int, default=200, help="限制训练样本数")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小") # ✅ 默认降低到 1
    parser.add_argument("--max_length", type=int, default=256, help="最大序列长度") # ✅ 添加 max_length 参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--output_dir", type=str, required=True, help="模型保存路径")

    args = parser.parse_args()
    train(args)