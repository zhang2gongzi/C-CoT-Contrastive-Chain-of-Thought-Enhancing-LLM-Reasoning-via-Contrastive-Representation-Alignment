import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from pyDatalog import pyDatalog

# =========================
# 1. 数据集定义
# =========================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_text": self.texts[idx]
        }

# =========================
# 2. 模型定义
# =========================
class BertClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("/home2/zzl/model/bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# =========================
# 3. 逻辑验证函数
# =========================
def logic_check(text):
    """
    简单逻辑验证：检查文本中是否存在矛盾的事实。
    这里仅示例：包含 '不可能' 或 '矛盾' 的句子判定为 False
    """
    if "不可能" in text or "矛盾" in text:
        return False
    return True

# =========================
# 4. 训练函数（逻辑掩码替换）
# =========================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        texts = batch["raw_text"]

        # 逻辑掩码：不合法样本丢弃
        logic_mask = torch.tensor([logic_check(t) for t in texts], dtype=torch.bool).to(device)

        if logic_mask.sum() == 0:
            continue  # 全部不合法就跳过

        input_ids = input_ids[logic_mask]
        attention_mask = attention_mask[logic_mask]
        labels = labels[logic_mask]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# =========================
# 5. 验证函数（三项指标）
# =========================
def evaluate(model, dataloader, device):
    model.eval()
    correct_baseline, correct_logic, total = 0, 0, 0
    passed = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["raw_text"]

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            # baseline
            correct_baseline += (preds == labels).sum().item()
            total += labels.size(0)

            # logic filtering
            for i, text in enumerate(texts):
                if logic_check(text):
                    passed += 1
                    if preds[i] == labels[i]:
                        correct_logic += 1

    baseline_acc = correct_baseline / total
    if passed > 0:
        logic_acc = correct_logic / passed
        pass_rate = passed / total
    else:
        logic_acc, pass_rate = 0.0, 0.0

    return baseline_acc, logic_acc, pass_rate

# =========================
# 6. 主程序
# =========================
def main():
    # toy 数据
    texts = [
        "这个案例完全合理",
        "这句话存在矛盾",
        "一切都不可能发生",
        "逻辑通顺，没有问题"
    ]
    labels = [1, 0, 0, 1]

    tokenizer = BertTokenizer.from_pretrained("/home2/zzl/model/bert-base-uncased")
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier(num_labels=2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(3):
        train_loss = train(model, dataloader, optimizer, criterion, device)
        baseline_acc, logic_acc, pass_rate = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Baseline Acc={baseline_acc:.4f}, "
              f"Logic Acc={logic_acc:.4f}, Pass Rate={pass_rate:.4f}")

if __name__ == "__main__":
    main()
