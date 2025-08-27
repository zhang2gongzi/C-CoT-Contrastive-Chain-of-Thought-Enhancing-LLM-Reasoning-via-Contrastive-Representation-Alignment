from config import *
from data_processor import read_raw_data, generate_cots, build_question_level_data, CCotDataset
from model import Encoder
from trainer import train, evaluate
import torch.optim as optim

def main():
    # 1. 数据准备
    raw_data = read_raw_data(RAW_DEV_JSONL)
    cot_data = generate_cots(raw_data)  # 生成CoT
    question_data = build_question_level_data(cot_data)  # 构建题级数据
    
    # 2. 数据集与加载器
    dataset = CCotDataset(question_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    # 3. 模型与优化器
    model = Encoder().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 4. 训练与评估
    for epoch in range(EPOCHS):
        train_loss = train(model, dataloader, optimizer)
        baseline_acc, logic_acc, pass_rate = evaluate(model, dataloader)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Baseline Acc: {baseline_acc:.4f}")
        print(f"Logic-Validated Acc: {logic_acc:.4f}")
        print(f"Logic Pass Rate: {pass_rate:.4f}")
    
    # 5. 保存模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "c_cot_model.pt"))
    print(f"模型保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()