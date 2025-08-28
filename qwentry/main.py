from config import *
from data_processor import read_pregen_cots, build_question_level_data, CCotDataset
from model import Encoder
from trainer import train, evaluate
import torch.optim as optim

def main():
    # 1. 数据准备：复用预生成CoT（核心步骤）
    cot_data = read_pregen_cots()  # 读取预生成CoT（含is_correct）
    question_data = build_question_level_data(cot_data)  # 构建题级数据
    
    # 2. 数据集与加载器
    dataset = CCotDataset(question_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    # 3. 模型与优化器初始化
    model = Encoder().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 4. 训练与评估循环
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        # 训练
        train_loss = train(model, dataloader, optimizer)
        # 评估
        baseline_acc, logic_acc, pass_rate = evaluate(model, dataloader)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Baseline Acc (All CoT): {baseline_acc:.4f}")
        print(f"Logic Acc (Correct CoT): {logic_acc:.4f}")
        print(f"Correct CoT Pass Rate: {pass_rate:.4f}")
    
    # 5. 保存模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "c_cot_model_no_logic.pt"))
    print(f"\n模型保存至：{os.path.join(OUTPUT_DIR, 'c_cot_model_no_logic.pt')}")

if __name__ == "__main__":
    main()