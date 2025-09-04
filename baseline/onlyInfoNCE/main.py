import torch
import torch.optim as optim
from config import *
from data_processor import get_dataloaders  # 修正函数名称
from model import CotEncoder  # 注意模型类名已改为CotEncoder
from trainer import train_one_epoch, evaluate, train_model  # 调整导入的函数

def main():
    # 准备数据加载器
    print("准备数据...")
    train_loader, val_loader, test_loader = get_dataloaders()  # 修正函数名称
    
    # 初始化模型
    print("初始化模型...")
    model = CotEncoder().to(DEVICE)  # 模型类名已改为CotEncoder
    
    # 开始训练
    print("开始训练...")
    train_model(model, train_loader, val_loader)  # 调用修正后的训练函数
    
    # 最终测试
    print("进行最终测试...")
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"\n测试结果：损失={test_loss:.4f}，推理链正确率={test_acc:.4f}")

if __name__ == "__main__":
    main()
    