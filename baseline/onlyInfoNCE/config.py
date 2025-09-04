import torch

# 1. 设备配置（自动检测GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # 数据加载线程数（建议设为CPU核心数的1/2）

# 2. 数据路径（替换为你的数据实际路径）
DATA_PATH = "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first100_flat_labeled_depth5.jsonl"  # 你的JSON Lines文件

# 3. 模型配置
BERT_MODEL_NAME = "/home2/zzl/model/bert-base-uncased"  # 预训练模型（可换bert-large-uncased）
MAX_SEQ_LEN = 512  # 推理链最大长度（你的数据中推理链较长，设512足够）
PROJECT_DIM = 128  # InfoNCE投影层输出维度（BERT 768维→128维）

# 4. 训练配置
BATCH_SIZE = 16  # 根据GPU显存调整（16G显存建议16，24G建议32）
EPOCHS = 10  # 训练轮数
LR = 2e-5  # BERT类模型最优学习率（2e-5~5e-5）
WEIGHT_DECAY = 1e-4  # 权重衰减（防止过拟合）
WARMUP_RATIO = 0.1  # 学习率预热比例（前10%轮次线性升温）

# 5. InfoNCE损失配置
INFONCE_TEMP = 0.07  # 温度参数（控制相似度分布，默认0.07）

# 6. 保存与日志配置
SAVE_MODEL = True
MODEL_SAVE_PATH = "/home2/zzl/C-CoT/baseline/onlyInfoNCE/saved_model/infonce_best.pt"  # 模型保存路径
LOG_INTERVAL = 20  # 每10个batch打印一次训练日志