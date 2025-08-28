import os
import random
import torch

# 路径配置
QWEN_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
RAW_DEV_JSONL = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth2/PARARULE_Plus_Depth2_shuffled_dev_huggingface.jsonl"
PREGEN_JSONL = "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first100_flat_labeled.jsonl"  # 预生成CoT路径
OUTPUT_DIR = "/home2/zzl/C-CoT/results"
BERT_MODEL = "/home2/zzl/model/bert-base-uncased"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 生成参数（复用预生成时无需调整，仅实时生成用）
NUM_EXAMPLES = 50       # 处理样本数
N_SAMPLES = 4           # 每题生成的CoT路径数
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# 模型与训练参数
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MAX_LEN = 256
PROJ_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# 固定随机种子
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()