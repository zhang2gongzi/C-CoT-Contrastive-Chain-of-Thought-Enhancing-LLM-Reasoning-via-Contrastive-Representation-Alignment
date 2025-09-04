# qwentry/config.py (REPLACEMENT)
import os
import random
import torch

# ---------------------
# 目录与模型路径（请根据本机修改）
# ---------------------
QWEN_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
RAW_DEV_JSONL = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth3/PARARULE_Plus_Depth5_shuffled_dev_huggingface.jsonl"
PREGEN_JSONL = "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first100_flat_labeled.jsonl"
OUTPUT_DIR = "/home2/zzl/C-CoT/results"
BERT_MODEL = "/home2/zzl/model/bert-base-uncased"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------
# 生成参数（如复用预生成CoT，可不用动）
# ---------------------
NUM_EXAMPLES = 50       # 处理样本数（若使用全部，请修改或用 None 逻辑）
N_SAMPLES = 4           # 每题生成的 CoT 路径数（dataset 会以此做 pad/trunc）
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------------------
# 训练/模型参数
# ---------------------
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MAX_LEN = 256
PROJ_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------------------
# multi-granularity 控制参数
# ---------------------
# step-level
MAX_STEPS = 8           # 每条 CoT 最多拆成多少 step（不足则 pad）
STEP_MAX_LEN = 64       # 每个 step 的最大 token 长度
STEP_CONTRAST_WEIGHT = 0.6

# token-level（用 attention 加权 mean pool 作为 token-level 表征中心）
USE_TOKEN_CONTRAST = True
TOKEN_CONTRAST_WEIGHT = 0.3
TOKEN_POOLING = "attn_mean"  # 当前仅实现 attn_mean pooling

# ---------------------
# 固定随机数种子（用于可复现）
# ---------------------
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
