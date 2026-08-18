# config.py
import os
import torch

class Config:
    def __init__(self):
        self.model_name = "llama2"
        self.model_path = "${MODEL_DIR}/Llama-2-7b-chat-hf"
        self.data_dir = "${PROJECT_ROOT}/database/StrategyQA"
        self.output_dir = "./outputs_llama"
        self.num_paths = 4
        self.temperature = 0.7
        self.max_new_tokens = 512
        self.batch_size = 4
        self.num_epochs = 10
        self.lr = 2e-5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.output_dir, exist_ok=True)