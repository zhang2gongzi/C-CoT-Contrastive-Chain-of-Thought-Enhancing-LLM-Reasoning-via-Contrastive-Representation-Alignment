# llama_client.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LlamaClient:
    def __init__(self, model_path: str):
        print(f"🚀 正在加载 Llama-2: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        print("✅ Llama-2 加载完成")

    def generate(self, prompt: str, max_new_tokens=512, temperature=0.7) -> str:
        try:
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            full_text = outputs[0]["generated_text"]
            return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return ""