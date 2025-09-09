import os
import re
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)


class QwenGSM8KEvaluator:
    def __init__(self):
        # ===== 参数 =====
        self.QWEN_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
        self.GSM8K_PARQUET_PATH = "/home2/zzl/C-CoT/database/gsm8k/test-00000-of-00001.parquet"
        self.OUTPUT_DIR = "./results_gsm8k"
        self.BERT_MODEL = "/home2/zzl/model/bert-base-uncased"

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # ===== 加载 Qwen 模型和分词器 =====
        print(">>> 正在加载 Qwen-7B-Chat ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.QWEN_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.QWEN_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # ===== 修复 pad_token =====
        if self.tokenizer.pad_token is None:
            print(">>> tokenizer.pad_token 未设置，修复中 ...")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ===== 生成配置 =====
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    # ---------- 数据 ----------
    def load_dataset(self):
        print(f">>> 读取 GSM8K 数据集: {self.GSM8K_PARQUET_PATH}")
        df = pd.read_parquet(self.GSM8K_PARQUET_PATH)
        print(f">>> 数据集大小: {len(df)} 条")
        return df

    @staticmethod
    def extract_numeric_answer(answer: str) -> str:
        """提取 GSM8K 答案中的最终数字"""
        match = re.search(r"####\s*(\d+)", answer)
        if match:
            return match.group(1)
        numbers = re.findall(r"\b\d+\b", answer)
        return numbers[-1] if numbers else ""

    @staticmethod
    def extract_pred_answer(text: str) -> str:
        """提取模型输出中的最终数字"""
        numbers = re.findall(r"\b\d+\b", text)
        if numbers:
            return numbers[-1]
        m = re.search(r"Answer[:\s]*(\d+)", text, re.IGNORECASE)
        if m:
            return m.group(1)
        return "Unknown"

    # ---------- 推理 ----------
    def generate_answer(self, question: str) -> str:
        prompt = f"Answer the following math problem step by step and give the final answer:\n\n{question}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

    # ---------- 主流程 ----------
    def evaluate(self, max_test: int = 100):
        df = self.load_dataset()
        results = []
        correct = 0
        total = 0

        for idx, row in df.iterrows():
            if idx >= max_test:
                break

            qid = row.get("id", idx)
            question = row["question"]
            gold = self.extract_numeric_answer(row["answer"])

            try:
                generated_answer = self.generate_answer(question)
                pred = self.extract_pred_answer(generated_answer)
                ok = (pred == gold)
            except Exception as e:
                generated_answer = f"[ERROR] {e}"
                pred, ok = "ERROR", False

            results.append({
                "id": qid,
                "question": question,
                "gold": gold,
                "pred_raw": generated_answer,
                "pred": pred,
                "correct": ok,
            })

            total += 1
            correct += int(ok)

            if (idx + 1) % 10 == 0:
                print(f">>> 已处理 {idx+1}/{len(df)} 条, 当前准确率: {correct/total:.2%}")

        # 保存结果
        output_path = os.path.join(self.OUTPUT_DIR, "gsm8k_qwen_eval.csv")
        pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
        print(f">>> 结果已保存到 {output_path}")
        print(f">>> 最终准确率: {correct}/{total} = {correct/total:.2%}")


if __name__ == "__main__":
    evaluator = QwenGSM8KEvaluator()
    evaluator.evaluate(max_test=100)  # 可以改成全量
