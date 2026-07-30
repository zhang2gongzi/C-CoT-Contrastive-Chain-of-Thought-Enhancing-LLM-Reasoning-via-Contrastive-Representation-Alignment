import os
import re
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


def load_gsm8k_data(file_path, num_samples=200):
    try:
        parquet_file = pq.ParquetFile(file_path)
        df = parquet_file.read().to_pandas()
        df_subset = df[["question", "answer"]].head(num_samples).reset_index(drop=True)
        print(f"成功加载GSM8K数据，共{len(df_subset)}条样本")
        return df_subset
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        raise


def build_contrastive_prompt(question):
    """构建对比思维链提示词（包含Few-shot示例）"""
    contrastive_examples = """
You are a math problem solver trained with contrastive chain-of-thought reasoning. 
First, learn from the positive (correct) and negative (incorrect) examples to understand valid reasoning and avoid mistakes.
Then solve the target problem step by step, and conclude with your final answer.

--- Example 1 ---
Problem: There are 25 students in a class. 12 are boys and the rest are girls. 5 girls wear glasses. How many girls do not wear glasses?

Positive Chain (Correct Reasoning):
1. Total students: 25. Boys: 12. So girls = total - boys = 25 - 12 = 13.
2. Girls who wear glasses: 5. So girls who do not wear glasses = total girls - girls with glasses = 13 - 5 = 8.
Conclusion: 8

Negative Chain (Incorrect Reasoning):
1. Total students: 25. Boys: 12. Mistakenly calculate girls as 25 + 12 = 37 (wrong: should subtract, not add).
2. Girls who wear glasses: 5. So girls without glasses = 37 - 5 = 32.
Conclusion: 32 (Mistake: Incorrectly added boys to total instead of subtracting)

--- Target Problem ---
Problem: {question}

Reasoning Chain:
"""
    return contrastive_examples.format(question=question)


def load_llama2_model(model_path):
    """加载Llama-2模型并手动设置对话模板"""
    try:
        print(f"开始加载模型：{model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 手动设置Llama-2对话模板（关键：避免依赖内置模板）
        tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'user' %}
<s>[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }} </s>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<s>[INST] 
{% endif %}
"""
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=False  # 显存不足时设为True
        )
        model.eval()
        print("模型加载完成")
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        raise


def run_inference(df, model, tokenizer, output_path):
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="推理中"):
        question = row["question"]
        ground_truth = row["answer"]

        # 1. 构建提示词
        prompt = build_contrastive_prompt(question)

        # 2. 手动构建对话格式（彻底避免apply_chat_template问题）
        # Llama-2对话格式：<s>[INST] 用户输入 [/INST]
        formatted_prompt_str = f"<s>[INST] {prompt} [/INST]"

        # 生成张量格式输入
        formatted_prompt = tokenizer(
            text=formatted_prompt_str,
            return_tensors="pt",
            padding="max_length",
            max_length=2048,
            truncation=True
        ).to(model.device)

        # 3. 模型推理
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=formatted_prompt["input_ids"],
                    attention_mask=formatted_prompt["attention_mask"],
                    max_new_tokens=1024,
                    temperature=0.2,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
        except Exception as e:
            print(f"推理失败（index={idx}）：{str(e)}")
            model_reasoning = "推理错误"
            model_answer = "错误"
        else:
            generated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

            # 截断输入部分（基于手动构建的prompt）
            if formatted_prompt_str in generated_text:
                input_len = generated_text.index(formatted_prompt_str) + len(formatted_prompt_str)
                model_reasoning = generated_text[input_len:].strip()
            else:
                model_reasoning = generated_text

            # 提取答案
            if "Conclusion:" in model_reasoning:
                model_answer = model_reasoning.split("Conclusion:")[-1].strip()
            else:
                numbers = re.findall(r"\b\d+\b", model_reasoning)
                model_answer = numbers[-1] if numbers else "无答案"

        # 保存结果
        results.append({
            "index": idx,
            "question": question,
            "ground_truth": ground_truth,
            "model_reasoning": model_reasoning,
            "model_answer": model_answer
        })

    # 保存与评估
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"推理完成，结果已保存至：{output_path}")

    def extract_final_number(text):
        numbers = re.findall(r"\b\d+\b", text)
        return numbers[-1] if numbers else ""

    results_df["pred_num"] = results_df["model_answer"].apply(extract_final_number)
    results_df["truth_num"] = results_df["ground_truth"].apply(extract_final_number)
    accuracy = (results_df["pred_num"] == results_df["truth_num"]).mean()
    print(f"初步准确率（数字匹配）：{accuracy:.2%}")

    return results_df


if __name__ == "__main__":
    # 升级transformers（若未升级，取消下面一行注释）
    # os.system("pip install --upgrade transformers")
    
    DATA_PATH = "/home2/zzl/C-CoT/database/gsm8k/train-00000-of-00001.parquet"
    MODEL_PATH = "/home2/zzl/model/Llama-2-7b-chat-hf"
    OUTPUT_PATH = "/home2/zzl/C-CoT/results/llama2_7b_contrastive_cot_gsm8k_200.csv"
    NUM_SAMPLES = 200

    df = load_gsm8k_data(DATA_PATH, NUM_SAMPLES)
    model, tokenizer = load_llama2_model(MODEL_PATH)
    results_df = run_inference(df, model, tokenizer, OUTPUT_PATH)