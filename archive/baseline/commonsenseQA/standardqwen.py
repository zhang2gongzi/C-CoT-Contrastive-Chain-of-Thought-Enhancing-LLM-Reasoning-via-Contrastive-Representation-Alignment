import os
import torch
import pandas as pd
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

# 配置路径
MODEL_PATH = "${MODEL_DIR}/../model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
TEST_DATA_PATH = "${PROJECT_ROOT}/database/commonsenseQA/test-00000-of-00001.parquet"
TRAIN_DATA_PATH = "${PROJECT_ROOT}/database/commonsenseQA/train-00000-of-00001.parquet"
OUTPUT_DIR = "${PROJECT_ROOT}/baseline/commonsenseQA/cot_results"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型和分词器
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    # 配置生成参数
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

# 加载数据集
def load_dataset(file_path, sample_size=None):
    df = pd.read_parquet(file_path)
    # 如果指定了样本量，只取部分数据
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    return df

# 构建标准CoT提示
def build_cot_prompt(question, choices):
    # 格式化选项
    choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    # 标准CoT提示模板
    prompt = f"""
    Answer the following question with step-by-step reasoning before giving your final answer.
    
    Question: {question}
    Options:
    {choices_str}
    
    Let's think step by step:
    """
    return prompt.strip()

# 生成CoT推理结果
def generate_cot_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成配置
    generation_kwargs = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(**inputs,** generation_kwargs)
    
    # 解码并提取生成的文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取推理部分（去掉原始提示）
    cot_part = response[len(prompt):].strip()
    
    return cot_part

# 主函数
def main():
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # 加载测试集（可以先使用小样本测试）
    print("Loading dataset...")
    test_df = load_dataset(TEST_DATA_PATH, sample_size=10)  # 先测试10个样本
    
    # 处理每个样本
    results = []
    for idx, row in test_df.iterrows():
        print(f"Processing sample {idx+1}/{len(test_df)}")
        
        # 构建提示
        prompt = build_cot_prompt(row['question'], row['choices'])
        
        # 生成CoT推理
        cot_response = generate_cot_response(model, tokenizer, prompt)
        
        # 保存结果
        results.append({
            'id': row.get('id', idx),
            'question': row['question'],
            'choices': row['choices'],
            'cot_reasoning': cot_response
        })
        
        # 每处理5个样本保存一次中间结果
        if (idx + 1) % 5 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_json(f"{OUTPUT_DIR}/cot_results_temp.jsonl", orient="records", lines=True)
    
    # 保存最终结果
    final_df = pd.DataFrame(results)
    final_df.to_json(f"{OUTPUT_DIR}/cot_results_final.jsonl", orient="records", lines=True)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
    