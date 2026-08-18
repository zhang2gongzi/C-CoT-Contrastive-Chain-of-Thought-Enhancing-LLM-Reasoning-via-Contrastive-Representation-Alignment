import torch
import re
import json
import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from evaluate import load

# 配置参数 - 改为Llama-2模型路径
LLAMA_DIR = "${MODEL_DIR}/Llama-2-7b-chat-hf"
JSON_FILE_PATH = "${PROJECT_ROOT}/database/StrategyQA/strategyqa_train_filtered.json"
RESULTS_FILE = "llama2_strategyqa_results.json"
SAMPLE_SIZE =200 # 若为None则使用全部样本，否则指定样本数
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.5

def build_prompt(question):
    """构建适合Llama-2的提示词，确保模型输出指定格式的答案"""
    return f"""请回答以下问题，先给出推理过程，最后必须以"答案：是"或"答案：否"结束，其他格式无效。

问题：{question}

推理：
"""

def load_model(model_dir):
    """加载Llama-2模型和Tokenizer，带4-bit量化以节省显存，并设置正确的聊天模板"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载Llama-2的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Llama-2需要手动添加pad_token（通常使用eos_token作为pad_token）
    tokenizer.pad_token = tokenizer.eos_token
    
    # 设置Llama-2的聊天模板（遵循其对话格式要求）
    tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
<<SYS>>
{{ message['content'] }}
<</SYS>>

{% elif message['role'] == 'user' %}
{{ message['content'] }}

{% elif message['role'] == 'assistant' %}
{{ message['content'] + eos_token }}

{% endif %}
{% endfor %}"""
    
    # 加载Llama-2模型
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()  # 切换到评估模式
    return model, tokenizer

def generate_answer(model, tokenizer, prompt):
    """调用Llama-2模型生成回答"""
    # Llama-2通常需要系统提示来引导其行为
    system_prompt = "你是一个帮助回答问题的助手，需要进行多步推理后给出准确答案。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

def extract_answer(response):
    """从模型输出中提取答案，映射为布尔值"""
    if re.search(r"答案：是", response):
        return True
    elif re.search(r"答案：否", response):
        return False
    return None  # 无法解析的输出

def load_local_dataset(json_path):
    """加载本地JSON数组格式的数据集"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = Dataset.from_list(data)
        print(f"成功加载本地数据集，共{len(dataset)}条数据")
        return dataset
    except FileNotFoundError:
        print(f"错误：未找到文件 {json_path}")
        raise
    except json.JSONDecodeError:
        print(f"错误：文件 {json_path} 不是有效的JSON格式")
        raise
    except Exception as e:
        print(f"加载数据集时发生错误：{str(e)}")
        raise

def main():
    # 加载本地数据集
    print("===== 加载本地数据集 =====")
    dataset = load_local_dataset(JSON_FILE_PATH)
    
    # 采样（可选）
    if SAMPLE_SIZE and SAMPLE_SIZE < len(dataset):
        dataset = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))
    print(f"用于评估的样本数: {len(dataset)}")
    
    # 加载Llama-2模型
    print("\n===== 加载Llama-2模型 =====")
    model, tokenizer = load_model(LLAMA_DIR)
    print(f"模型运行设备: {model.device}")
    
    # 推理与评估
    print("\n===== 开始推理 =====")
    correct = 0
    total_samples = 0
    valid_samples = 0
    results = []
    
    for sample in tqdm.tqdm(dataset, desc="处理进度"):
        total_samples += 1
        question = sample["question"]
        true_answer = sample["answer"]
        
        # 生成回答
        prompt = build_prompt(question)
        response = generate_answer(model, tokenizer, prompt)
        model_answer = extract_answer(response)
        
        # 统计有效样本和正确样本
        if model_answer is not None:
            valid_samples += 1
            if model_answer == true_answer:
                correct += 1
        
        # 保存结果
        results.append({
            "qid": sample.get("qid", ""),
            "question": question,
            "true_answer": true_answer,
            "model_answer": model_answer,
            "response": response
        })
    
    # 保存结果到文件
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n推理结果已保存到 {RESULTS_FILE}")
    
    # 计算并输出评估指标
    accuracy = correct / valid_samples if valid_samples > 0 else 0.0
    
    print("\n===== 评估结果 =====")
    print(f"总样本数: {total_samples}")
    print(f"有效输出样本数（可解析答案）: {valid_samples}")
    print(f"正确样本数: {correct}")
    print(f"准确率: {accuracy:.2%}")

if __name__ == "__main__":
    main()
