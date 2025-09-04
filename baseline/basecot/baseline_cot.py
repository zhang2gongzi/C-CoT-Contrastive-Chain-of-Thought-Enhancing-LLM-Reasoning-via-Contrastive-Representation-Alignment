import torch
import json
import jsonlines
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import os
from tqdm import tqdm

# 配置路径和设备
DATASET_PATH = "/home2/zzl/C-CoT/database/gsm8k/test-00000-of-00001.parquet"
MODEL_PATH = "/home2/zzl/model/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "/home2/zzl/C-CoT/baseline/basecot/gsm8k_cot_results.jsonl"
MAX_SAMPLES = 100  # 只测试前100条数据

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 加载数据集
def load_gsm8k_dataset(path):
    """加载GSM8K测试集并只返回前100条数据"""
    dataset = load_dataset("parquet", data_files=path)
    # 只取前MAX_SAMPLES条数据
    return dataset["train"].select(range(min(MAX_SAMPLES, len(dataset["train"]))))

# 加载模型和tokenizer
def load_model_and_tokenizer(model_path, device):
    """加载模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用float16节省显存
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model, tokenizer

# 提取答案的函数
def extract_answer(text):
    """从模型输出中提取答案数字"""
    # 尝试匹配常见的答案模式，如"答案是：X"或"所以答案是X"
    patterns = [
        r"答案是：(\d+)",
        r"答案：(\d+)",
        r"所以答案是(\d+)",
        r"最终答案是(\d+)",
        r"(\d+)"  # 最后尝试提取任何数字作为备选
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return None  # 如果没有找到答案

# 基础CoT提示模板
def create_cot_prompt(question):
    """创建基础CoT提示"""
    cot_prompt = f"""请解决以下数学问题。在给出答案之前，详细展示你的推理过程。最后，用"答案是：X"的格式给出答案，其中X是你的最终答案。

问题：{question}

推理过程：
"""
    return cot_prompt

# 生成CoT推理
def generate_cot(model, tokenizer, question, device):
    """生成Chain-of-Thought推理过程"""
    prompt = create_cot_prompt(question)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)
    
    with torch.no_grad():  # 禁用梯度计算以节省内存
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 为推理过程分配足够的长度
            temperature=0.7,     # 控制随机性
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的文本（排除提示部分）
    generated_text = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):], 
        skip_special_tokens=True
    )
    
    return generated_text

# 评估答案正确性
def is_correct(predicted_answer, true_answer):
    """判断预测答案是否正确"""
    # 从真实答案中提取数字（GSM8K的答案格式通常是#### X）
    try:
        true_num = int(re.search(r"#### (\d+)", true_answer).group(1))
        return predicted_answer == true_num
    except:
        return False

# 主函数
def main():
    # 加载数据、模型和tokenizer
    print(f"加载数据集: {DATASET_PATH} (仅使用前{MAX_SAMPLES}条)")
    dataset = load_gsm8k_dataset(DATASET_PATH)
    print(f"实际加载样本数: {len(dataset)}")
    
    print(f"加载模型: {MODEL_PATH}")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, DEVICE)
    print(f"使用设备: {DEVICE}")
    
    # 记录正确数量和总数量
    total = 0
    correct = 0
    
    # 处理每个样本
    print("开始推理...")
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for example in tqdm(dataset, desc="处理样本", total=len(dataset)):
            question = example["question"]
            true_answer = example["answer"]
            
            # 生成CoT推理
            cot_reasoning = generate_cot(model, tokenizer, question, DEVICE)
            
            # 提取答案
            predicted_answer = extract_answer(cot_reasoning)
            
            # 判断是否正确
            accuracy = is_correct(predicted_answer, true_answer)
            if accuracy:
                correct += 1
            total += 1
            
            # 保存结果
            result = {
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "cot_reasoning": cot_reasoning,
                "correct": accuracy,
                "current_accuracy": correct / total if total > 0 else 0.0
            }
            writer.write(result)
            
            # 每处理10个样本打印一次当前准确率
            if total % 10 == 0:
                tqdm.write(f"处理进度: {total}/{len(dataset)}, 当前准确率: {correct/total:.4f}")
    
    # 计算并打印最终准确率
    final_accuracy = correct / total if total > 0 else 0.0
    print(f"\n最终准确率: {final_accuracy:.4f} ({correct}/{total})")
    
    # 将最终准确率追加到输出文件
    with jsonlines.open(OUTPUT_FILE, mode='a') as writer:
        writer.write({
            "final_accuracy": final_accuracy,
            "correct": correct,
            "total": total,
            "samples_used": MAX_SAMPLES
        })

if __name__ == "__main__":
    main()
    