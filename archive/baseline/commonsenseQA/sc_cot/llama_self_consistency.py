import os
import torch
import pandas as pd
import re
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# -------------------------- 1. 路径与参数配置 --------------------------
MODEL_PATH = "${MODEL_DIR}/Llama-2-7b-chat-hf"
DATA_PATH = "${PROJECT_ROOT}/database/commonsenseQA/train-00000-of-00001.parquet"
OUTPUT_DIR = "${PROJECT_ROOT}/baseline/commonsenseQA/sc_cot/llama_sc_results"
SAMPLE_SIZE = 200  # 样本数量
NUM_GENERATIONS = 5  # 每个问题生成的推理路径数量（自一致性核心参数）
ANSWER_COLUMN = "answerKey"


# -------------------------- 2. 工具函数 --------------------------
def load_model_and_tokenizer(model_path):
    """加载Llama模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def load_dataset(file_path, sample_size=None):
    """加载并预处理数据集，转换choices格式"""
    df = pd.read_parquet(file_path)
    
    # 转换choices格式（处理NumPy数组）
    df["choices"] = df["choices"].apply(convert_choices_format)
    
    # 抽样
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df


def convert_choices_format(choices):
    """将NumPy数组格式转换为标准字典列表"""
    try:
        labels = choices["label"]
        texts = choices["text"]
        
        # 转换为普通列表
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        return [{"label": labels[i], "text": texts[i]} for i in range(len(labels))]
    except Exception as e:
        raise ValueError(f"转换choices格式失败：{str(e)}")


def build_sc_prompt(question, choices):
    """构建自一致性提示词"""
    choices_str = ""
    for choice in choices:
        label = choice.get("label", "").upper()
        text = choice.get("text", "")
        if label and text:
            choices_str += f"{label}. {text}\n"
    
    if not choices_str:
        raise ValueError(f"无法解析choices格式")
    
    # Llama对话模板
    prompt = f"""[INST] Answer the following question with step-by-step reasoning.
Finally, you MUST output the answer in the format "Final Answer: X" (X is A/B/C/D/E), no extra content.

Question: {question}
Options:
{choices_str.strip()}

Let's think step by step: [/INST]"""
    return prompt.strip()


def extract_answer(cot_response, choices):
    """从推理中提取答案"""
    valid_labels = [choice.get("label", "").upper() for choice in choices if choice.get("label")]
    
    # 匹配"Final Answer: X"
    lines = [line.strip() for line in cot_response.split("\n") if line.strip()]
    final_line = None
    for line in reversed(lines):
        if line.lower().startswith("final answer:"):
            final_line = line
            break
    
    if final_line:
        answer = final_line.split(":", 1)[1].strip().upper()
        if answer in valid_labels:
            return answer
    
    # 备选匹配
    pattern = r"answer\s*(is|:\s*)\s*([A-E])"
    match = re.search(pattern, cot_response, re.IGNORECASE)
    if match and match.group(2).upper() in valid_labels:
        return match.group(2).upper()
    
    return None  # 无法提取时返回None


def majority_vote(answers):
    """多数投票确定最终答案"""
    # 过滤None值
    valid_answers = [ans for ans in answers if ans is not None]
    if not valid_answers:
        return "Unknown"
    
    # 多数投票
    vote_counts = Counter(valid_answers)
    return vote_counts.most_common(1)[0][0]


# -------------------------- 3. 自一致性推理主流程 --------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== 1. 加载Llama模型与分词器 ===")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    print("\n=== 2. 加载数据集 ===")
    df = load_dataset(DATA_PATH, sample_size=SAMPLE_SIZE)
    print(f"加载完成：{len(df)} 个样本，每个样本生成 {NUM_GENERATIONS} 条推理路径\n")
    
    results = []
    print("=== 3. 自一致性推理 ===")
    
    for sample_idx in range(len(df)):
        row = df.iloc[sample_idx]
        question_id = row["id"]
        print(f"\n【样本 {sample_idx+1}/{len(df)}】ID: {question_id}")
        
        # 获取标准答案
        ground_truth = row[ANSWER_COLUMN].upper()
        
        # 构建提示词
        try:
            prompt = build_sc_prompt(row["question"], row["choices"])
        except ValueError as e:
            print(f"⚠️  构建提示词失败：{e}，跳过该样本")
            continue
        
        # 生成多个推理路径（自一致性核心）
        all_answers = []
        all_reasonings = []
        
        for gen_idx in range(NUM_GENERATIONS):
            print(f"  生成推理路径 {gen_idx+1}/{NUM_GENERATIONS}...")
            
            # 每次生成使用不同的随机种子（增加多样性）
            generation_config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,  # 非零温度增加随机性
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                seed=42 + gen_idx  # 不同种子确保多样性
            )
            
            # 生成推理
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)
            
            # 解码
            prompt_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])
            cot_response = tokenizer.decode(
                outputs[0][prompt_length:],
                skip_special_tokens=True
            ).strip()
            
            # 提取答案
            answer = extract_answer(cot_response, row["choices"])
            all_answers.append(answer)
            all_reasonings.append(cot_response)
            
            # 打印当前推理的答案
            print(f"  推理路径 {gen_idx+1} 答案：{answer if answer else '无法提取'}")
        
        # 多数投票确定最终答案
        final_answer = majority_vote(all_answers)
        is_correct = (final_answer == ground_truth)
        
        # 统计各答案出现次数
        vote_counts = Counter([ans for ans in all_answers if ans is not None])
        print(f"\n  投票结果：{dict(vote_counts)}")
        print(f"  最终答案：{final_answer} | 标准答案：{ground_truth} | {'正确' if is_correct else '错误'}")
        
        # 保存结果
        results.append({
            "sample_idx": sample_idx,
            "question_id": question_id,
            "question": row["question"],
            "choices": row["choices"],
            "ground_truth": ground_truth,
            "num_generations": NUM_GENERATIONS,
            "all_reasonings": all_reasonings,
            "all_answers": all_answers,
            "vote_counts": dict(vote_counts),
            "final_answer": final_answer,
            "is_correct": is_correct
        })
    
    # 计算整体准确度
    correct_count = sum(1 for res in results if res["is_correct"])
    total_count = len(results)
    accuracy = round(correct_count / total_count, 4) if total_count > 0 else 0.0
    
    print(f"\n=== 4. 自一致性推理结果统计 ===")
    print(f"总样本数：{total_count}")
    print(f"正确样本数：{correct_count}")
    print(f"准确度：{accuracy}（{correct_count}/{total_count}）")
    print(f"每个样本生成的推理路径数：{NUM_GENERATIONS}")
    
    # 保存完整结果
    save_path = f"{OUTPUT_DIR}/llama_self_consistency_results.jsonl"
    pd.DataFrame(results).to_json(save_path, orient="records", lines=True, force_ascii=False)
    print(f"\n结果保存至：{save_path}")


if __name__ == "__main__":
    main()
    