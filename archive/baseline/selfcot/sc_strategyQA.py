import torch
import re
import json
import tqdm
from datasets import Dataset
from collections import Counter
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)

# 配置参数
QWEN_DIR = "${MODEL_DIR}/../model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
JSON_FILE_PATH = "${PROJECT_ROOT}/database/StrategyQA/strategyqa_train_filtered.json"
RESULTS_FILE = "${PROJECT_ROOT}/baseline/selfcot/qwen_strategyqa_sc_results.json"
SAMPLE_SIZE = 200  # 若为None则使用全部样本
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7  # SC方法需要较高温度以产生多样性
NUM_GENERATIONS = 5  # 每个问题生成的推理路径数量
MAJORITY_THRESHOLD = 0.5  # 多数投票阈值

def build_prompt(question):
    """构建提示词，引导模型进行多步推理"""
    return f"""请仔细思考以下问题，先进行详细的多步推理，然后给出最终答案。
务必在回答的最后用"答案：是"或"答案：否"明确表示你的结论。

问题：{question}

推理过程：
"""

def load_model(model_dir):
    """加载Qwen模型和Tokenizer"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # 设置Qwen的聊天模板
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer

def generate_sc_answers(model, tokenizer, prompt, num_generations=5):
    """使用Self-Consistency方法生成多个推理路径和答案"""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    all_responses = []
    all_answers = []
    
    for _ in range(num_generations):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,  # SC方法需要采样以产生多样性
                top_p=0.95
            )
        
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        answer = extract_answer(response)
        
        all_responses.append(response)
        all_answers.append(answer)
    
    return all_responses, all_answers

def extract_answer(response):
    """从模型输出中提取答案"""
    if re.search(r"答案：是", response):
        return True
    elif re.search(r"答案：否", response):
        return False
    return None  # 无法解析的输出

def majority_vote(answers):
    """多数投票确定最终答案"""
    # 过滤无效答案
    valid_answers = [a for a in answers if a is not None]
    
    if not valid_answers:
        return None, {}
    
    # 统计投票结果
    vote_counts = Counter(valid_answers)
    # 找到得票最多的答案
    final_answer = vote_counts.most_common(1)[0][0]
    
    return final_answer, vote_counts

def load_local_dataset(json_path):
    """加载本地数据集"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = Dataset.from_list(data)
        print(f"成功加载本地数据集，共{len(dataset)}条数据")
        return dataset
    except Exception as e:
        print(f"加载数据集时发生错误：{str(e)}")
        raise

def main():
    # 加载本地数据集
    print("===== 加载本地数据集 =====")
    dataset = load_local_dataset(JSON_FILE_PATH)
    
    # 采样
    if SAMPLE_SIZE and SAMPLE_SIZE < len(dataset):
        dataset = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))
    print(f"用于评估的样本数: {len(dataset)}")
    
    # 加载模型
    print("\n===== 加载Qwen模型 =====")
    model, tokenizer = load_model(QWEN_DIR)
    print(f"模型运行设备: {model.device}")
    
    # 推理与评估
    print("\n===== 开始推理 (Self-Consistency) =====")
    correct = 0
    total_samples = 0
    valid_samples = 0
    results = []
    
    for sample in tqdm.tqdm(dataset, desc="处理进度"):
        total_samples += 1
        question = sample["question"]
        true_answer = sample["answer"]
        
        # 生成多个推理路径和答案
        prompt = build_prompt(question)
        all_responses, all_answers = generate_sc_answers(
            model, 
            tokenizer, 
            prompt, 
            num_generations=NUM_GENERATIONS
        )
        
        # 多数投票确定最终答案
        final_answer, vote_counts = majority_vote(all_answers)
        
        # 统计结果
        if final_answer is not None:
            valid_samples += 1
            if final_answer == true_answer:
                correct += 1
        
        # 保存详细结果
        results.append({
            "qid": sample.get("qid", ""),
            "question": question,
            "true_answer": true_answer,
            "all_responses": all_responses,
            "all_answers": all_answers,
            "vote_counts": dict(vote_counts),
            "final_answer": final_answer,
            "is_correct": final_answer == true_answer if final_answer is not None else None
        })
    
    # 保存结果到文件
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n推理结果已保存到 {RESULTS_FILE}")
    
    # 计算并输出评估指标
    accuracy = correct / valid_samples if valid_samples > 0 else 0.0
    
    print("\n===== 评估结果 =====")
    print(f"总样本数: {total_samples}")
    print(f"有效输出样本数: {valid_samples}")
    print(f"正确样本数: {correct}")
    print(f"准确率: {accuracy:.2%}")
    print(f"每个问题生成的推理路径数: {NUM_GENERATIONS}")

if __name__ == "__main__":
    main()
