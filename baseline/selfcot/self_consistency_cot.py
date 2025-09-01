import json
import re
import random
from collections import defaultdict
from tqdm import tqdm
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer

# 配置参数
class Config:
    # 模型路径
    QWEN_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
    BERT_MODEL = "/home2/zzl/model/bert-base-uncased"
    
    # 数据路径
    RAW_DATA_PATH = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth5/PARARULE_Plus_Depth5_shuffled_dev_huggingface.jsonl"
    
    # 实验参数
    NUM_SAMPLES = 5  # 每条问题采样的CoT数量
    MAX_NEW_TOKENS = 256  # 生成CoT的最大长度
    TEMPERATURE = 0.7  # 控制采样随机性，较高的值会增加多样性
    TOP_P = 0.95       #  nucleus sampling参数
    NUM_EXAMPLES = 100 # 实验样本数量
    SEED = 42          # 随机种子，保证可复现性

# 设置随机种子
def set_seed(seed=Config.SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 解析CoT中的答案
YES_PAT = re.compile(r"\b(answer[:：]?\s*)?(yes|true)\b", re.I)
NO_PAT = re.compile(r"\b(answer[:：]?\s*)?(no|false)\b", re.I)

def parse_answer(cot_text):
    """从CoT文本中解析出答案（1表示yes，0表示no，-1表示无法解析）"""
    cot_text = cot_text.lower()
    if YES_PAT.search(cot_text):
        return 1
    if NO_PAT.search(cot_text):
        return 0
    return -1

# 加载原始数据
def load_raw_data(path, num_examples=Config.NUM_EXAMPLES):
    """加载原始问题数据"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            item = json.loads(line)
            data.append({
                "id": item["id"],
                "context": item["context"],
                "question": item["question"],
                "gold_label": item["label"]
            })
    return data

# 生成多条CoT
def generate_multiple_cots(data, model, tokenizer):
    """为每条问题生成多条CoT推理链"""
    results = []
    for item in tqdm(data, desc="生成CoT推理链"):
        context = item["context"]
        question = item["question"]
        
        # 构建提示词
        prompt = f"""Context: {context}
Q: {question}
Let's reason step by step using the facts and rules in the context, then answer "yes" or "no".
Reasoning:"""
        
        # 编码提示词
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成多条CoT
        cots = []
        with torch.no_grad():
            for _ in range(Config.NUM_SAMPLES):
                # 每次生成使用不同的随机种子增加多样性
                torch.manual_seed(Config.SEED + len(cots))
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=Config.TEMPERATURE,
                    top_p=Config.TOP_P,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # 提取生成的CoT
                cot = tokenizer.decode(outputs[0], skip_special_tokens=True)
                cot = cot[len(prompt):].strip()  # 去除提示词部分
                
                # 解析答案
                answer = parse_answer(cot)
                
                cots.append({
                    "cot": cot,
                    "answer": answer
                })
        
        results.append({
            "id": item["id"],
            "question": question,
            "gold_label": item["gold_label"],
            "cots": cots
        })
    
    return results

# 执行自一致性投票
def self_consistency_voting(results):
    """对多条CoT的答案进行投票，选择多数答案作为最终预测"""
    metrics = {
        "total": 0,
        "correct": 0,
        "invalid": 0  # 所有CoT都无法解析答案的样本数
    }
    
    detailed_results = []
    
    for item in results:
        metrics["total"] += 1
        
        # 收集有效答案
        valid_answers = [cot["answer"] for cot in item["cots"] if cot["answer"] != -1]
        
        if not valid_answers:
            # 没有有效答案
            metrics["invalid"] += 1
            pred_label = -1
        else:
            # 投票选择多数答案
            answer_counts = defaultdict(int)
            for ans in valid_answers:
                answer_counts[ans] += 1
            pred_label = max(answer_counts.items(), key=lambda x: x[1])[0]
            
            # 检查是否正确
            if pred_label == item["gold_label"]:
                metrics["correct"] += 1
        
        detailed_results.append({
            "id": item["id"],
            "question": item["question"],
            "gold_label": item["gold_label"],
            "pred_label": pred_label,
            "answer_counts": dict(answer_counts),
            "is_correct": pred_label == item["gold_label"] if pred_label != -1 else None
        })
    
    # 计算准确率
    metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    metrics["valid_rate"] = 1 - (metrics["invalid"] / metrics["total"]) if metrics["total"] > 0 else 0
    
    return detailed_results, metrics

# 保存结果
def save_results(results, metrics, output_file="/home2/zzl/C-CoT/baseline/selfcot/depth5_self_consistency_results.json"):
    """保存实验结果和评估指标"""
    output = {
        "config": {
            "num_samples": Config.NUM_SAMPLES,
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P,
            "num_examples": Config.NUM_EXAMPLES
        },
        "metrics": metrics,
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存至 {output_file}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"有效样本率: {metrics['valid_rate']:.4f}")
    print(f"总样本数: {metrics['total']}")
    print(f"正确样本数: {metrics['correct']}")
    print(f"无效样本数: {metrics['invalid']}")

# 主函数
def main():
    # 设置随机种子
    set_seed()
    
    # 加载模型和分词器
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(Config.QWEN_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        Config.QWEN_DIR, 
        device_map="auto", 
        trust_remote_code=True
    ).eval()
    
    # 加载数据
    print("加载数据...")
    data = load_raw_data(Config.RAW_DATA_PATH)
    
    # 生成多条CoT
    print(f"为每条问题生成 {Config.NUM_SAMPLES} 条CoT...")
    results = generate_multiple_cots(data, model, tokenizer)
    
    # 执行自一致性投票
    print("执行自一致性投票...")
    detailed_results, metrics = self_consistency_voting(results)
    
    # 保存结果
    save_results(detailed_results, metrics)

if __name__ == "__main__":
    main()
    