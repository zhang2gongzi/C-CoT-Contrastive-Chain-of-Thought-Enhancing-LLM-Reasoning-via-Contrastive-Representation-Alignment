import os
import re
import json
import torch
import pandas as pd
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ======================
# 配置参数
# ======================
class Config:
    LLAMA_DIR = "/home2/zzl/model/Llama-2-7b-chat-hf"
    PARARULE_BASE_PATH = "/home2/zzl/ChatLogic/PARARULE-Plus"  # 基础路径，包含不同深度的子目录
    OUTPUT_DIR = "/home2/zzl/C-CoT/baseline/LLama/llama_pararule_results"
    
    # 逻辑推理任务超参数
    MAX_NEW_TOKENS = 500    # 增加 tokens 以容纳推理步骤
    TEMPERATURE = 0.4       # 适度提高随机性，使不同难度表现出差异
    TOP_P = 0.9
    BATCH_SIZE = 4          # 减小批次大小，避免内存问题
    SAMPLE_SIZE = 500       # 每个深度的测试样本量
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEPTHS = [2, 3, 4, 5]   # 要评估的不同推理深度

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# ======================
# 数据加载函数
# ======================
def load_pararule_data_by_depth(depth: int) -> Tuple[List[Dict], str]:
    """加载特定推理深度的PARARULE-Plus数据集（JSONL格式）"""
    file_path = os.path.join(
        cfg.PARARULE_BASE_PATH, 
        f"Depth{depth}", 
        f"PARARULE_Plus_Depth{depth}_shuffled_dev_huggingface.jsonl"
    )
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                # 验证必要字段
                required_fields = ["id", "context", "question", "label", "meta"]
                if not all(field in sample for field in required_fields):
                    print(f"警告: 深度{depth}，行{line_num}缺少必要字段，已跳过")
                    continue
                # 添加深度信息
                sample["depth"] = depth
                data.append(sample)
            except json.JSONDecodeError:
                print(f"警告: 深度{depth}，行{line_num}格式错误，已跳过")
    
    return data, file_path


# ======================
# 答案提取与判断函数
# ======================
def extract_boolean_answer(text: str) -> int:
    """
    从生成文本中提取布尔答案（1表示正确，0表示错误）
    适配常见表述：Yes/No、True/False、正确/错误等
    """
    text_lower = text.lower()
    
    # 匹配肯定表述
    positive_patterns = [
        r"yes", r"true", r"correct", r"right",
        r"is true", r"is correct", r"does hold",
        r"answer: 1", r"label: 1"
    ]
    for pattern in positive_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    # 匹配否定表述
    negative_patterns = [
        r"no", r"false", r"incorrect", r"wrong",
        r"is false", r"is incorrect", r"does not hold",
        r"answer: 0", r"label: 0"
    ]
    for pattern in negative_patterns:
        if re.search(pattern, text_lower):
            return 0
    
    # 无法提取时，根据最后出现的肯定/否定词判断
    words = re.findall(r"\b\w+\b", text_lower)
    positive_words = {"yes", "true", "correct", "right"}
    negative_words = {"no", "false", "incorrect", "wrong"}
    
    for word in reversed(words):
        if word in positive_words:
            return 1
        if word in negative_words:
            return 0
    
    # 完全无法判断时返回-1（表示提取失败）
    return -1

def calculate_accuracy_by_depth(predictions: List[Dict]) -> Dict:
    """按推理深度计算准确率及其他评估指标"""
    # 按深度分组
    depth_groups = {}
    for pred in predictions:
        depth = pred["depth"]
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(pred)
    
    # 计算每个深度的指标
    metrics = {}
    overall = {"total": 0, "correct": 0, "failed": 0}
    
    for depth, items in depth_groups.items():
        total = len(items)
        correct = sum(1 for item in items if item["prediction"] == item["label"] and item["prediction"] != -1)
        failed = sum(1 for item in items if item["prediction"] == -1)
        
        metrics[depth] = {
            "total": total,
            "correct": correct,
            "failed": failed,
            "accuracy": correct / total if total > 0 else 0.0,
            "success_rate": (total - failed) / total if total > 0 else 0.0
        }
        
        # 更新整体指标
        overall["total"] += total
        overall["correct"] += correct
        overall["failed"] += failed
    
    # 计算整体指标
    metrics["overall"] = {
        "total": overall["total"],
        "correct": overall["correct"],
        "failed": overall["failed"],
        "accuracy": overall["correct"] / overall["total"] if overall["total"] > 0 else 0.0,
        "success_rate": (overall["total"] - overall["failed"]) / overall["total"] if overall["total"] > 0 else 0.0
    }
    
    return metrics


# ======================
# 模型加载与推理函数
# ======================
def load_llama_model():
    """加载LLaMA模型和分词器"""
    print(f"加载LLaMA模型: {cfg.LLAMA_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.LLAMA_DIR,
        trust_remote_code=True,
        padding_side="left"  # 修复之前的右填充警告
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.LLAMA_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def count_rules(context: str) -> int:
    """统计上下文中的规则数量，帮助模型理解推理复杂度"""
    # 简单规则检测模式
    rule_patterns = [
        r"if .+ then",
        r"all .+ are",
        r".+ people are",
        r".+ is .+"
    ]
    
    count = 0
    for pattern in rule_patterns:
        count += len(re.findall(pattern, context.lower()))
    return max(1, count)  # 至少1条规则

def batch_inference(
    contexts: List[str],
    questions: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM
) -> List[str]:
    """批量推理函数，处理上下文+问题格式，增强对推理步骤的敏感性"""
    # 构建提示词（引导模型关注推理步骤）
    prompts = []
    for ctx, q in zip(contexts, questions):
        rule_count = count_rules(ctx)
        prompts.append(
            f"Context: {ctx}\n"
            f"Question: {q}\n"
            f"This question requires logical reasoning based on {rule_count} rules in the context.\n"
            f"Please think step by step, considering each relevant rule, "
            f"then determine if the question is correct.\n"
            f"After your reasoning, clearly state 'Yes' if correct or 'No' if incorrect."
        )
    
    # 分词处理
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # 增加最大长度以容纳更长的上下文和推理步骤
        padding=True
    ).to(cfg.DEVICE)
    
    # 生成答案
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码结果
    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )


# ======================
# 主评估流程
# ======================
def evaluate():
    # 1. 加载模型
    tokenizer, model = load_llama_model()
    
    # 2. 加载并评估所有深度的数据集
    all_predictions = []
    all_results = []
    data_paths = {}
    
    for depth in cfg.DEPTHS:
        print(f"\n加载深度为{depth}的数据集...")
        data, data_path = load_pararule_data_by_depth(depth)
        data_paths[depth] = data_path
        print(f"深度{depth}原始数据量: {len(data)}")
        
        # 采样数据
        if cfg.SAMPLE_SIZE and cfg.SAMPLE_SIZE < len(data):
            import random
            random.seed(42 + depth)  # 为不同深度设置不同种子，保证可复现性同时增加差异
            data = random.sample(data, cfg.SAMPLE_SIZE)
        print(f"深度{depth}评估数据量: {len(data)}")
        
        # 批量推理
        total_batches = (len(data) + cfg.BATCH_SIZE - 1) // cfg.BATCH_SIZE
        
        for batch_idx in tqdm(range(total_batches), desc=f"深度{depth}评估进度"):
            # 提取批次数据
            start = batch_idx * cfg.BATCH_SIZE
            end = min((batch_idx + 1) * cfg.BATCH_SIZE, len(data))
            batch = data[start:end]
            
            contexts = [item["context"] for item in batch]
            questions = [item["question"] for item in batch]
            labels = [item["label"] for item in batch]
            ids = [item["id"] for item in batch]
            depths = [item["depth"] for item in batch]
            
            # 生成答案
            generated_texts = batch_inference(contexts, questions, tokenizer, model)
            
            # 解析答案并保存结果
            for text, label, ctx, q, sample_id, d in zip(
                generated_texts, labels, contexts, questions, ids, depths
            ):
                pred = extract_boolean_answer(text)
                all_predictions.append({
                    "id": sample_id,
                    "depth": d,
                    "label": label,
                    "prediction": pred
                })
                all_results.append({
                    "id": sample_id,
                    "depth": d,
                    "context": ctx,
                    "question": q,
                    "label": label,
                    "prediction": pred,
                    "generated_text": text,
                    "is_correct": pred == label if pred != -1 else None
                })
    
    # 3. 计算按深度分组的指标
    metrics = calculate_accuracy_by_depth(all_predictions)
    
    # 4. 打印评估结果
    print("\n===== 按推理深度的评估结果 =====")
    for depth in cfg.DEPTHS:
        if depth in metrics:
            print(f"\n深度 {depth}:")
            print(f"  总样本数: {metrics[depth]['total']}")
            print(f"  正确数: {metrics[depth]['correct']}")
            print(f"  提取失败数: {metrics[depth]['failed']}")
            print(f"  准确率: {metrics[depth]['accuracy']:.4f} ({metrics[depth]['accuracy']*100:.2f}%)")
    
    print("\n===== 整体评估结果 =====")
    print(f"总样本数: {metrics['overall']['total']}")
    print(f"正确数: {metrics['overall']['correct']}")
    print(f"提取失败数: {metrics['overall']['failed']}")
    print(f"准确率: {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
    print(f"成功提取率: {metrics['overall']['success_rate']:.4f}")
    
    # 5. 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_excel(
        os.path.join(cfg.OUTPUT_DIR, "pararule_results_by_depth.xlsx"),
        index=False
    )
    print(f"\n详细结果已保存至: {cfg.OUTPUT_DIR}")
    
    # 保存指标 summary
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics_by_depth.txt"), "w") as f:
        f.write(f"评估时间: {pd.Timestamp.now()}\n")
        f.write(f"模型路径: {cfg.LLAMA_DIR}\n")
        for depth in cfg.DEPTHS:
            f.write(f"深度{depth}数据集路径: {data_paths.get(depth, '未知')}\n")
        f.write("\n===== 按深度指标 =====\n")
        for depth in cfg.DEPTHS:
            if depth in metrics:
                f.write(f"深度 {depth}:\n")
                f.write(f"  总样本数: {metrics[depth]['total']}\n")
                f.write(f"  正确数: {metrics[depth]['correct']}\n")
                f.write(f"  提取失败数: {metrics[depth]['failed']}\n")
                f.write(f"  准确率: {metrics[depth]['accuracy']:.4f}\n")
        f.write("\n===== 整体指标 =====\n")
        f.write(f"总样本数: {metrics['overall']['total']}\n")
        f.write(f"正确数: {metrics['overall']['correct']}\n")
        f.write(f"提取失败数: {metrics['overall']['failed']}\n")
        f.write(f"准确率: {metrics['overall']['accuracy']:.4f}\n")
        f.write(f"成功提取率: {metrics['overall']['success_rate']:.4f}\n")


if __name__ == "__main__":
    evaluate()
    