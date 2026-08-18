import torch
import json
import re
from collections import Counter
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def load_pararule_data(file_path, max_samples=200):
    """简化版：加载PARARULE-Plus数据（仅保留核心字段）"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                # 只提取必需字段，跳过异常样本
                if "context" in data and "question" in data and "label" in data:
                    samples.append({
                        "idx": idx,
                        "context": data["context"].strip(),
                        "question": data["question"].strip(),
                        "label": int(data["label"])  # 1=正确，0=错误
                    })
            except:
                continue
    print(f"加载完成：{len(samples)}/{max_samples}条有效样本")
    return samples

def extract_conclusion(generated_text):
    """简化版：提取True/False结论"""
    text = generated_text.lower()
    # 优先匹配明确结论
    if re.search(r"conclusion[:：=]\s*(true|false)", text):
        return 1 if "true" in text else 0
    # 匹配简单判断
    if any(w in text for w in ["true", "正确", "成立"]):
        return 1
    if any(w in text for w in ["false", "错误", "不成立"]):
        return 0
    return None

def generate_paths(model, tokenizer, context, question, num_paths=5):
    """简化版：生成推理路径"""
    # 简洁提示词，聚焦逻辑推理
    prompt = f"""Based on the context (facts + rules), judge if the question is correct.
Context: {context}
Question: {question}
Analyze step by step, then output "Conclusion: True/False" at last.
Analysis:"""
    
    # 简化生成配置
    gen_config = GenerationConfig(
        temperature=0.7,
        num_return_sequences=num_paths,
        max_new_tokens=200,  # 缩短生成长度
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    # 生成推理
    with torch.no_grad():
        outputs = model.generate(** inputs, generation_config=gen_config)
    
    # 解码结果
    return [tokenizer.decode(o, skip_special_tokens=True).split("Analysis:")[-1].strip() for o in outputs]

def calc_accuracy(model, tokenizer, samples, num_paths=5):
    """简化版：计算Self-C准确率"""
    correct = 0
    total = len(samples)
    
    print(f"\n开始测评：{total}条样本，{num_paths}条路径/样本")
    for idx, sample in enumerate(samples):
        # 生成路径并提取结论
        paths = generate_paths(model, tokenizer, sample["context"], sample["question"], num_paths)
        conclusions = [extract_conclusion(p) for p in paths]
        valid_conclu = [c for c in conclusions if c is not None]
        
        # 投票预测
        if valid_conclu:
            pred = Counter(valid_conclu).most_common(1)[0][0]
            is_correct = (pred == sample["label"])
            correct += 1 if is_correct else 0
        else:
            is_correct = False
        
        # 打印进度
        progress = (idx+1)/total*100
        print(f"样本{idx+1}/{total}（{progress:.1f}%）：真实={sample['label']}, 预测={pred if valid_conclu else '无'}, {'√' if is_correct else '×'}")
    
    # 计算准确率
    accuracy = (correct/total)*100 if total >0 else 0
    print(f"\n测评结束：正确{correct}条，准确率{accuracy:.2f}%")
    return accuracy

def main():
    # 核心配置（按需修改）
    MODEL_PATH = "${MODEL_DIR}/Llama-2-7b-chat-hf"
    DATA_PATH = "${PROJECT_ROOT}/../ChatLogic/PARARULE-Plus/Depth5/PARARULE_Plus_Depth5_shuffled_train_huggingface.jsonl"
    MAX_SAMPLES = 200  # 仅前200条
    NUM_PATHS = 5      # 仅5条推理路径
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载Tokenizer
    print("加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"":0}
    ).eval()
    
    # 加载数据
    print("加载数据...")
    samples = load_pararule_data(DATA_PATH, MAX_SAMPLES)
    
    # 执行测评
    calc_accuracy(model, tokenizer, samples, NUM_PATHS)

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    try:
        main()
    except Exception as e:
        print(f"运行错误：{str(e)}")
        import traceback
        traceback.print_exc()