import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re
from collections import Counter
import time

def load_gsm8k_data(file_path, max_samples=1):
    """
    加载GSM8K测试数据集并提取前N条样本
    Args:
        file_path: GSM8K parquet文件路径
        max_samples: 最大样本数量，默认200
    Returns:
        list: 包含问题和答案的字典列表
    """
    # 加载parquet文件
    df = pd.read_parquet(file_path)
    # 提取前max_samples条数据
    samples = []
    for idx, row in df.head(max_samples).iterrows():
        samples.append({
            "question": row["question"],
            "answer": row["answer"]  # 原始答案格式如"#### 42"
        })
    print(f"成功加载GSM8K测试数据，共{len(samples)}条样本")
    return samples

def extract_numeric_answer(answer_str):
    """
    从答案字符串中提取数字（处理GSM8K的"#### 数字"格式）
    Args:
        answer_str: 模型生成的答案字符串或数据集原始答案
    Returns:
        float/int: 提取的数字，失败返回None
    """
    # 处理数据集原始答案（如"#### 42"）
    if "####" in answer_str:
        match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_str)
    else:
        # 处理模型生成答案（可能包含计算过程，提取最后出现的数字）
        matches = re.findall(r"[-+]?\d*\.?\d+", answer_str)
        if matches:
            # 优先选择最后一个数字（通常是最终答案）
            match = re.match(r"([-+]?\d*\.?\d+)", matches[-1])
        else:
            match = None
    
    if match:
        num_str = match.group(1)
        # 转换为int或float
        if "." in num_str:
            return float(num_str)
        else:
            return int(num_str)
    return None

def generate_inference_paths(model, tokenizer, question, num_paths=40, temperature=0.7, max_new_tokens=200):
    """
    为单个问题生成多条推理路径（Self-Consistency核心步骤）
    Args:
        model: 加载的Llama-2模型
        tokenizer: 对应的tokenizer
        question: 问题文本
        num_paths: 生成的推理路径数量（默认40，参考Self-C原论文设置）
        temperature: 采样温度（控制多样性，默认0.7）
        max_new_tokens: 最大生成token数（默认200，足够覆盖GSM8K计算过程）
    Returns:
        list: 每条路径提取的数字答案列表
    """
    # 构建提示词（采用思维链提示格式）
    prompt = f"""Below is a math word problem. Please solve it step by step, and finally give the answer in the format "#### [your answer]".

Problem: {question}

Solution:"""
    
    # 配置生成参数（启用采样以获取多样化路径）
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        num_return_sequences=num_paths,  # 一次生成多条路径
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True  # 关键：启用采样而非贪婪解码
    )
    
    # 编码提示词
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 根据Llama-2输入限制调整
    ).to(model.device)
    
    # 生成推理路径
    with torch.no_grad():  # 禁用梯度计算以节省内存
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # 解码并提取答案
    answer_list = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        # 提取生成文本中prompt之后的部分（即推理过程）
        inference_text = generated_text[len(prompt):].strip()
        # 提取数字答案
        numeric_answer = extract_numeric_answer(inference_text)
        if numeric_answer is not None:
            answer_list.append(numeric_answer)
    
    return answer_list

def self_consistency_accuracy(model, tokenizer, gsm8k_samples, num_paths=40):
    """
    计算Self-Consistency方法的准确率
    Args:
        model: 加载的Llama-2模型
        tokenizer: 对应的tokenizer
        gsm8k_samples: GSM8K样本列表（含问题和正确答案）
        num_paths: 每条样本生成的推理路径数量
    Returns:
        float: 准确率（百分比）
        list: 详细结果（问题、正确答案、预测答案、是否正确）
    """
    correct_count = 0
    detailed_results = []
    total_samples = len(gsm8k_samples)
    
    print(f"\n开始Self-Consistency测评（每条样本生成{num_paths}条推理路径）...")
    print(f"总样本数：{total_samples}，模型：Llama-2-7b-chat-hf\n")
    
    for idx, sample in enumerate(gsm8k_samples, 1):
        question = sample["question"]
        correct_answer = extract_numeric_answer(sample["answer"])
        
        if correct_answer is None:
            print(f"警告：样本{idx}无法提取正确答案，跳过")
            continue
        
        # 生成多条推理路径并提取答案
        start_time = time.time()
        path_answers = generate_inference_paths(model, tokenizer, question, num_paths)
        end_time = time.time()
        
        # 处理无有效答案的情况
        if not path_answers:
            print(f"样本{idx}：无有效推理路径答案，判定错误")
            detailed_results.append({
                "sample_idx": idx,
                "question": question[:50] + "..." if len(question) > 50 else question,
                "correct_answer": correct_answer,
                "predicted_answer": None,
                "is_correct": False,
                "time_cost": round(end_time - start_time, 2)
            })
            continue
        
        # 投票选择最一致的答案（多数投票）
        answer_counter = Counter(path_answers)
        predicted_answer = answer_counter.most_common(1)[0][0]
        
        # 判断是否正确（允许浮点数微小误差，如2.0和2视为正确）
        is_correct = False
        if isinstance(correct_answer, (int, float)) and isinstance(predicted_answer, (int, float)):
            if abs(correct_answer - predicted_answer) < 1e-6:
                correct_count += 1
                is_correct = True
        
        # 记录详细结果
        detailed_results.append({
            "sample_idx": idx,
            "question": question[:50] + "..." if len(question) > 50 else question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "path_answer_distribution": dict(answer_counter),  # 各答案出现次数
            "is_correct": is_correct,
            "time_cost": round(end_time - start_time, 2)
        })
        
        # 打印进度
        progress = (idx / total_samples) * 100
        print(f"样本{idx}/{total_samples}（{progress:.1f}%）："
              f"正确答案={correct_answer}, 预测答案={predicted_answer}, "
              f"是否正确={'√' if is_correct else '×'}, 耗时={end_time - start_time:.2f}s")
    
    # 计算准确率
    accuracy = (correct_count / total_samples) * 100
    print(f"\n测评完成！总样本数：{total_samples}，正确数：{correct_count}，准确率：{accuracy:.2f}%")
    
    return accuracy, detailed_results

def main():
    # 1. 配置参数
    MODEL_PATH = "/home2/zzl/model/Llama-2-7b-chat-hf"  # 模型路径
    GSM8K_PATH = "/home2/zzl/C-CoT/database/gsm8k/test-00000-of-00001.parquet"  # 数据集路径
    MAX_SAMPLES = 200  # 测试前200条
    NUM_PATHS = 40  # Self-Consistency推理路径数量（原论文推荐40）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 优先使用GPU
    
    print(f"使用设备：{DEVICE}")
    print(f"模型路径：{MODEL_PATH}")
    print(f"数据集路径：{GSM8K_PATH}")
    
    # 2. 加载模型和Tokenizer（Llama-2专用配置）
    print("\n开始加载模型和Tokenizer...")
    start_load_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"  # 避免生成时警告
    )
    tokenizer.pad_token = tokenizer.eos_token  # Llama-2默认无pad_token，需手动设置
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 用float16节省GPU内存（7B模型约需13-15GB）
        device_map="auto"  # 自动分配设备（优先GPU）
    )
    model.eval()  # 切换到评估模式，禁用Dropout
    
    end_load_time = time.time()
    print(f"模型加载完成，耗时：{end_load_time - start_load_time:.2f}s")
    
    # 3. 加载GSM8K数据
    print("\n开始加载GSM8K数据...")
    gsm8k_samples = load_gsm8k_data(GSM8K_PATH, MAX_SAMPLES)
    
    # 4. 执行Self-Consistency测评
    accuracy, detailed_results = self_consistency_accuracy(
        model=model,
        tokenizer=tokenizer,
        gsm8k_samples=gsm8k_samples,
        num_paths=NUM_PATHS
    )
    
    # 5. 保存详细结果到CSV（便于后续分析）
    results_df = pd.DataFrame(detailed_results)
    results_df.drop(columns=["path_answer_distribution"], inplace=True)  # 分布信息不适合CSV，可按需保留
    save_path = f"/home2/zzl/C-CoT/baseline/LLama/self_consistency_gsm8k_results_{MAX_SAMPLES}samples.csv"
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存到：{save_path}")

if __name__ == "__main__":
    # 解决部分环境中的编码问题
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # 捕获运行时错误
    try:
        main()
    except Exception as e:
        print(f"\n运行出错：{str(e)}")
        raise  # 打印详细错误栈