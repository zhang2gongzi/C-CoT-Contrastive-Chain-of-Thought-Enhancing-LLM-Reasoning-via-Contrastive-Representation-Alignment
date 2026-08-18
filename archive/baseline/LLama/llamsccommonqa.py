import torch
import pandas as pd
import numpy as np  # 新增：处理数组格式的label和text
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re
from collections import Counter
import time

def load_commonsenseqa_data(file_path, max_samples=200):
    """
    适配choices字段格式：dict{
        'label': array(['A','B','C','D','E'], dtype=object),
        'text': array(['选项1','选项2','选项3','选项4','选项5'], dtype=object)
    }
    同时处理answerKey为空的情况
    """
    df = pd.read_parquet(file_path)
    samples = []
    for idx, row in df.head(max_samples).iterrows():
        # 1. 验证choices字段是字典且包含label和text
        if not isinstance(row["choices"], dict) or "label" not in row["choices"] or "text" not in row["choices"]:
            print(f"警告：样本{idx}的choices格式异常（非标准字典），跳过")
            continue
        
        # 2. 提取label和text（数组转列表，处理numpy数组）
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        # 若为numpy数组，转为Python列表
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # 3. 验证label和text长度一致（均为5个选项）
        if len(labels) != 5 or len(texts) != 5:
            print(f"警告：样本{idx}的选项数量异常（label:{len(labels)}个, text:{len(texts)}个），跳过")
            continue
        
        # 4. 构建选项字符串（A. 选项1 B. 选项2 ...）
        options = []
        valid_labels = []  # 记录有效标签（A-E）
        for label, text in zip(labels, texts):
            label = str(label).strip().upper()  # 标准化标签（如'a'→'A'）
            text = str(text).strip()
            # 验证标签是A-E
            if label in ["A", "B", "C", "D", "E"] and text:
                options.append(f"{label}. {text}")
                valid_labels.append(label)
            else:
                print(f"警告：样本{idx}的选项（{label}:{text}）格式异常，跳过该选项")
        
        # 5. 验证有效选项（至少4个，避免过多异常）
        if len(options) < 4:
            print(f"警告：样本{idx}有效选项不足（{len(options)}个），跳过")
            continue
        
        # 6. 处理answerKey为空的情况
        correct_key = str(row["answerKey"]).strip().upper() if pd.notna(row["answerKey"]) else ""
        # 若answerKey为空或不在有效标签中，标记为待确认（不跳过，后续测评标记为"无正确答案"）
        if not correct_key or correct_key not in valid_labels:
            print(f"提示：样本{idx}的answerKey为空或无效（{correct_key}），后续测评标记为'无正确答案'")
            correct_key = None
        
        samples.append({
            "sample_idx": idx,
            "question": str(row["question"]).strip(),
            "options": " ".join(options),
            "correct_key": correct_key,  # 正确标签（A-E或None）
            "valid_labels": valid_labels  # 有效标签列表（用于后续验证）
        })
    
    print(f"\n数据加载完成！共加载{len(samples)}条有效样本（原始前{max_samples}条）")
    return samples

def extract_option_answer(generated_text, valid_labels):
    """
    从生成文本中提取选项答案（仅返回valid_labels中的标签，提高准确性）
    Args:
        generated_text: 模型生成的推理文本
        valid_labels: 该样本的有效标签列表（如['A','B','C','D','E']）
    Returns:
        str/None: 提取的有效标签或None
    """
    # 1. 优先匹配明确的答案声明（如"Answer: A"、"答案是B"）
    explicit_pattern = r"(?:Answer|答案|选择|最终答案|正确选项)\s*[:：=]\s*([ABCDE])"
    explicit_match = re.search(explicit_pattern, generated_text, re.IGNORECASE)
    if explicit_match:
        pred_label = explicit_match.group(1).upper()
        if pred_label in valid_labels:
            return pred_label
    
    # 2. 匹配推理过程中明确提到的选项（如"所以应该选C"）
    process_pattern = r"(?:选|选择|应该是|正确的是)\s*([ABCDE])"
    process_match = re.search(process_pattern, generated_text, re.IGNORECASE)
    if process_match:
        pred_label = process_match.group(1).upper()
        if pred_label in valid_labels:
            return pred_label
    
    # 3. 匹配单独出现的标签（作为最后手段，降低误判）
    single_pattern = r"\b([ABCDE])\b"
    single_matches = re.findall(single_pattern, generated_text, re.IGNORECASE)
    if single_matches:
        # 取最后一个出现的有效标签（通常是最终结论）
        for label in reversed(single_matches):
            pred_label = label.upper()
            if pred_label in valid_labels:
                return pred_label
    
    # 4. 未提取到有效标签
    return None

def generate_inference_paths(model, tokenizer, question, options, num_paths=20, temperature=0.7, max_new_tokens=150):
    """生成多条推理路径（适配常识推理的提示词）"""
    # 提示词优化：明确推理要求+格式引导，提高答案提取准确性
    prompt = f"""You are a commonsense reasoning expert. Analyze the following question and options step by step, then choose the correct answer from A to E. Finally, clearly state the answer in the format "Final Answer: X" (X is A/B/C/D/E).

Question: {question}
Options: {options}

Step-by-step analysis:"""
    
    # 生成配置（保持与原逻辑一致，优化显存占用）
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        num_return_sequences=num_paths,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=1.1  # 新增：减少重复生成，提高推理多样性
    )
    
    # 编码提示词（控制长度，避免超出模型限制）
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # 生成推理路径（禁用梯度计算，节省显存）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            use_cache=True  # 启用缓存，加速生成
        )
    
    # 解码并返回生成文本（跳过特殊token）
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True).replace(prompt, "").strip()
        for output in outputs
    ]
    return generated_texts

def self_consistency_accuracy(model, tokenizer, csqa_samples, num_paths=20):
    """计算Self-Consistency准确率（兼容answerKey为空的样本）"""
    total_valid_samples = 0  # 有正确答案的样本数
    correct_count = 0        # 预测正确的样本数
    detailed_results = []
    
    print(f"\n=== 开始CommonsenseQA Self-Consistency测评 ===")
    print(f"模型：Llama-2-7b-chat-hf | 推理路径数：{num_paths} | 总样本数：{len(csqa_samples)}")
    print(f"=============================================\n")
    
    for sample in csqa_samples:
        sample_idx = sample["sample_idx"]
        question = sample["question"]
        options = sample["options"]
        correct_key = sample["correct_key"]
        valid_labels = sample["valid_labels"]
        
        # 跳过无正确答案的样本（不参与准确率计算）
        if correct_key is None:
            detailed_results.append({
                "sample_idx": sample_idx,
                "question": question[:60] + "..." if len(question) > 60 else question,
                "correct_key": "无",
                "predicted_key": "无",
                "is_correct": "N/A",
                "path_distribution": {},
                "time_cost": 0.0
            })
            print(f"样本{sample_idx}：无正确答案，跳过准确率计算")
            continue
        
        # 记录有效样本数
        total_valid_samples += 1
        
        # 生成推理路径并提取答案
        start_time = time.time()
        generated_texts = generate_inference_paths(
            model=model,
            tokenizer=tokenizer,
            question=question,
            options=options,
            num_paths=num_paths
        )
        # 提取每条路径的答案
        path_answers = [
            extract_option_answer(text, valid_labels) 
            for text in generated_texts
        ]
        # 过滤无效答案（只保留有效标签）
        valid_path_answers = [ans for ans in path_answers if ans is not None]
        end_time = time.time()
        time_cost = round(end_time - start_time, 2)
        
        # 处理无有效答案的情况
        if not valid_path_answers:
            print(f"样本{sample_idx}：{num_paths}条路径均无有效答案，判定错误")
            detailed_results.append({
                "sample_idx": sample_idx,
                "question": question[:60] + "..." if len(question) > 60 else question,
                "correct_key": correct_key,
                "predicted_key": "无",
                "is_correct": False,
                "path_distribution": {},
                "time_cost": time_cost
            })
            continue
        
        # 多数投票：选择出现次数最多的答案
        answer_counter = Counter(valid_path_answers)
        predicted_key = answer_counter.most_common(1)[0][0]
        is_correct = (predicted_key == correct_key)
        if is_correct:
            correct_count += 1
        
        # 打印进度信息
        progress = (total_valid_samples / len([s for s in csqa_samples if s["correct_key"] is not None])) * 100
        print(f"样本{sample_idx}（{progress:.1f}%）："
              f"正确={correct_key}, 预测={predicted_key}, "
              f"结果={'√' if is_correct else '×'}, "
              f"有效路径={len(valid_path_answers)}/{num_paths}, "
              f"耗时={time_cost}s")
        
        # 记录详细结果
        detailed_results.append({
            "sample_idx": sample_idx,
            "question": question[:60] + "..." if len(question) > 60 else question,
            "correct_key": correct_key,
            "predicted_key": predicted_key,
            "is_correct": is_correct,
            "path_distribution": dict(answer_counter),
            "valid_path_ratio": f"{len(valid_path_answers)}/{num_paths}",
            "time_cost": time_cost
        })
    
    # 计算准确率（仅基于有正确答案的样本）
    if total_valid_samples == 0:
        accuracy = 0.0
        print(f"\n测评完成！无有效正确答案的样本，准确率：0.00%")
    else:
        accuracy = (correct_count / total_valid_samples) * 100
        print(f"\n=== 测评总结 ===")
        print(f"有效样本数（有正确答案）：{total_valid_samples}")
        print(f"预测正确数：{correct_count}")
        print(f"准确率：{accuracy:.2f}%")
    
    return accuracy, detailed_results

def main():
    # 1. 配置参数（根据实际环境调整）
    MODEL_PATH = "${MODEL_DIR}/Llama-2-7b-chat-hf"
    CSQA_PATH = "${PROJECT_ROOT}/database/commonsenseQA/train-00000-of-00001.parquet"
    MAX_SAMPLES = 200        # 测试前200条
    NUM_PATHS = 20           # 推理路径数（RTX 3090 24GB显存适配）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_CSV = True          # 是否保存详细结果到CSV
    CSV_SAVE_PATH = "csqa_self_consistency_results.csv"
    
    # 2. 打印配置信息
    print(f"=== 配置信息 ===")
    print(f"使用设备：{DEVICE}")
    print(f"模型路径：{MODEL_PATH}")
    print(f"数据集路径：{CSQA_PATH}")
    print(f"测试样本数：前{MAX_SAMPLES}条")
    print(f"推理路径数：{NUM_PATHS}")
    print(f"结果保存：{'是' if SAVE_CSV else '否'}（{CSV_SAVE_PATH}）")
    print(f"==============\n")
    
    # 3. 加载Tokenizer和模型（优化显存占用）
    print("开始加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",  # 右侧填充，避免生成时警告
        use_fast=False         # 禁用fast tokenizer，提高兼容性
    )
    # Llama-2默认无pad_token，手动设置为eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer加载完成！")
    
    print("\n开始加载Llama-2-7b-chat-hf模型...")
    start_load_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 用float16节省显存（约13GB）
        device_map={"": 0},         # 强制加载到第0号GPU
        load_in_8bit=False          # 若显存不足，可改为True（需安装bitsandbytes）
    )
    # 切换到评估模式，禁用Dropout
    model.eval()
    end_load_time = time.time()
    print(f"模型加载完成！耗时：{round(end_load_time - start_load_time, 2)}s")
    
    # 4. 加载CommonsenseQA数据
    print("\n开始加载CommonsenseQA数据集...")
    csqa_samples = load_commonsenseqa_data(CSQA_PATH, MAX_SAMPLES)
    if not csqa_samples:
        print("错误：未加载到任何有效样本，程序退出")
        return
    
    # 5. 执行Self-Consistency测评
    accuracy, detailed_results = self_consistency_accuracy(
        model=model,
        tokenizer=tokenizer,
        csqa_samples=csqa_samples,
        num_paths=NUM_PATHS
    )
    
    # 6. 保存详细结果到CSV
    if SAVE_CSV and detailed_results:
        results_df = pd.DataFrame(detailed_results)
        # 处理path_distribution（字典转字符串，便于CSV存储）
        results_df["path_distribution"] = results_df["path_distribution"].apply(
            lambda x: str(x) if x else "{}"
        )
        results_df.to_csv(CSV_SAVE_PATH, index=False, encoding="utf-8-sig")
        print(f"\n详细结果已保存到：{CSV_SAVE_PATH}")

if __name__ == "__main__":
    # 解决编码问题和异常捕获
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    try:
        main()
    except Exception as e:
        print(f"\n运行出错：{str(e)}")
        # 打印详细错误栈（便于排查）
        import traceback
        traceback.print_exc()