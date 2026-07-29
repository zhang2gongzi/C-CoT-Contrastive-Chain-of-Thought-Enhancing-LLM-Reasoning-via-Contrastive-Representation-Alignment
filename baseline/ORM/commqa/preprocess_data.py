#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D-ProtoCoT / ORM 数据预处理脚本 (适配 CommonsenseQA 及通用格式)
功能：Parquet读取 → LLM路径生成 → 答案提取与清洗 → 格式转换 → Train/Val/Test划分
输出格式：{"question": "...", "ground_truth": "...", "paths": [{"reasoning": "...", "predicted_answer": "..."}]}
"""

import os
import json
import re
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 1. 答案提取与归一化工具
# ==========================================
def extract_answer(text):
    """从CoT文本中可靠提取最终答案，优先适配选择题字母"""
    # 优先匹配选择题答案字母 (A-E)
    letters = re.findall(r'\b[A-E]\b', text)
    if letters:
        return letters[-1].upper()

    # 原有正则逻辑保留（适配数学/策略问答）
    patterns = [
        r'[Aa]nswer\s*[:：]\s*(.*)',
        r'[Tt]he\s+answer\s+is\s+([^\n.]+)',
        r'####\s*(.*)',
        r'[Ff]inal\s+[Aa]nswer\s*[:：]\s*(.*)'
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if ans: return re.sub(r'[^\w\s.%-]', '', ans).strip()

    # 兜底：取最后一个数字或单词
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if nums: return nums[-1]
    return text.strip().split()[-1] if text.strip() else ""

# ==========================================
# 2. LLM 路径生成模块
# ==========================================
def setup_pipeline(model_name, device_map="auto"):
    logger.info(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    return gen_pipe, tokenizer

def generate_k_paths(prompt_text, gen_pipe, tokenizer, k=10, temperature=0.7, top_p=0.9, max_new_tokens=1024):
    """为单个问题生成K条独立CoT路径"""
    paths = []
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for i in range(k):
        try:
            outputs = gen_pipe(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )
            reasoning = outputs[0]["generated_text"].strip()
            pred_ans = extract_answer(reasoning)
            paths.append({"reasoning": reasoning, "predicted_answer": pred_ans})
        except Exception as e:
            logger.warning(f"Failed generation #{i+1}: {e}")
            paths.append({"reasoning": "Generation failed.", "predicted_answer": ""})
    return paths

# ==========================================
# 3. 主流程：读取 → 生成 → 划分 → 保存
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing for D-ProtoCoT/ORM")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to input .parquet file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--k_paths", type=int, default=10, help="Number of CoT paths per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--split_ratio", type=str, default="0.8,0.1,0.1", help="train,val,test ratio")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to resume from checkpoint JSON")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N questions")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据
    logger.info(f"Reading parquet: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    logger.info(f"Total questions: {len(df)}")

    # 2. 初始化模型
    gen_pipe, tokenizer = setup_pipeline(args.model_name)

    # 3. 生成路径 (支持断点续传)
    processed_data = []
    start_idx = 0
    ckpt_path = os.path.join(args.output_dir, "ckpt_generated.json")

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt_path = args.resume_ckpt

    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        start_idx = len(processed_data)
        logger.info(f"Resuming from checkpoint. Already processed: {start_idx}/{len(df)}")
    else:
        logger.info("Starting fresh generation...")

    # 智能识别列名
    has_choices = "choices" in df.columns
    answer_col = "answerKey" if "answerKey" in df.columns else ("answer" if "answer" in df.columns else "ground_truth")

    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        q_text = str(row["question"]).strip()
        gt_text = str(row[answer_col]).strip()

        # 构建 Prompt
        if has_choices and isinstance(row.get("choices"), dict):
            # 适配 CommonsenseQA 格式
            labels = list(row["choices"].get("label", []))
            texts = list(row["choices"].get("text", []))
            opts = ", ".join([f"{l}. {t}" for l, t in zip(labels, texts)])
            full_prompt = f"Question: {q_text}\nChoices: {opts}\nReason step by step and end with the answer letter."
        else:
            # 通用数学/策略问答格式
            full_prompt = f"Question: {q_text}\nReason step by step and end with the final answer."

        paths = generate_k_paths(
            full_prompt, gen_pipe, tokenizer,
            k=args.k_paths, temperature=args.temperature,
            top_p=args.top_p, max_new_tokens=args.max_new_tokens
        )

        processed_data.append({
            "question": q_text,
            "ground_truth": gt_text,
            "paths": paths
        })

        if (idx + 1) % args.save_every == 0:
            with open(ckpt_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Checkpoint saved at idx {idx+1}")

        logger.info(f"Progress: {idx+1}/{len(df)} | Q: {q_text[:50]}... | GT: {gt_text}")

    # 保存完整生成结果
    full_path = os.path.join(args.output_dir, "all_generated.json")
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ All paths generated and saved to {full_path}")

    # 4. 划分 Train/Val/Test
    ratios = [float(x) for x in args.split_ratio.split(',')]
    assert abs(sum(ratios) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    train_ratio, val_ratio, test_ratio = ratios
    train_val, test_data = train_test_split(processed_data, test_size=test_ratio, random_state=args.seed)
    val_size = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(train_val, test_size=val_size, random_state=args.seed)

    logger.info(f"📊 Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = os.path.join(args.output_dir, f"{name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"📄 Saved {name}.json ({len(data)} samples)")

    logger.info("🎉 Preprocessing completed successfully.")

if __name__ == "__main__":
    main()