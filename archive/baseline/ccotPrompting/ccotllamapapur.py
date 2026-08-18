import os
import re
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


def load_pararule_data(file_path, num_samples=200):
    try:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    "id": item.get("id", ""),
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "label": item.get("label", 0)
                })
                if len(data) >= num_samples:
                    break
        df = pd.DataFrame(data)
        print(f"成功加载数据，共{len(df)}条样本（label=1: {len(df[df['label']==1])}, label=0: {len(df[df['label']==0])}）")
        return df
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        raise


def build_contrastive_prompt(context, question):
    """反驳式推理Prompt：先找否定证据，再确认条件"""
    contrastive_examples = """
You are a strict logical validator. Follow the "Refutation-First" 6-step process to judge the question:
1. Target Object: Subject of the question (e.g., "The cat").
2. Direct Attributes: List ONLY attributes EXPLICITLY stated in the context (no assumptions).
3. Derived Attributes: Use rules to derive new attributes → MUST cite the exact rule (e.g., "Rule: If A→B → derived B").
4. Critical Rule: Find the rule where Y = the claim in the question (e.g., "If X→[claim]").
5. Refutation Check (FIRST): 
   - Do any direct/derived attributes PROVE X (rule conditions) is NOT met? If YES → Conclusion: No.
6. Confirmation Check (SECOND): 
   - Are ALL conditions in X EXPLICITLY met (by direct/derived attributes)? If YES → Conclusion: Yes.
   - If ANY condition is unknown (no direct/derived evidence) → Conclusion: No.

--- Example 1 (label=0: Refutation succeeded) ---
Context: 
- The cat is black. The cat is not fat.
- Rule: If black AND fat → cute.
Question: The cat is cute.

Reasoning:
1. Target Object: The cat.
2. Direct Attributes: black, not fat.
3. Derived Attributes: No rules to derive new attributes.
4. Critical Rule: If black AND fat → cute (Y="cute").
5. Refutation Check: Direct attribute "not fat" PROVES "fat" (condition) is NOT met → Conclusion: No.

--- Example 2 (label=1: Confirmation succeeded, no refutation) ---
Context: 
- The dog is furry. The dog is not big.
- Rules: If furry→lovely; If lovely AND not big→small.
Question: The dog is small.

Reasoning:
1. Target Object: The dog.
2. Direct Attributes: furry, not big.
3. Derived Attributes: 
   - Rule: If furry→lovely → derived "lovely".
4. Critical Rule: If lovely AND not big→small (Y="small").
5. Refutation Check: No attributes prove "lovely" or "not big" are unmet.
6. Confirmation Check: All conditions ("lovely", "not big") are explicitly met → Conclusion: Yes.

--- Example 3 (label=0: Condition unknown → refute) ---
Context: 
- The bird is sleepy.
- Rule: If sleepy AND rough→big.
Question: The bird is big.

Reasoning:
1. Target Object: The bird.
2. Direct Attributes: sleepy.
3. Derived Attributes: No rules to derive "rough".
4. Critical Rule: If sleepy AND rough→big (Y="big").
5. Refutation Check: No attributes prove "sleepy" is unmet.
6. Confirmation Check: Condition "rough" is unknown (no evidence) → Conclusion: No.

--- Target Problem ---
Context: {context}
Question: {question}

Reasoning:
"""
    return contrastive_examples.format(context=context, question=question)


def load_llama2_model(model_path):
    try:
        print(f"开始加载模型：{model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.chat_template = "<s>[INST] {user_message} [/INST]"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=False
        )
        model.eval()
        print("模型加载完成")
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        raise


def extract_answer(generated_text):
    """严格提取Conclusion中的Yes/No"""
    match = re.search(r"Conclusion:\s*(Yes|No)", generated_text, re.IGNORECASE)
    return match.group(1).strip() if match else "Unknown"


def run_inference(df, model, tokenizer, OUTPUT_PATH):  # 修正：参数名改为OUTPUT_PATH（与调用一致）
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="推理中"):
        sample_id = row["id"]
        context = row["context"]
        question = row["question"]
        ground_truth = row["label"]

        # 1. 构建Prompt
        prompt = build_contrastive_prompt(context, question)
        formatted_prompt_str = tokenizer.chat_template.format(user_message=prompt)
        
        # 2. 模型输入
        inputs = tokenizer(
            formatted_prompt_str,
            return_tensors="pt",
            padding="max_length",
            max_length=2048,
            truncation=True
        ).to(model.device)

        # 3. 模型推理
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=350,
                    temperature=0.1,
                    top_p=0.85,
                    repetition_penalty=1.3,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id
                )
        except Exception as e:
            print(f"推理失败（{sample_id}）：{e}")
            model_answer = "Unknown"
            model_label = -1
            model_reasoning = "推理错误"
        else:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            model_reasoning = generated_text.split(formatted_prompt_str)[-1].strip() if formatted_prompt_str in generated_text else generated_text
            model_answer = extract_answer(model_reasoning)
            model_label = 1 if model_answer.lower() == "yes" else 0 if model_answer.lower() == "no" else -1

        # 4. 保存结果
        results.append({
            "id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "model_label": model_label,
            "model_reasoning": model_reasoning,
            "did_refutation": "Refutation Check" in model_reasoning,
            "did_confirmation": "Confirmation Check" in model_reasoning,
            "is_correct": (model_label == ground_truth) if model_label != -1 else False
        })

    # 5. 结果保存与分析
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)  # 修正：使用OUTPUT_PATH
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")  # 修正：使用OUTPUT_PATH
    print(f"\n结果保存至：{OUTPUT_PATH}")
    
    # 核心指标计算
    total = len(results_df)
    correct = results_df["is_correct"].sum()
    acc = correct / total if total > 0 else 0
    print(f"\n=== 核心指标 ===")
    print(f"准确率：{acc:.2%}（{correct}/{total}）")
    print(f"执行反驳检查比例：{results_df['did_refutation'].mean():.2%}")
    print(f"执行确认检查比例：{results_df['did_confirmation'].mean():.2%}")
    
    # 错误分布分析
    wrong = results_df[~results_df["is_correct"] & (results_df["model_label"] != -1)]
    if not wrong.empty:
        false_yes = len(wrong[wrong["ground_truth"] == 0])
        false_no = len(wrong[wrong["ground_truth"] == 1])
        print(f"\n=== 错误分布 ===")
        print(f"误判Yes（实际0）：{false_yes}条 | 误判No（实际1）：{false_no}条")
        
        # 输出典型错误案例
        if false_yes > 0:
            bad_case = wrong[wrong["ground_truth"] == 0].iloc[0]
            print(f"\n=== 误判Yes案例（ID：{bad_case['id']}）===")
            print(f"问题：{bad_case['question']}")
            print(f"是否执行反驳：{'是' if bad_case['did_refutation'] else '否'}")
            print(f"推理链片段：{bad_case['model_reasoning'][:300]}...")
    
    return results_df


if __name__ == "__main__":
    # 路径配置（变量名统一为大写，避免大小写错误）
    DATA_PATH = "${PROJECT_ROOT}/../ChatLogic/PARARULE-Plus/Depth2/PARARULE_Plus_Depth2_shuffled_train_huggingface.jsonl"
    MODEL_PATH = "${MODEL_DIR}/Llama-2-7b-chat-hf"
    OUTPUT_PATH = "${PROJECT_ROOT}/results/llama2_7b_pararule_v6.csv"  # 定义为OUTPUT_PATH
    NUM_SAMPLES = 10  # 从输出看，当前测试用10条样本，可按需改回200
    
    # 执行流程（调用时用OUTPUT_PATH，与定义一致）
    df = load_pararule_data(DATA_PATH, NUM_SAMPLES)
    model, tokenizer = load_llama2_model(MODEL_PATH)
    results_df = run_inference(df, model, tokenizer, OUTPUT_PATH)  # 修正：参数用OUTPUT_PATH