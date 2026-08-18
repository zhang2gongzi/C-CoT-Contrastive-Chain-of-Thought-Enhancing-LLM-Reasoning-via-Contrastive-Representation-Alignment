import json
import re
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ===== 配置 =====
MODEL_PATH = "${MODEL_DIR}/../model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
VAL_PATH = "${PROJECT_ROOT}/../ChatLogic/PARARULE-Plus/Depth2/PARARULE_Plus_Depth2_shuffled_dev_huggingface.jsonl"
SAVE_PATH = "${PROJECT_ROOT}/test_C-CoT/cot_generated_1000_flat_labeled_depth2.jsonl"

NUM_EXAMPLES = 1000      
N_SAMPLES = 4           # 每题生成 N 条 CoT
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 加载模型 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()

# # ===== 读取前 NUM_EXAMPLES 条验证集 =====
# dataset = []
# with open(VAL_PATH, "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         if i >= NUM_EXAMPLES:
#             break
#         dataset.append(json.loads(line))
        # ===== 读取数据（修改部分） =====
dataset = []
with open(VAL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # 若 NUM_EXAMPLES 为 None，则读取全部；否则读取前 NUM_EXAMPLES 条
        if NUM_EXAMPLES is not None and i >= NUM_EXAMPLES:
            break
        dataset.append(json.loads(line))

# ===== 从 CoT 文本解析模型的最终回答（Yes/No → label） =====
def extract_label_from_cot(cot_text):
    text = cot_text.strip().lower()

    match = re.search(r"cot_text[:：]?\s*(yes|no)", text)
    if match:
        return 1 if match.group(1) == "yes" else 0

    if "yes" in text and "no" not in text:
        return 1
    elif "no" in text and "yes" not in text:
        return 0

    return None

# ===== 判断是否正确 =====
def judge_correctness(pred_label, gold_label):
    if pred_label is None:
        return 0
    return 1 if pred_label == gold_label else 0

# ===== 推理并存储 =====
total_correct = 0
total_count = 0

with open(SAVE_PATH, "w", encoding="utf-8") as fout:
    for example in tqdm(dataset, desc="Generating CoTs"):
        prompt = f"""Context: {example['context']}
        Q: {example['question']}
        Let's reason step by step using the facts and rules in the context, then answer "yes" or "no".
        """
        gold_label = example["label"]

        for _ in range(N_SAMPLES):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            cot_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cot_text = cot_text[len(prompt):].strip()

            pred_label = extract_label_from_cot(cot_text)
            correctness = judge_correctness(pred_label, gold_label)

            fout.write(json.dumps({
                "idx": example.get("idx"),
                "raw_example": example,
                "prompt": prompt,
                "cot": cot_text,
                "pred_label": pred_label,
                "gold_label": gold_label,
                "is_correct": correctness
            }, ensure_ascii=False) + "\n")

            total_count += 1
            total_correct += correctness

# ===== 输出准确率 =====
accuracy = total_correct / total_count if total_count > 0 else 0
print(f"✅ 生成完成，已保存到 {SAVE_PATH}")
print(f"📊 总样本数: {total_count}, 正确数: {total_correct}, 准确率: {accuracy:.2%}")
