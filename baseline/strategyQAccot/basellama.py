# zero_shot_llama.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import re

# =========================
# 1. 路径与配置
# =========================
MODEL_PATH = "/home2/zzl/model/Llama-2-7b-chat-hf"
TEST_PATH = "/home2/zzl/C-CoT/database/StrategyQA/strategyqa_train.json"  # 必须是带答案的 dev/train
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256

# =========================
# 2. 加载模型和 tokenizer
# =========================
print("✅ 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

print("✅ 加载 Llama-2-7b-chat-hf 模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配 GPU
)
print("✅ 模型加载完成")

# 创建 pipeline（简化生成）
llama_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# =========================
# 3. 答案提取函数
# =========================
def extract_yes_no(text):
    """从文本末尾提取 yes/no"""
    # 优先看最后一句
    text = text.strip().lower()
    # 匹配最后出现的 yes/no
    matches = re.findall(r'\b(yes|no|true|false)\b', text)
    if matches:
        last = matches[-1]
        return "yes" if last in ["yes", "true"] else "no"
    return "no"  # 默认 fallback

# =========================
# 4. 加载测试集
# =========================
try:
    with open(TEST_PATH, 'r') as f:
        test_data = json.load(f)
    print(f"✅ 加载开发集: {len(test_data)} 条")
except FileNotFoundError:
    print(f"❌ 找不到 {TEST_PATH}")
    print("请确认是否存在 strategyqa_dev.json 或 strategyqa_train.json")
    exit()

# =========================
# 5. Zero-shot 推理
# =========================
correct = 0
total = 0

PROMPT_TEMPLATE = """[INST] <<SYS>>
You are a helpful, precise, and honest assistant.
<</SYS>>

Question: {question}
Let's think step by step. [/INST]"""

print("🔍 开始 zero-shot 推理...")

for item in tqdm(test_data[:100], desc="Generating"):  # 先测100条
    question = item["question"].strip()

    # === 提取真实答案 ===
    gold = item.get("answer") or item.get("label")
    if gold is None:
        continue
    if isinstance(gold, bool):
        gold = "yes" if gold else "no"
    else:
        gold = str(gold).strip().lower()
        gold = "yes" if "yes" in gold or "true" in gold else "no"

    # === 构造 prompt ===
    prompt = PROMPT_TEMPLATE.format(question=question)

    # === 生成回复 ===
    outputs = llama_pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # greedy decoding
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    raw_output = outputs[0]["generated_text"]
    # 去掉 prompt
    answer_text = raw_output[len(prompt):].strip()

    # === 提取预测答案 ===
    pred = extract_yes_no(answer_text)

    # === 判断正确与否 ===
    if pred == gold:
        correct += 1
    total += 1

    # === 打印前3个样例 ===
    if total <= 3:
        print(f"\n📌 Question: {question}")
        print(f"📝 Model: {answer_text}")
        print(f"✅ Gold: {gold}, Pred: {pred}, Correct: {pred == gold}")

# =========================
# 6. 输出准确率
# =========================
acc = correct / total if total > 0 else 0
print(f"\n🎯 Llama-2-7b-chat-hf Zero-shot Accuracy: {acc:.4f} ({correct}/{total})")