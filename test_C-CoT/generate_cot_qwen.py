# generate_cot_qwen.py
import os
import json
import re
from tqdm import tqdm

import torch

# modelscope imports (你说的方式)
from modelscope import AutoModelForCausalLM, AutoTokenizer

# ----------------- 配置（可调整） -----------------
MODEL_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
DEV_JSONL = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth2/PARARULE_Plus_Depth2_shuffled_dev_huggingface.jsonl"
OUT_JSONL = "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first10.jsonl"

NUM_EXAMPLES = 10          # 读取前 10 条
N_SAMPLES = 5             # 每题生成 N 条 CoT（默认 10）
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------

def read_jsonl_first_k(path, k):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            items.append(json.loads(line))
    return items

def build_prompt_from_example(ex):
    """
    根据你的 jsonl 样式定制 prompt。
    我这里通用处理：优先用 "question" 或 "input" 字段；否则打印键供你确认。
    """
    # 常见字段尝试顺序
    for key in ("question", "Question", "input", "text", "prompt"):
        if key in ex:
            q = ex[key]
            break
    else:
        # 如果结构不同，打印 keys 并用整个对象的字符串作为提示
        print("Warning: example has unexpected keys:", list(ex.keys()))
        q = json.dumps(ex, ensure_ascii=False)
    # 这个是 CoT 风格的简单模板（可根据你需要调整）
    prompt = f"Q: {q}\nLet's think step by step."
    return prompt

def extract_answer_from_cot(text):
    """
    从生成的 CoT 文本中尽量抽取最终答案（启发式）。
    优先查找 'Answer:' / 'Final answer:' 等关键字，否则取最后一行的短片段。
    这个函数只作初筛，后续可根据数据集具体答案格式定制。
    """
    # 常见标识符
    patterns = [
        r"Final answer[:：]\s*(.+)$",
        r"Answer[:：]\s*(.+)$",
        r"Therefore[:,：]\s*(.+)$",
        r"Hence[:,：]\s*(.+)$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I | re.M)
        if m:
            return m.group(1).strip()
    # 否则取最后非空行
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]
    # 如果最后行是“Q: ...”之类的，回退
    if last.lower().startswith("q:") or "let's think" in last.lower():
        # fallback: try second last
        if len(lines) >= 2:
            return lines[-2]
    return last

def safe_model_load(model_dir):
    """
    尝试加载 modelscope 的 tokenizer & model。
    modelscope API 的具体参数可能与 transformers 略有差异，这里做 try/except 提示。
    """
    print("Loading tokenizer and model from:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        # 尝试常见的 device_map/8bit 加速（如可用）
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    except Exception as e:
        print("Warning: device_map='auto' failed; trying cpu/gpu load. Error:", e)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.to(DEVICE)
    model.eval()
    return tokenizer, model

def generate_multiple_cots(model, tokenizer, prompt, n=N_SAMPLES):
    """
    返回 n 条生成文本（raw），使用 sampling 配置。
    如果 modelscope 的 generate 接口需要不同参数名，请按照报错调整（我基于 transformers 风格）。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    generated_texts = []
    # 为避免一次性生成 n 个，按循环逐个生成，便于节省显存并增加多样性（也可以尝试 num_return_sequences）
    with torch.no_grad():
        for _ in range(n):
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # 移除 prompt 前缀（如果 model.generate 返回包含 prompt）
            if txt.startswith(prompt):
                cot = txt[len(prompt):].strip()
            else:
                # 有些模型不会回写 prompt
                # 尝试把生成文本中最接近 prompt 后面的部分作为 cot
                cot = txt
            generated_texts.append(cot)
    return generated_texts

def main():
    # 1) 加载数据
    examples = read_jsonl_first_k(DEV_JSONL, NUM_EXAMPLES)
    print(f"Loaded {len(examples)} examples (first {NUM_EXAMPLES}).")

    # 2) 加载模型
    tokenizer, model = safe_model_load(MODEL_DIR)

    # 3) 逐条生成并保存
    out_f = open(OUT_JSONL, 'w', encoding='utf-8')
    for idx, ex in enumerate(tqdm(examples, desc="Examples")):
        prompt = build_prompt_from_example(ex)
        cots = generate_multiple_cots(model, tokenizer, prompt, n=N_SAMPLES)

        # 抽取每条 cot 的“答案”
        cot_answers = [extract_answer_from_cot(c) for c in cots]

        record = {
            "idx": idx,
            "raw_example": ex,
            "prompt": prompt,
            "cots": cots,
            "cot_answers": cot_answers
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 简要打印每题的第1条作为 sanity check
        print(f"\n==== example {idx} ====")
        print("prompt:", prompt)
        print("cot sample #0:\n", cots[0][:500])
        print("extracted answer #0:", cot_answers[0])
    out_f.close()
    print("Saved generated CoTs to:", OUT_JSONL)

if __name__ == "__main__":
    main()
