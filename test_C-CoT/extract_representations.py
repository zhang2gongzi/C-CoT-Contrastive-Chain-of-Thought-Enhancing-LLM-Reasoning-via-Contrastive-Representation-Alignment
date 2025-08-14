import json, os, re, math
from tqdm import tqdm
import torch
import numpy as np
from modelscope import AutoModelForCausalLM, AutoTokenizer

# ====== 路径配置 ======
MODEL_PATH = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
INPUT_JSONL = "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first10_flat_labeled.jsonl"   # 你已生成的平面化文件
OUT_DIR = "/home2/zzl/C-CoT/repr_cache_qwen7b"                             # 向量输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# ====== 超参 ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 1024                 # 够用即可，防止溢出
POOL_LAST_N_LAYERS = 8         # 常用 trick：取最后4层平均
INCLUDE_PROMPT = False         # True 表示用 prompt+cot 作为输入；False 只编码 cot
SAVE_TOKEN_LEVEL = False       # 如需token级保存，改 True（很占空间）

# ====== 加载模型 & tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()

HAS_FAST = getattr(tokenizer, "is_fast", False)
print(f"[Info] tokenizer.is_fast = {HAS_FAST}")

# ====== 工具函数 ======
def pool_last_layers(hidden_states, last_n=1):
    """hidden_states: tuple(len=layers), 每层 [1, seq, hid]"""
    if last_n <= 1:
        return hidden_states[-1][0]  # [seq, hid]
    hs = torch.stack(hidden_states[-last_n:], dim=0).mean(dim=0)  # [1, seq, hid]
    return hs[0]

def masked_mean(hidden, attention_mask):
    """hidden: [seq, hid], attention_mask: [seq] (0/1)"""
    mask = attention_mask.float().unsqueeze(-1)  # [seq,1]
    s = (hidden * mask).sum(dim=0)
    denom = mask.sum(dim=0).clamp(min=1.0)       # 防止除0
    return s / denom

def encode_and_pool(text):
    """对任意文本编码，返回 (seq_vec, token_hidden, attn_mask)"""
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)
    hid = pool_last_layers(out.hidden_states, last_n=POOL_LAST_N_LAYERS)  # [seq, hid]
    attn = enc["attention_mask"][0]  # [seq]
    seq_vec = masked_mean(hid, attn)  # [hid]
    return seq_vec, hid, attn

def split_steps(cot_text: str):
    """优先按换行切；若无换行，用句号/分号兜底。"""
    if "\n" in cot_text:
        parts = [p.strip() for p in cot_text.split("\n")]
    else:
        parts = re.split(r"[.;；。]\s*", cot_text)
        parts = [p.strip() for p in parts]
    return [p for p in parts if p]

def step_vecs_fast_alignment(full_text, steps, full_hid, tokenizer, max_len=MAX_LEN):
    """
    Fast tokenizer 路径：用 offset_mapping 做字符→token 对齐，然后从 full_hid 切片池化。
    """
    enc2 = tokenizer(full_text, return_offsets_mapping=True, truncation=True, max_length=max_len)
    offsets = enc2["offset_mapping"]  # List[(start_char, end_char)]
    vecs = []
    cursor = 0
    for st in steps:
        pos = full_text.find(st, cursor)
        if pos < 0:
            # 找不到就回退：单独编码该 step
            v, _, _ = encode_and_pool(st)
            vecs.append(v.cpu().numpy())
            continue
        start_char, end_char = pos, pos + len(st)
        cursor = end_char

        tok_start, tok_end = None, None
        for ti, (a, b) in enumerate(offsets):
            if a <= start_char < b:
                tok_start = ti
                break
        for tj, (a, b) in enumerate(offsets):
            if a < end_char <= b:
                tok_end = tj + 1
                break
        # 近似回退
        if tok_start is None:
            for ti, (a, b) in enumerate(offsets):
                if a >= start_char:
                    tok_start = ti; break
        if tok_end is None:
            for tj, (a, b) in enumerate(offsets):
                if b >= end_char:
                    tok_end = tj + 1; break

        if tok_start is None or tok_end is None or tok_end <= tok_start:
            v, _, _ = encode_and_pool(st)
            vecs.append(v.cpu().numpy())
        else:
            slice_hid = full_hid[tok_start:tok_end]                  # [len, hid]
            v = slice_hid.mean(dim=0)                                # 简单平均
            vecs.append(v.cpu().numpy())
    return vecs

def step_vecs_encode_each(steps):
    """无 Fast tokenizer 时的兜底：每个 step 单独编码再池化。"""
    vecs = []
    for st in steps:
        if st:
            v, _, _ = encode_and_pool(st)
            vecs.append(v.cpu().numpy())
    return vecs

# ====== 主循环 ======
seq_vecs, step_vecs_all, meta_rows = [], [], []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    records = [json.loads(l) for l in f]

for rid, ex in tqdm(list(enumerate(records)), desc="Extracting"):
    cot = (ex.get("cot") or "").strip()
    prompt = (ex.get("prompt") or "").strip()
    if not cot:
        continue

    text = (prompt + "\n" + cot) if INCLUDE_PROMPT else cot

    # 整序列向量
    seq_vec, full_hid, full_attn = encode_and_pool(text)
    seq_vecs.append(seq_vec.cpu().numpy())

    # step 向量
    steps = split_steps(cot)
    if HAS_FAST:
        # 有 Fast tokenizer：优先用 offset 对齐到 full_hid
        step_vecs = step_vecs_fast_alignment(text, steps, full_hid, tokenizer, max_len=MAX_LEN)
    else:
        # 没有 Fast：每步单独编码
        step_vecs = step_vecs_encode_each(steps)
    step_vecs_all.append(step_vecs)

    meta_rows.append({
        "rid": rid,
        "idx": ex.get("idx"),
        "n_steps": len(step_vecs),
        "pred_label": ex.get("pred_label"),
        "gold_label": ex.get("gold_label"),
        "is_correct": ex.get("is_correct"),
    })

# ====== 落盘 ======
np.save(os.path.join(OUT_DIR, "seq_vecs.npy"), np.stack(seq_vecs, axis=0))  # [N, hidden]
np.save(os.path.join(OUT_DIR, "step_vecs.npy"), np.array(step_vecs_all, dtype=object), allow_pickle=True)

with open(os.path.join(OUT_DIR, "meta.jsonl"), "w", encoding="utf-8") as fw:
    for m in meta_rows:
        fw.write(json.dumps(m, ensure_ascii=False) + "\n")

print("Done:",
      len(seq_vecs), "sequence embeddings ->", os.path.join(OUT_DIR, "seq_vecs.npy"),
      "| step sets ->", os.path.join(OUT_DIR, "step_vecs.npy"),
      "| meta ->", os.path.join(OUT_DIR, "meta.jsonl"))
