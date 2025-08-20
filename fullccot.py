# -*- coding: utf-8 -*-
"""
Full C-CoT experiment: Baseline vs. Baseline+Multi-Path+Logic-Verified Contrastive Alignment
- 支持两种来源：
  A) 直接读取已生成的 flat jsonl（每行一个样本含 "cot"）
  B) 调用本地 Qwen-7B-Chat 生成多路径 CoT
- 逻辑验证：调用 /home2/zzl/ChatLogic/pyDatalog_processing.py
- 表征：BERT (bert-base-uncased) 共享编码，投影到对比空间
- 损失：InfoNCE（输入↔有效CoT为正，↔无效/其他为负）+ 轻量分类 CE（用于准确率）
- 评估：Baseline Acc、Logic-Filtered Acc、Logic Pass Rate
"""

import os
import re
import json
import random
import subprocess
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== 配置 =====
# 数据 / 模型路径
# 如果已有 flat jsonl（每行含 "cot"），设置 USE_PREGEN_COT=True 并填好 PREGEN_JSONL
USE_PREGEN_COT = True
PREGEN_JSONL = "/home2/zzl/C-CoT/test_C-CoT/cot_generated_first10_flat_labeled.jsonl"

# 若需在线生成 CoT，配置 Qwen 与原始验证集路径
QWEN_DIR = "/home2/zzl/model_eval/modelscope_models/Qwen/Qwen-7B-Chat"
RAW_DEV_JSONL = "/home2/zzl/ChatLogic/PARARULE-Plus/Depth2/PARARULE_Plus_Depth2_shuffled_dev_huggingface.jsonl"
NUM_EXAMPLES = 50       # 生成/读取多少道题
N_SAMPLES = 4           # 每题多少条 CoT
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# 逻辑验证脚本
PYDATALOG_PATH = "/home2/zzl/ChatLogic/pyDatalog_processing.py"

# BERT 编码器（本地路径或名称），建议用你本地缓存路径，避免联网
BERT_MODEL = "/home2/zzl/model/bert-base-uncased"

# 训练参数
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MAX_LEN = 256
PROJ_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ===== 可选：多路径 CoT 生成（Qwen）=====
def maybe_generate_cots_qwen(raw_examples, n_paths=6):
    """对原始验证集生成多条 CoT。若 USE_PREGEN_COT=True 则跳过。"""
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    print("[C-CoT] Loading Qwen for multi-path decoding...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN_DIR, device_map="auto", trust_remote_code=True).eval()

    out = []
    for ex in raw_examples:
        q = ex.get("question") or ex.get("input") or str(ex)
        prompt = f"Q: {q}\nLet's think step by step, and end with 'Answer: yes' or 'Answer: no'."
        paths = []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            for _ in range(n_paths):
                out_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, top_p=TOP_P
                )
                txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                cot = txt[len(prompt):].strip() if txt.startswith(prompt) else txt
                paths.append(cot)
        out.append({"raw_example": ex, "paths": paths})
    return out


def read_raw_first_k(path, k):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= k: break
            items.append(json.loads(line))
    return items


# ===== 逻辑验证 =====
def logic_verify(cot_text):
    """调用 pyDatalog_processing.py 判断推理是否逻辑一致（True/False）"""
    if not os.path.exists(PYDATALOG_PATH):
        return 0
    try:
        res = subprocess.run(
            ["python3", PYDATALOG_PATH, cot_text],
            capture_output=True, text=True, timeout=10
        )
        out = (res.stdout or "").strip().lower()
        return 1 if ("true" in out and "false" not in out) else 0
    except Exception:
        return 0


# ===== 标签解析（从 CoT 尾部抽答案 yes/no）=====
YES_PAT = re.compile(r"\b(answer[:：]?\s*)?(yes|true)\b", re.I)
NO_PAT  = re.compile(r"\b(answer[:：]?\s*)?(no|false)\b", re.I)
def parse_pred_label(cot_text):
    t = cot_text.strip().lower()
    if YES_PAT.search(t): return 1
    if NO_PAT.search(t):  return 0
    return None  # 未解析到


# ===== 数据构建（题级聚合：每题多路径）=====
def build_question_level_data_from_pregen(pregen_jsonl, max_q=NUM_EXAMPLES):
    """
    输入是 flat 格式：每行一个 { raw_example, cot, gold_label, is_correct }
    按题目聚合到：qid -> {context, question, gold_label, paths=[{text, is_valid, pred_label}]}
    """
    groups = defaultdict(list)
    with open(pregen_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ex = d["raw_example"]
            qid = ex.get("id") or ex.get("question")  # 简单聚合键
            groups[qid].append(d)

    data = []
    for i, (qid, items) in enumerate(groups.items()):
        if i >= max_q: break
        ex0 = items[0]["raw_example"]
        context = ex0.get("context", "")
        question = ex0.get("question", "")
        gold = ex0.get("label", None)
        paths = []
        for it in items:
            cot = it.get("cot", "")
            pred = parse_pred_label(cot)
            # 若 flat 已有 is_correct，保持；否则调用逻辑验证
            is_valid = it.get("is_correct")
            if is_valid is None:
                is_valid = logic_verify(cot)
            paths.append({"text": cot, "is_valid": int(is_valid), "pred_label": pred})
        data.append({
            "qid": qid,
            "context": context,
            "question": question,
            "gold_label": gold,
            "paths": paths
        })
    return data


# ===== Dataset：题级样本（含多路径）=====
class CCotQuestionDataset(Dataset):
    def __init__(self, questions, tokenizer, max_len=256):
        """
        每个样本是一道题，内部有多条 CoT：
        {
          context, question, gold_label,
          paths: [{text, is_valid, pred_label}, ...]
        }
        """
        self.qs = questions
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.qs)

    def encode_text(self, text):
        return self.tok(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )

    def __getitem__(self, idx):
        q = self.qs[idx]
        inp_text = f"Context: {q['context']}\nQuestion: {q['question']}"
        enc_inp = self.encode_text(inp_text)
        # 收集 CoT（文本、逻辑标记、预测标签）
        path_texts, valids, pred_labels = [], [], []
        for p in q["paths"]:
            path_texts.append(p["text"] or "No reasoning.")
            valids.append(int(p.get("is_valid", 0)))
            pl = p.get("pred_label")
            pred_labels.append(-1 if pl is None else int(pl))

        # 编码 CoT（打包成 list of dict）
        enc_paths = [self.encode_text(t) for t in path_texts]
        item = {
            "input_ids": enc_inp["input_ids"].squeeze(0),
            "attention_mask": enc_inp["attention_mask"].squeeze(0),
            "gold_label": torch.tensor(-1 if q["gold_label"] is None else int(q["gold_label"]), dtype=torch.long),
            "path_texts": path_texts,
            "valids": torch.tensor(valids, dtype=torch.float),
            "pred_labels": torch.tensor(pred_labels, dtype=torch.long),
            "paths_input_ids": torch.stack([e["input_ids"].squeeze(0) for e in enc_paths]),
            "paths_attn_mask": torch.stack([e["attention_mask"].squeeze(0) for e in enc_paths]),
        }
        return item


# ===== 模型：共享 BERT + 投影 + 轻量分类头 =====
from transformers import BertTokenizer, BertModel

class EncoderProj(nn.Module):
    def __init__(self, bert_name=BERT_MODEL, proj_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        h = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, proj_dim)
        )
        self.clf_head = nn.Linear(h + proj_dim, 2)  # 用于 accuracy 的轻量分类

    def encode_text(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]           # [B, H]
        z = F.normalize(self.proj(cls), dim=-1)        # 对比空间 [B, D]
        return cls, z

    def forward(self, input_ids, attention_mask,
                paths_input_ids=None, paths_attn_mask=None):
        # 输入编码
        cls_q, z_q = self.encode_text(input_ids, attention_mask)   # [B,H], [B,D]

        outputs = {
            "cls_q": cls_q, "z_q": z_q,
        }

        if paths_input_ids is not None:
            # paths: [B, P, L] 展平成 [B*P, L]
            B, P, L = paths_input_ids.size()
            flat_ids = paths_input_ids.view(B * P, L)
            flat_mask = paths_attn_mask.view(B * P, L)
            cls_p, z_p = self.encode_text(flat_ids, flat_mask)     # [B*P,H],[B*P,D]
            cls_p = cls_p.view(B, P, -1)                           # [B,P,H]
            z_p   = z_p.view(B, P, -1)                             # [B,P,D]
            outputs.update({"cls_p": cls_p, "z_p": z_p})
        return outputs


# ===== 对比损失（每题多正样本）=====
def info_nce_multi(z_q, z_p, valids, temperature=0.07):
    """
    z_q: [B, D]
    z_p: [B, P, D]
    valids: [B, P] (0/1)
    我们将同一道题内的有效 CoT 作为正，其余（该题内无效 + 其他题所有 paths）视为负。
    这里用 “类内聚合” 策略：对每题的所有正样本取平均作为正中心；若无正样本，则只计算负项（跳过该样本）。
    """
    B, P, D = z_p.size()
    device = z_q.device
    # 正中心
    pos_mask = (valids > 0.5).float()                  # [B,P]
    pos_sum = torch.einsum("bp,bpd->bd", pos_mask, z_p)  # sum 正样本 [B,D]
    pos_cnt = pos_mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)     # [B,1]
    pos_center = F.normalize(pos_sum / pos_cnt, dim=-1)            # [B,D]

    # 构造负样本库（跨 batch 全部 paths）
    z_p_all = z_p.reshape(B*P, D)                       # [B*P, D]
    # logits: q 对 正中心 & 所有 paths（含正/负）
    pos_logit = (z_q * pos_center).sum(dim=-1, keepdim=True) / temperature  # [B,1]
    all_logits = torch.matmul(z_q, z_p_all.T) / temperature                  # [B,B*P]

    # 构造 label：正类在拼接后的第0列
    logits = torch.cat([pos_logit, all_logits], dim=1)  # [B, 1+B*P]
    labels = torch.zeros(B, dtype=torch.long, device=device)

    # 若某样本没有正样本（pos_mask全0），我们不对它计算监督（给个掩码）
    has_pos = (pos_mask.sum(dim=1) > 0).float()         # [B]
    loss_all = F.cross_entropy(logits, labels, reduction="none")    # [B]
    loss = (loss_all * has_pos).sum() / has_pos.clamp(min=1.0).sum()
    return loss


# ===== 分类损失（用于 accuracy 的提升观察）=====
def classification_loss(model_out, z_p, valids, gold_labels):
    """
    使用最佳有效 CoT（若存在）与输入拼接做分类；否则使用“最佳”无效 CoT。
    """
    cls_q = model_out["cls_q"]        # [B,H]
    z_q = model_out["z_q"]            # [B,D]
    z_p = z_p                         # [B,P,D]
    valids = valids                   # [B,P]
    B, P, D = z_p.size()
    device = z_p.device

    # 选择与 z_q 余弦相似度最高的有效 CoT；若无有效，则选全局最高
    sim = torch.matmul(z_q.unsqueeze(1), z_p.transpose(1,2)).squeeze(1)  # [B,P]
    # mask 无效为 -inf
    sim_valid = sim.masked_fill(valids < 0.5, float("-inf"))
    best_valid_idx = sim_valid.argmax(dim=1)  # [B]
    has_pos = (valids.sum(dim=1) > 0.5)       # [B]

    # 若无有效，fallback 到总体最高
    best_all_idx = sim.argmax(dim=1)          # [B]
    pick_idx = torch.where(has_pos, best_valid_idx, best_all_idx)  # [B]

    # 取对应 CoT 的投影 + 输入 CLS 拼接做分类
    B_idx = torch.arange(B, device=device)
    picked = z_p[B_idx, pick_idx, :]                                      # [B,D]
    feat = torch.cat([model_out["cls_q"], picked], dim=-1)                # [B,H+D]
    logits = model.clf_head(feat)                                         # [B,2]

    valid_mask = (gold_labels >= 0)                                       # 有 gold label 才算
    if valid_mask.any():
        loss = F.cross_entropy(logits[valid_mask], gold_labels[valid_mask])
    else:
        loss = torch.tensor(0.0, device=device)

    preds = logits.argmax(dim=1)
    return loss, preds


# ===== 评估：三项指标 =====
def evaluate(model, dataloader):
    model.eval()
    total, base_correct = 0, 0
    logic_total, logic_correct, logic_pass = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            paths_ids = batch["paths_input_ids"].to(DEVICE)       # [B,P,L]
            paths_mask = batch["paths_attn_mask"].to(DEVICE)
            valids = batch["valids"].to(DEVICE)                   # [B,P]
            gold = batch["gold_label"].to(DEVICE)                 # [B]

            out = model(input_ids=input_ids, attention_mask=attn,
                        paths_input_ids=paths_ids, paths_attn_mask=paths_mask)

            # Baseline：仅用输入 CLS 做一个最弱分类（不看 CoT）
            # 这里用同一个分类头，但把 CoT 部分置零（可理解为最弱基线）
            fake_zero = torch.zeros(out["z_q"].size(0), out["z_q"].size(1), device=DEVICE)
            feat_base = torch.cat([out["cls_q"], fake_zero], dim=-1)
            logits_base = model.clf_head(feat_base)
            preds_base = logits_base.argmax(dim=1)

            # Logic-Filtered：按上面 classification_loss 的策略
            z_p = out["z_p"]                                       # [B,P,D]
            cls_loss, preds_logic = classification_loss(out, z_p, valids, gold)

            # 统计
            mask_gold = (gold >= 0)
            total += mask_gold.sum().item()
            base_correct += ((preds_base == gold) & mask_gold).sum().item()

            # 通过逻辑验证：至少有一个有效 CoT
            has_pos = (valids.sum(dim=1) > 0.5) & mask_gold
            logic_pass += has_pos.sum().item()
            logic_total += has_pos.sum().item()
            if has_pos.any():
                logic_correct += (preds_logic[has_pos] == gold[has_pos]).sum().item()

    baseline_acc = base_correct / max(total, 1)
    logic_acc = (logic_correct / max(logic_total, 1)) if logic_total > 0 else 0.0
    pass_rate = logic_pass / max(total, 1)

    print(f"[Eval] Baseline Acc={baseline_acc:.4f} | Logic-Filtered Acc={logic_acc:.4f} | Pass Rate={pass_rate:.4f}")
    return baseline_acc, logic_acc, pass_rate


# ===== 训练循环 =====
def train(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            paths_ids = batch["paths_input_ids"].to(DEVICE)       # [B,P,L]
            paths_mask = batch["paths_attn_mask"].to(DEVICE)
            valids = batch["valids"].to(DEVICE)                   # [B,P]
            gold = batch["gold_label"].to(DEVICE)                 # [B]

            # 前向
            out = model(input_ids=input_ids, attention_mask=attn,
                        paths_input_ids=paths_ids, paths_attn_mask=paths_mask)

            # 对比损失（C-CoT 主体）
            z_q = out["z_q"]
            z_p = out["z_p"]
            loss_con = info_nce_multi(z_q, z_p, valids, temperature=0.07)

            # 分类损失（用于观测准确率）
            loss_clf, _ = classification_loss(out, z_p, valids, gold)

            # 逻辑感知权重：按 batch 内通过率加权（通过率越高，对比损失权重越大）
            pass_rate = (valids.sum(dim=1) > 0.5).float().mean().item()
            w_con = 1.0 + pass_rate        # [1,2] 之间
            w_clf = 1.0

            loss = w_con * loss_con + w_clf * loss_clf

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {ep}/{epochs} | Train Loss={avg_loss:.4f}")

        # 每个 epoch 做一次评估
        evaluate(model, val_loader)


# ===== 主流程：数据准备 → 训练/评估 =====
def main():
    # 1) 准备题级数据（多路径）
    if USE_PREGEN_COT:
        print("[C-CoT] Using pre-generated CoT jsonl...")
        qdata = build_question_level_data_from_pregen(PREGEN_JSONL, max_q=NUM_EXAMPLES)
    else:
        print("[C-CoT] Generating CoTs with Qwen...")
        raw = read_raw_first_k(RAW_DEV_JSONL, NUM_EXAMPLES)
        qdata = []
        gen = maybe_generate_cots_qwen(raw, n_paths=N_SAMPLES)
        for item in gen:
            ex = item["raw_example"]
            paths = [{"text": t, "is_valid": logic_verify(t), "pred_label": parse_pred_label(t)} for t in item["paths"]]
            qdata.append({
                "qid": ex.get("id") or ex.get("question"),
                "context": ex.get("context", ""),
                "question": ex.get("question", ""),
                "gold_label": ex.get("label", None),
                "paths": paths
            })

    # 简单划分 train/val
    random.shuffle(qdata)
    n_train = int(0.8 * len(qdata))
    train_qs, val_qs = qdata[:n_train], qdata[n_train:]
    print(f"[C-CoT] Train questions: {len(train_qs)} | Val questions: {len(val_qs)}")

    # 2) Tokenizer & Dataset
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_ds = CCotQuestionDataset(train_qs, tokenizer, max_len=MAX_LEN)
    val_ds   = CCotQuestionDataset(val_qs, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 3) 模型
    global model
    model = EncoderProj(bert_name=BERT_MODEL, proj_dim=PROJ_DIM).to(DEVICE)

    # 4) 训练 & 评估
    train(model, train_loader, val_loader, epochs=EPOCHS)

    # 5) 最终评估一遍
    print("[C-CoT] Final evaluation on val:")
    evaluate(model, val_loader)


if __name__ == "__main__":
    main()
