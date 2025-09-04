# qwentry/trainer.py (REPLACEMENT)
import torch
import torch.optim as optim
from tqdm import tqdm

from config import *
from model import info_nce_loss, classification_loss

# 训练函数
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        # load batch (注意 batch 里 path tensors shapes)
        input_ids = batch["input_ids"].to(DEVICE)                    # [B, L]
        attn_mask = batch["attention_mask"].to(DEVICE)               # [B, L]
        path_ids = batch["path_input_ids"].to(DEVICE)                # [B, P, Lp]
        path_mask = batch["path_attn_mask"].to(DEVICE)               # [B, P, Lp]
        gold_label = batch["gold_label"].to(DEVICE)                  # [B]
        path_is_correct = batch["path_is_correct"].to(DEVICE)       # [B, P]
        path_preds = batch["path_preds"].to(DEVICE)                  # [B, P]

        # step-level
        path_step_ids = batch.get("path_step_input_ids", None)
        path_step_mask = batch.get("path_step_attn_mask", None)
        if path_step_ids is not None:
            path_step_ids = path_step_ids.to(DEVICE)                 # [B, P, S, Ls]
            path_step_mask = path_step_mask.to(DEVICE)               # [B, P, S, Ls]

        # forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            path_input_ids=path_ids,
            path_attn_mask=path_mask,
            path_step_input_ids=path_step_ids,
            path_step_attn_mask=path_step_mask
        )
        cls_q = outputs["cls_q"]
        z_q = outputs["z_q"]
        z_p = outputs["z_p"]
        clf_head = outputs["clf_head"]
        last_hidden_p = outputs.get("last_hidden_p", None)          # [B, P, Lp, H] or None
        z_p_step = outputs.get("z_p_step", None)

        # losses
        contrast_loss = info_nce_loss(
            z_q=z_q, z_p=z_p, path_is_correct=path_is_correct,
            z_p_step=z_p_step, last_hidden_p=last_hidden_p, path_attn_mask=path_mask
        )
        clf_loss = classification_loss(cls_q, z_p, clf_head, gold_label)
        loss = contrast_loss + 0.5 * clf_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数（保持原有逻辑，只使用预存在 path_preds 与 is_correct）
def evaluate(model, dataloader):
    model.eval()
    baseline_correct = 0
    logic_correct = 0
    total = 0
    passed = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            path_ids = batch["path_input_ids"].to(DEVICE)
            path_mask = batch["path_attn_mask"].to(DEVICE)
            path_is_correct = batch["path_is_correct"].cpu().numpy()
            gold_labels = batch["gold_label"].to(DEVICE)
            path_preds = batch["path_preds"].cpu().numpy()

            for b in range(len(gold_labels)):
                gold_label = gold_labels[b].item()
                correct_mask = path_is_correct[b]
                preds = path_preds[b]
                total += 1

                # logic: only use correct CoT voting
                valid_preds = [p for p, c in zip(preds, correct_mask) if c == 1 and p != -1]
                if valid_preds:
                    passed += 1
                    pred = max(set(valid_preds), key=valid_preds.count)
                    if pred == gold_label:
                        logic_correct += 1

                # baseline: all CoT vote
                all_preds = [p for p in preds if p != -1]
                if all_preds:
                    baseline_pred = max(set(all_preds), key=all_preds.count)
                    if baseline_pred == gold_label:
                        baseline_correct += 1

    baseline_acc = baseline_correct / total if total > 0 else 0.0
    logic_acc = logic_correct / passed if passed > 0 else 0.0
    pass_rate = passed / total if total > 0 else 0.0
    return baseline_acc, logic_acc, pass_rate
