# -*- coding: utf-8 -*-
"""
ORM (Outcome Reward Model) baseline -- implemented carefully and fairly.

This directly answers Reviewer #3: the original ORM collapsed to near-random.
Here we:
  * use the SAME chunking encoder as D-ProtoCoT (fair representation),
  * concatenate [z_q ; z_path] -> Linear -> 1 logit,
  * train with BCEWithLogitsLoss and pos_weight = #neg/#pos to counter the
    heavy class imbalance (e.g. Qwen/GSM8K where ~90% of paths are correct),
  * select the path with the HIGHEST predicted correctness (correct sign),
  * report train/val loss, path-level accuracy, F1, AUROC and positive ratio.

If these diagnostics look healthy but accuracy is still low, the problem is the
data, not the optimizer -- which is exactly what the reviewer asked us to show.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from encoder import MultiGranularEncoder
from data import question_text, path_text

try:
    from sklearn.metrics import f1_score, roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


class ORM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.enc = MultiGranularEncoder(cfg)
        self.head = nn.Linear(2 * self.enc.hidden, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def score_path(self, q_text: str, p_text: str) -> torch.Tensor:
        z_q = self.enc.encode_text_pooled(q_text)
        _, z_p = self.enc.encode_path(p_text)
        feat = torch.cat([z_q, z_p], dim=-1)
        return self.head(feat).squeeze(-1)  # scalar logit


def _flatten(groups, cfg):
    items = []
    for g in groups:
        q = question_text(g, cfg)
        for p in g["paths"]:
            items.append((q, path_text(p["cot"], g, cfg), int(p["is_correct"])))
    return items


def train_orm(cfg: Config, train_groups, val_groups):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = ORM(cfg).to(device)
    model.train()

    train_items = _flatten(train_groups, cfg)
    n_pos = sum(y for *_, y in train_items)
    n_neg = len(train_items) - n_pos
    pos_ratio = n_pos / max(1, len(train_items))
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    print(f"[ORM] train paths={len(train_items)} pos={n_pos} neg={n_neg} "
          f"pos_ratio={pos_ratio:.3f} pos_weight={pos_weight.item():.3f}")

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        import random
        random.Random(cfg.seed + ep).shuffle(train_items)
        running, opt_steps = 0.0, 0
        opt.zero_grad()
        for i, (q, p, y) in enumerate(train_items, 1):
            logit = model.score_path(q, p)
            loss = crit(logit.unsqueeze(0), torch.tensor([float(y)], device=device))
            (loss / cfg.batch_size).backward()
            running += loss.item()
            if i % cfg.batch_size == 0 or i == len(train_items):
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step(); opt.zero_grad(); opt_steps += 1
        tr_loss = running / max(1, len(train_items))
        va = eval_orm_diagnostics(model, cfg, val_groups)
        print(f"[ORM] epoch {ep}/{cfg.epochs} train_loss={tr_loss:.4f} | "
              f"val_loss={va['loss']:.4f} path_acc={va['acc']:.4f} "
              f"F1={va['f1']:.4f} AUROC={va['auroc']}")
    return model


@torch.no_grad()
def eval_orm_diagnostics(model, cfg, groups):
    """Path-level classification diagnostics (loss/acc/F1/AUROC)."""
    device = model.device
    model.eval()
    items = _flatten(groups, cfg)
    if not items:
        return {"loss": 0.0, "acc": 0.0, "f1": 0.0, "auroc": "n/a"}
    ys, probs, losses = [], [], []
    bce = nn.BCEWithLogitsLoss()
    for q, p, y in items:
        logit = model.score_path(q, p)
        losses.append(bce(logit.unsqueeze(0), torch.tensor([float(y)], device=device)).item())
        probs.append(torch.sigmoid(logit).item())
        ys.append(y)
    preds = [1 if pr >= 0.5 else 0 for pr in probs]
    acc = sum(int(a == b) for a, b in zip(preds, ys)) / len(ys)
    if _HAS_SK:
        f1 = f1_score(ys, preds, zero_division=0)
        auroc = round(roc_auc_score(ys, probs), 4) if len(set(ys)) > 1 else "n/a"
    else:
        tp = sum(1 for a, b in zip(preds, ys) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(preds, ys) if a == 1 and b == 0)
        fn = sum(1 for a, b in zip(preds, ys) if a == 0 and b == 1)
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        auroc = "n/a(no sklearn)"
    return {"loss": sum(losses) / len(losses), "acc": acc, "f1": f1, "auroc": auroc}


@torch.no_grad()
def orm_select(model, cfg, group) -> int:
    """Select the path index with the highest predicted correctness."""
    model.eval()
    q = question_text(group, cfg)
    scores = [model.score_path(q, path_text(p["cot"], group, cfg)).item()
              for p in group["paths"]]
    return int(max(range(len(scores)), key=lambda i: scores[i]))
