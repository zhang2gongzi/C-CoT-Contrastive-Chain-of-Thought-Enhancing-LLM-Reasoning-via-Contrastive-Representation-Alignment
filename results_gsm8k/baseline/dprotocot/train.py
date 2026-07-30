# -*- coding: utf-8 -*-
"""
Train the D-ProtoCoT encoder with the step-level InfoNCE objective.

One optimizer step aggregates the per-question losses over a mini-batch of
`cfg.batch_size` questions. Only questions that contain both correct and
incorrect paths contribute a contrastive signal.
"""

import os
import random
import torch

from config import Config
from encoder import MultiGranularEncoder
from losses import info_nce_step
from data import question_text, path_text


def train_encoder(cfg: Config, train_groups, val_groups=None):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = MultiGranularEncoder(cfg).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    groups = list(train_groups)

    for ep in range(1, cfg.epochs + 1):
        random.Random(cfg.seed + ep).shuffle(groups)
        opt.zero_grad()
        running, n_used, in_batch = 0.0, 0, 0
        for i, g in enumerate(groups, 1):
            q_text = question_text(g, cfg)
            texts = [path_text(p["cot"], g, cfg) for p in g["paths"]]
            labels = [p["is_correct"] for p in g["paths"]]

            z_q = model.encode_text_pooled(q_text)
            step_mats, _ = model.encode_paths(texts)
            loss = info_nce_step(z_q, step_mats, labels, cfg.temperature,
                                 train_repr=cfg.train_repr)
            if loss is None:
                continue
            (loss / cfg.batch_size).backward()
            running += loss.item(); n_used += 1; in_batch += 1

            if in_batch == cfg.batch_size or i == len(groups):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step(); opt.zero_grad(); in_batch = 0

        avg = running / max(1, n_used)
        msg = f"[train] epoch {ep}/{cfg.epochs} contrastive_loss={avg:.4f} (questions used={n_used})"
        if val_groups is not None:
            msg += f" | val_loss={_val_loss(model, cfg, val_groups):.4f}"
        print(msg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save(cfg.output_dir)
    print(f"[train] encoder saved to {cfg.output_dir}")
    return model


@torch.no_grad()
def _val_loss(model, cfg, val_groups):
    model.eval()
    tot, n = 0.0, 0
    for g in val_groups:
        labels = [p["is_correct"] for p in g["paths"]]
        if not (0 < sum(labels) < len(labels)):
            continue
        z_q = model.encode_text_pooled(question_text(g, cfg))
        step_mats, _ = model.encode_paths([path_text(p["cot"], g, cfg) for p in g["paths"]])
        loss = info_nce_step(z_q, step_mats, labels, cfg.temperature, train_repr=cfg.train_repr)
        if loss is not None:
            tot += loss.item(); n += 1
    model.train()
    return tot / max(1, n)
