# -*- coding: utf-8 -*-
"""
Step-level InfoNCE contrastive objective (paper Eq. in Section 3.3).

For one question with pooled embedding z_q:
    positives = every step embedding of every CORRECT path
    denominator = every step embedding of ALL paths (correct + incorrect)

    L = - 1/(|P|*M) * sum_{i in P} sum_j log
            exp(sim(z_q, s_ij)/tau) / sum_{all k,l} exp(sim(z_q, s_kl)/tau)

This propagates the single outcome-level label to |P|*M positive pairs without
any step-level annotation. A `train_repr="path"` variant collapses each path to
one vector (|P| positives) for the granularity ablation.
"""

import torch
import torch.nn.functional as F


def _sim(a, b):
    # cosine similarity between [D] and [N, D]
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return b @ a  # [N]


def info_nce_step(z_q, step_mats, is_correct, tau, train_repr="step"):
    """
    z_q:        [H] question embedding
    step_mats:  list of length K, each [M_i, H] step embeddings of path i
    is_correct: list/1-D tensor length K of {0,1}
    Returns a scalar loss (or None if the question has no positive path).
    """
    device = z_q.device
    if train_repr == "path":
        # collapse each path to a single vector (mean of its steps)
        units, unit_pos = [], []
        for mat, c in zip(step_mats, is_correct):
            units.append(mat.mean(dim=0))
            unit_pos.append(int(c))
        cand = torch.stack(units, dim=0)                 # [K, H]
        pos_mask = torch.tensor(unit_pos, device=device, dtype=torch.bool)
    else:
        # every step is an independent candidate
        rows, flags = [], []
        for mat, c in zip(step_mats, is_correct):
            rows.append(mat)
            flags.extend([int(c)] * mat.size(0))
        cand = torch.cat(rows, dim=0)                    # [sum_M, H]
        pos_mask = torch.tensor(flags, device=device, dtype=torch.bool)

    if pos_mask.sum() == 0:
        return None

    logits = _sim(z_q, cand) / tau                       # [N]
    logsumexp_all = torch.logsumexp(logits, dim=0)       # scalar denominator
    pos_logits = logits[pos_mask]                        # [P']
    # mean over positives of  log softmax  = mean(pos_logit - logsumexp_all)
    loss = -(pos_logits - logsumexp_all).mean()
    return loss
