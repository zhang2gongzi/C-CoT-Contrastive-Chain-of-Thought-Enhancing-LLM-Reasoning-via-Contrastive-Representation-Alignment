# -*- coding: utf-8 -*-
"""
Dynamic prototype construction and path selection (paper Sections 3.4-3.5).

Given a question embedding z_q and K path embeddings {z_i}:
    w_i   = softmax_i( sim(z_q, z_i) )          # similarity weights, NO gold labels
    p_q   = sum_i w_i * z_i                      # dynamic per-question prototype
    a_i   = sim(z_i, p_q)                        # alignment score
    c*    = argmax_i a_i                          # selected path

This uses ALL K paths and requires no labels at inference time.
"""

import torch
import torch.nn.functional as F


def build_prototype(z_q: torch.Tensor, path_embs: torch.Tensor):
    """
    z_q:        [H]
    path_embs:  [K, H]
    returns:    (prototype [H], weights [K])
    """
    zq = F.normalize(z_q, dim=-1)
    zp = F.normalize(path_embs, dim=-1)
    sims = zp @ zq                          # [K]
    weights = torch.softmax(sims, dim=0)    # [K]
    prototype = (weights.unsqueeze(-1) * path_embs).sum(dim=0)  # [H]
    return prototype, weights


def select_path(z_q: torch.Tensor, path_embs: torch.Tensor) -> int:
    """Return the index of the path best aligned with the dynamic prototype."""
    prototype, _ = build_prototype(z_q, path_embs)
    proto = F.normalize(prototype, dim=-1)
    zp = F.normalize(path_embs, dim=-1)
    align = zp @ proto                      # [K]
    return int(torch.argmax(align).item())


def select_path_centroid(path_embs: torch.Tensor) -> int:
    """Naive baseline: pick the path closest to the (unweighted) centroid."""
    centroid = F.normalize(path_embs.mean(dim=0), dim=-1)
    zp = F.normalize(path_embs, dim=-1)
    return int(torch.argmax(zp @ centroid).item())
