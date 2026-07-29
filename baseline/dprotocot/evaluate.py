# -*- coding: utf-8 -*-
"""
Unified evaluation on a SINGLE shared test split, so every number is directly
comparable (addresses Reviewer #4's main-vs-ablation mismatch). Reports the
test-set size explicitly (Reviewer #2).

Methods evaluated (all share the same K sampled paths per question):
  * Standard CoT        -- first sampled path
  * Self-Consistency    -- majority vote over extracted final answers
  * Raw-BERT + Centroid -- frozen BERT, nearest to unweighted centroid
  * D-ProtoCoT          -- trained encoder + dynamic prototype selection
  * ORM (optional)      -- highest predicted correctness
  * Self-Certainty-BERT -- BERT semantic-certainty proxy (no logprobs needed)
  * USC / GenSelect / Pairwise-LLM (optional, requires LLM API)
"""

import re
import torch

from config import Config
from encoder import MultiGranularEncoder
from prototype import select_path, select_path_centroid
from data import question_text, path_text, extract_final_answer
from baselines import (
    sel_self_certainty_bert,
    make_llm_selectors,
)


def _norm(ans):
    if ans is None:
        return "<none>"
    return re.sub(r"[^a-z0-9]", "", str(ans).lower())


# --------------------------------------------------------------------------- #
# selectors: group -> selected path index
# --------------------------------------------------------------------------- #
def sel_standard(group, **_):
    return 0


def self_consistency_correct(group):
    """Return 1 if the majority-voted answer is correct, else 0."""
    buckets = {}
    for p in group["paths"]:
        key = _norm(extract_final_answer(p["cot"]))
        buckets.setdefault(key, []).append(p["is_correct"])
    # majority bucket by count, tie-break by index of first occurrence
    best_key = max(buckets, key=lambda k: len(buckets[k]))
    votes = buckets[best_key]
    return int(round(sum(votes) / len(votes)))  # correctness of the majority answer


@torch.no_grad()
def _path_embs(encoder, cfg, group):
    texts = [path_text(p["cot"], group, cfg) for p in group["paths"]]
    step_mats, path_mat = encoder.encode_paths(texts)
    return step_mats, path_mat


@torch.no_grad()
def sel_dprotocot(group, encoder=None, cfg=None):
    z_q = encoder.encode_text_pooled(question_text(group, cfg))
    step_mats, path_mat = _path_embs(encoder, cfg, group)
    if cfg.select_repr == "path":
        return select_path(z_q, path_mat)
    # step-select variant: prototype over all step units, score path by best step
    import torch.nn.functional as F
    rows, owner = [], []
    for idx, mat in enumerate(step_mats):
        rows.append(mat); owner.extend([idx] * mat.size(0))
    steps = torch.cat(rows, dim=0)
    zq = F.normalize(z_q, dim=-1)
    w = torch.softmax(F.normalize(steps, dim=-1) @ zq, dim=0)
    proto = F.normalize((w.unsqueeze(-1) * steps).sum(0), dim=-1)
    align = F.normalize(steps, dim=-1) @ proto
    best = {}
    for a, o in zip(align.tolist(), owner):
        best[o] = max(best.get(o, -1e9), a)
    return max(best, key=best.get)


@torch.no_grad()
def sel_centroid(group, encoder=None, cfg=None):
    _, path_mat = _path_embs(encoder, cfg, group)
    return select_path_centroid(path_mat)


def _acc(groups, selector, **kw):
    if not groups:
        return 0.0
    correct = 0
    for g in groups:
        idx = selector(g, **kw)
        correct += int(g["paths"][idx]["is_correct"])
    return 100.0 * correct / len(groups)


def evaluate_all(cfg: Config, test_groups, trained_encoder, orm_model=None):
    """Compute accuracy (%) for every method on the SAME test split."""
    n = len(test_groups)
    results = {"_test_questions": n}

    results["Standard CoT"] = _acc(test_groups, sel_standard)
    results["Self-Consistency"] = 100.0 * sum(
        self_consistency_correct(g) for g in test_groups) / max(1, n)

    # frozen (untrained) BERT for the naive centroid baseline
    frozen = MultiGranularEncoder(cfg).to(trained_encoder.device).eval()
    results["Raw-BERT + Centroid"] = _acc(test_groups, sel_centroid,
                                          encoder=frozen, cfg=cfg)

    # Self-Certainty (BERT proxy) — no logprobs needed
    results["Self-Certainty-BERT"] = _acc(test_groups, sel_self_certainty_bert,
                                          encoder=frozen, cfg=cfg)
    del frozen

    trained_encoder.eval()
    results["D-ProtoCoT"] = _acc(test_groups, sel_dprotocot,
                                 encoder=trained_encoder, cfg=cfg)

    if orm_model is not None:
        from orm import orm_select
        results["ORM"] = _acc(test_groups, lambda g: orm_select(orm_model, cfg, g))

    return results


def evaluate_llm_baselines(
    test_groups,
    llm_call,
    include_pairwise: bool = False,
) -> dict:
    """Evaluate LLM-based baselines (USC, GenSelect) on the test set.

    ``llm_call`` is an LLMCall function (see llm_utils.py).  These baselines
    are separate from evaluate_all because they incur API cost and latency.

    Returns a dict like {"USC": 72.5, "GenSelect": 74.1}.
    """
    selectors = make_llm_selectors(llm_call)
    if not include_pairwise:
        selectors.pop("Pairwise-LLM", None)

    results = {}
    n = len(test_groups)
    for name, sel in selectors.items():
        correct = 0
        for g in test_groups:
            try:
                idx = sel(g)
                correct += int(g["paths"][idx]["is_correct"])
            except Exception as e:
                print(f"[warn] {name} failed on qid={g.get('qid','?')}: {e}; "
                      f"fallback to path 0")
                correct += int(g["paths"][0]["is_correct"])
        results[name] = 100.0 * correct / max(1, n)
    results["_test_questions"] = n
    return results


def print_results(tag, results):
    print(f"\n==== Results [{tag}] | test questions = {results['_test_questions']} ====")
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"  {k:24s}: {v:6.2f}%")
