# -*- coding: utf-8 -*-
"""
Inference-time path-selection baselines for comparison with D-ProtoCoT.

Each baseline follows the selector interface:
    selector(group, **kwargs) -> int   (index of selected path in group["paths"])

Implemented:
  1. Self-Certainty (Kang et al., NeurIPS 2025)
     Variant A — uses token logprobs stored in jsonl (if available).
     Variant B — uses a BERT encoder to compute "semantic certainty" as the
                 mean cosine similarity of a path to all other paths.  Higher
                 similarity ≈ the model consistently thinks in this direction.

  2. USC — Universal Self-Consistency (Chen et al., 2023)
     Asks an LLM to read all K paths and pick the best one.

  3. GenSelect (Toshniwal et al., ICML 2025 Workshop)
     Asks a reasoning-tuned LLM (QwQ / DeepSeek-R1) to do deep analysis and
     select the best path.  Optionally uses a structured output format.

  4. Pairwise-LLM — pairwise comparison + tournament (scalable fallback when
     K is too large for single-prompt USC).

All LLM-based methods accept an ``llm_call`` function so they work with any
backend (vLLM, OpenAI-compatible API, HuggingFace, etc.).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from data import question_text, path_text, extract_final_answer


# ============================================================================
# Type alias
# ============================================================================
# llm_call(messages: List[dict]) -> str
#   messages = [{"role": "system"/"user", "content": "..."}]
#   returns the LLM text response.
LLMCall = Callable[[List[Dict[str, str]]], str]

# selector(question_dict, **kwargs) -> int
Selector = Callable[..., int]


# ============================================================================
# Helper
# ============================================================================
def _parse_index(response: str, k: int) -> int:
    """Best-effort parse of a path number (1-indexed) from an LLM response."""
    # try explicit "Path X" or "X." patterns
    m = re.search(r"(?:path|#)\s*(\d+)", response, re.I)
    if m:
        idx = int(m.group(1)) - 1
        return max(0, min(idx, k - 1))
    # try the last integer in the response (often the answer)
    nums = re.findall(r"\b(\d+)\b", response)
    if nums:
        idx = int(nums[-1]) - 1
        return max(0, min(idx, k - 1))
    return 0  # safe fallback


# ============================================================================
# 1. Self-Certainty — Variant A (logprobs-based)
# ============================================================================
def sel_self_certainty_logprobs(group: dict, **_kw) -> int:
    """Select path with highest length-normalised log-probability.

    Requires each path dict to contain a ``logprobs`` key: either the
    aggregated scalar (``path["logprobs"]``) or a list of per-token
    logprobs (``path["token_logprobs"]``) that will be summed.
    """
    scores = []
    for p in group["paths"]:
        lp = p.get("logprobs", None)
        if lp is None:
            tok = p.get("token_logprobs", None)
            lp = sum(tok) if tok else 0.0
        length = max(1, len(p["cot"].split()))
        scores.append(lp / length)
    return int(max(range(len(scores)), key=lambda i: scores[i]))


# ============================================================================
# 2. Self-Certainty — Variant B (BERT-based semantic certainty)
# ============================================================================
@torch.no_grad()
def sel_self_certainty_bert(group: dict, encoder=None, cfg=None, **_kw) -> int:
    """BERT-based proxy: path with highest mean similarity to all other paths.

    Intuition: when the encoder sees a path as "typical" w.r.t. other sampled
    paths for this question, the path is more likely to be correct.  This
    mirrors the Self-Certainty paper's insight without needing logprobs.

    Requires ``encoder`` (MultiGranularEncoder, any state — frozen or trained)
    and ``cfg`` (Config).
    """
    texts = [path_text(p["cot"], group, cfg) for p in group["paths"]]
    _, path_mat = encoder.encode_paths(texts)          # [K, H]
    z = F.normalize(path_mat, dim=-1)                  # [K, H]
    sim_mat = z @ z.T                                   # [K, K]
    # mean similarity to all *other* paths
    k = sim_mat.size(0)
    eye = torch.eye(k, device=sim_mat.device)
    scores = (sim_mat * (1 - eye)).sum(dim=1) / max(1, k - 1)
    return int(torch.argmax(scores).item())


# ============================================================================
# 3. USC — Universal Self-Consistency
# ============================================================================
def sel_usc(group: dict, llm_call: LLMCall, cfg=None, **_kw) -> int:
    """Ask an LLM to select the best reasoning path from K candidates.

    Requires ``llm_call(messages) -> str``.
    """
    q = group["question"]
    k = len(group["paths"])
    if k <= 1:
        return 0

    paths_block = ""
    for i, p in enumerate(group["paths"], 1):
        paths_block += (
            f"### Path {i}\n"
            f"{p['cot']}\n\n"
        )

    prompt = (
        f"Question: {q}\n\n"
        f"Below are {k} different reasoning paths for this question.\n\n"
        f"{paths_block}"
        f"Carefully read all {k} paths.  Identify which path contains the "
        f"most logically sound reasoning and is most likely to arrive at "
        f"the correct answer.  Respond with ONLY the path number (e.g., {1}).\n"
    )
    messages = [{"role": "user", "content": prompt}]
    response = llm_call(messages)
    return _parse_index(response, k)


# ============================================================================
# 4. GenSelect — Generative Selection
# ============================================================================
def sel_genselect(
    group: dict,
    llm_call: LLMCall,
    cfg=None,
    system_prompt: Optional[str] = None,
    **_kw,
) -> int:
    """Generative selection: let a reasoning-tuned LLM do deep analysis first.

    Suitable for QwQ, DeepSeek-R1, o1, etc.  The default system prompt
    encourages step-by-step comparison before committing to a selection.

    Requires ``llm_call(messages) -> str``.
    """
    q = group["question"]
    k = len(group["paths"])
    if k <= 1:
        return 0

    paths_block = ""
    for i, p in enumerate(group["paths"], 1):
        paths_block += (
            f"## Path {i}\n"
            f"{p['cot']}\n\n"
        )

    if system_prompt is None:
        system_prompt = (
            "You are an expert reasoning evaluator.  Your task is to compare "
            "multiple chain-of-thought solutions and identify the best one."
        )

    user = (
        f"Question: {q}\n\n"
        f"I generated {k} reasoning paths for this question:\n\n"
        f"{paths_block}"
        f"First, analyse each path step-by-step — where does each go wrong "
        f"or right?  Then decide which ONE path is most likely correct.  "
        f"Your final answer MUST be a single number on its own line, e.g.:\n"
        f"BEST: 3\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    response = llm_call(messages)
    return _parse_index(response, k)


# ============================================================================
# 5. Pairwise LLM — tournament selection (useful when K is large)
# ============================================================================
def sel_pairwise_llm(
    group: dict,
    llm_call: LLMCall,
    max_rounds: int = 3,
    cfg=None,
    **_kw,
) -> int:
    """Single-elimination tournament: compare pairs, advance winner.

    O(K) comparisons instead of O(K²).  ``max_rounds`` caps the total
    number of comparisons to avoid runaway cost.
    """
    q = group["question"]
    paths = group["paths"]
    k = len(paths)
    if k <= 1:
        return 0

    # indices still in the tournament
    candidates = list(range(k))
    rng = __import__("random").Random(hash(q) % (2**31))

    while len(candidates) > 1 and max_rounds > 0:
        max_rounds -= 1
        rng.shuffle(candidates)
        winners = []
        for i in range(0, len(candidates), 2):
            if i + 1 >= len(candidates):
                winners.append(candidates[i])
                continue
            a, b = candidates[i], candidates[i + 1]
            prompt = (
                f"Question: {q}\n\n"
                f"## Path A\n{paths[a]['cot']}\n\n"
                f"## Path B\n{paths[b]['cot']}\n\n"
                f"Which path has better reasoning?  Answer with just A or B."
            )
            resp = llm_call([{"role": "user", "content": prompt}])
            winner = b if re.search(r"\bB\b", resp) else a
            winners.append(winner)
        candidates = winners

    return candidates[0]


# ============================================================================
# Factory — build all LLM-based selectors from a single llm_call
# ============================================================================
def make_llm_selectors(llm_call: LLMCall) -> Dict[str, Selector]:
    """Return a dict of name -> selector for all LLM-based baselines.

    Usage::

        selectors = make_llm_selectors(my_vllm_call)
        for name, sel in selectors.items():
            acc = evaluate_groups(test_groups, sel)
            print(f"{name}: {acc:.2f}%")
    """
    from functools import partial

    return {
        "USC": partial(sel_usc, llm_call=llm_call),
        "GenSelect": partial(sel_genselect, llm_call=llm_call),
        "Pairwise-LLM": partial(sel_pairwise_llm, llm_call=llm_call),
    }
