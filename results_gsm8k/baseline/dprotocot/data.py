# -*- coding: utf-8 -*-
"""
Data loading and answer-leakage controls.

Key design points that address reviewer concerns:
  * Grouping + splitting is done BY QUESTION ID, so paths of the same question
    never straddle train/test (no path-level leakage).                (Reviewer #2)
  * `input_mode` lets us strip / replace the final answer or keep only
    question+answer, which produces the leakage ablation.             (Reviewer #5)
  * Supports either official train/test files or a ratio split of one file,
    and reports the resulting test-set size.                          (Reviewer #2, #4)
"""

import json
import re
import random
from collections import OrderedDict, defaultdict
from typing import List, Dict, Optional

from config import Config


# --------------------------------------------------------------------------- #
# Answer extraction / masking (used for the leakage ablation)
# --------------------------------------------------------------------------- #
_ANSWER_PATTERNS = [
    re.compile(r"####\s*(.+)$", re.I),                                  # GSM8K style
    re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+)$", re.I),
    re.compile(r"\bso\s+the\s+answer\s+is\s*(.+)$", re.I),
    re.compile(r"\b(yes|no|true|false)\b\s*\.?\s*$", re.I),             # StrategyQA style
]


def extract_final_answer(cot: str) -> Optional[str]:
    """Best-effort extraction of the final answer span from a CoT path."""
    text = cot.strip()
    # search the last non-empty line first, then the whole text
    lines = [l for l in text.splitlines() if l.strip()]
    search_scopes = ([lines[-1]] if lines else []) + [text]
    for scope in search_scopes:
        for pat in _ANSWER_PATTERNS:
            m = pat.search(scope.strip())
            if m:
                return m.group(1).strip(" .:;)('\"")
    # fallback: trailing number
    m = re.search(r"(-?\d[\d,\.]*)\s*\.?\s*$", text)
    return m.group(1).strip() if m else None


def apply_input_mode(cot: str, cfg: Config) -> str:
    """Transform a path's text according to the ablation input mode."""
    if cfg.input_mode == "full":
        return cot
    ans = extract_final_answer(cot)
    if cfg.input_mode == "mask":
        if not ans:
            return cot
        # replace the LAST occurrence of the answer span with a placeholder
        idx = cot.rfind(ans)
        if idx == -1:
            return cot
        return cot[:idx] + cfg.answer_placeholder + cot[idx + len(ans):]
    if cfg.input_mode == "qa_only":
        # steps removed; keep only the extracted answer (question added by encoder)
        return ans if ans else ""
    return cot


# --------------------------------------------------------------------------- #
# Loading / grouping
# --------------------------------------------------------------------------- #
def _to_int_label(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    if s in {"yes", "true", "1"}:
        return 1
    if s in {"no", "false", "0"}:
        return 0
    return s  # keep raw (e.g. CSQA answer letter / GSM8K number)


def _read_flat_jsonl(path: str, cfg: Config) -> "OrderedDict[str, dict]":
    """Return qid -> {question, context, gold_label, paths:[{cot,is_correct}]}"""
    groups: "OrderedDict[str, dict]" = OrderedDict()
    n_lines, n_bad = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                d = json.loads(line)
                raw = d[cfg.f_raw]
                qid = str(raw[cfg.f_id])
                cot = d[cfg.f_cot]
                is_correct = int(bool(d[cfg.f_is_correct]))
            except (KeyError, json.JSONDecodeError):
                n_bad += 1
                continue
            if qid not in groups:
                groups[qid] = {
                    "qid": qid,
                    "question": raw.get(cfg.f_question, ""),
                    "context": raw.get(cfg.f_context, "") if cfg.use_context else "",
                    "gold_label": _to_int_label(d.get(cfg.f_gold, raw.get("label"))),
                    "paths": [],
                }
            groups[qid]["paths"].append({"cot": cot, "is_correct": is_correct})
    print(f"[data] {path}: {n_lines} lines, {n_bad} skipped, "
          f"{len(groups)} questions, "
          f"{sum(len(g['paths']) for g in groups.values())} paths total")
    return groups


def question_text(group: dict, cfg: Config) -> str:
    if cfg.use_context and group.get("context"):
        return f"Context: {group['context']}\nQuestion: {group['question']}"
    return group["question"]


def path_text(cot: str, group: dict, cfg: Config) -> str:
    """Path text fed to the encoder, honoring the ablation input mode."""
    transformed = apply_input_mode(cot, cfg)
    if cfg.input_mode == "qa_only":
        # question + answer only, no reasoning steps
        return f"{group['question']} {transformed}".strip()
    return transformed


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_splits(cfg: Config):
    """
    Returns (train_groups, val_groups, test_groups) as lists of question dicts.

    * If cfg.train_path/test_path are set -> official split (val carved from train).
    * Else -> split the single cfg.data_path by question with cfg.split_ratio.
    """
    rng = random.Random(cfg.seed)

    if cfg.train_path and cfg.test_path:
        train_g = list(_read_flat_jsonl(cfg.train_path, cfg).values())
        test_g = list(_read_flat_jsonl(cfg.test_path, cfg).values())
        rng.shuffle(train_g)
        n_val = max(1, int(0.1 * len(train_g)))
        val_g, train_g = train_g[:n_val], train_g[n_val:]
    else:
        groups = list(_read_flat_jsonl(cfg.data_path, cfg).values())
        rng.shuffle(groups)
        if cfg.subset_questions:
            groups = groups[: cfg.subset_questions]
        n = len(groups)
        r_tr, r_va, _ = cfg.split_ratio
        n_tr, n_va = int(r_tr * n), int(r_va * n)
        train_g = groups[:n_tr]
        val_g = groups[n_tr:n_tr + n_va]
        test_g = groups[n_tr + n_va:]

    print(f"[data] split -> train={len(train_g)} val={len(val_g)} "
          f"test={len(test_g)} questions "
          f"(TEST SET SIZE = {len(test_g)} questions)")
    return train_g, val_g, test_g


def trainable_questions(groups: List[dict]) -> List[dict]:
    """Keep questions usable for contrastive training: >=1 correct and >=1 incorrect path."""
    out = []
    for g in groups:
        n_pos = sum(p["is_correct"] for p in g["paths"])
        if 0 < n_pos < len(g["paths"]):
            out.append(g)
    print(f"[data] trainable questions (have both pos & neg paths): "
          f"{len(out)}/{len(groups)}")
    return out
