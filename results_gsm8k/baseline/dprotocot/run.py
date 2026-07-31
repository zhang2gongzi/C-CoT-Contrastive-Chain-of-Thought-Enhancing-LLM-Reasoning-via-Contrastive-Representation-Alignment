# -*- coding: utf-8 -*-
"""
CLI entry point for the D-ProtoCoT reimplementation.

Examples
--------
# 1) Main comparison (Standard CoT / Self-Consistency / Centroid / D-ProtoCoT)
python run.py main \
    --data_path /path/strategyqa_flat_labeled.jsonl --use_context \
    --output_dir runs/sqa_llama

# 2) Add the fixed ORM baseline with full diagnostics
python run.py orm \
    --data_path /path/gsm8k_flat_labeled.jsonl --output_dir runs/gsm8k

# 3) Answer-leakage ablation (retrains for full / mask / qa_only, same split)
python run.py leakage --data_path /path/gsm8k_flat_labeled.jsonl

# 4) Representation-granularity ablation (path/path, step/step, step/path)
python run.py granularity --data_path /path/strategyqa_flat_labeled.jsonl --use_context

# 5) LLM-based baselines (USC, GenSelect) on the test set
python run.py baselines --data_path /path/data.jsonl --llm_backend vllm \
    --llm_model qwen3-8b --llm_base_url http://localhost:8000/v1

# Official split instead of a ratio split:
#   --train_path train.jsonl --test_path test.jsonl
"""

import argparse
import copy
import os

from config import Config
from data import load_splits, trainable_questions
from train import train_encoder
from evaluate import evaluate_all, evaluate_llm_baselines, print_results


def build_cfg(args) -> Config:
    cfg = Config()
    for k, v in vars(args).items():
        if k in {"command"} or v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg.resolve()


def _prepare(cfg):
    train_g, val_g, test_g = load_splits(cfg)
    train_t = trainable_questions(train_g)
    val_t = trainable_questions(val_g)
    return train_t, val_t, test_g


def cmd_main(cfg):
    train_t, val_t, test_g = _prepare(cfg)
    enc = train_encoder(cfg, train_t, val_t)
    res = evaluate_all(cfg, test_g, enc)
    print_results(f"main | input={cfg.input_mode}", res)


def cmd_orm(cfg):
    from orm import train_orm, eval_orm_diagnostics
    train_g, val_g, test_g = load_splits(cfg)
    train_t = trainable_questions(train_g)
    val_t = trainable_questions(val_g)
    enc = train_encoder(cfg, train_t, val_t)
    orm_model = train_orm(cfg, train_g, val_g)          # ORM uses ALL paths, not only mixed ones
    diag = eval_orm_diagnostics(orm_model, cfg, test_g)
    print(f"\n[ORM] TEST diagnostics: loss={diag['loss']:.4f} "
          f"path_acc={diag['acc']:.4f} F1={diag['f1']:.4f} AUROC={diag['auroc']}")
    res = evaluate_all(cfg, test_g, enc, orm_model=orm_model)
    print_results(f"main+ORM | input={cfg.input_mode}", res)


def cmd_leakage(cfg):
    """Retrain + evaluate under full / mask / qa_only on the SAME split."""
    for mode in ["full", "mask", "qa_only"]:
        c = copy.copy(cfg); c.input_mode = mode
        train_t, val_t, test_g = _prepare(c)
        enc = train_encoder(c, train_t, val_t)
        res = evaluate_all(c, test_g, enc)
        print_results(f"leakage-ablation | input={mode}", res)
    print("\n[hint] if qa_only ~= full, gains are NOT from reasoning-process modeling.")


def cmd_granularity(cfg):
    for tr, se in [("path", "path"), ("step", "step"), ("step", "path")]:
        c = copy.copy(cfg); c.train_repr = tr; c.select_repr = se
        train_t, val_t, test_g = _prepare(c)
        enc = train_encoder(c, train_t, val_t)
        res = evaluate_all(c, test_g, enc)
        print_results(f"granularity | train={tr}/select={se}", res)


def cmd_baselines(cfg, args):
    """Standard methods + Self-Certainty-BERT + Centroid + D-ProtoCoT."""
    train_t, val_t, test_g = _prepare(cfg)
    enc = train_encoder(cfg, train_t, val_t)
    res = evaluate_all(cfg, test_g, enc)
    print_results(f"baselines | input={cfg.input_mode}", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["main", "orm", "leakage", "granularity", "baselines"])
    ap.add_argument("--bert_model")
    ap.add_argument("--data_path")
    ap.add_argument("--train_path")
    ap.add_argument("--test_path")
    ap.add_argument("--output_dir")
    ap.add_argument("--use_context", action="store_true", default=None)
    ap.add_argument("--subset_questions", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--input_mode", choices=["full", "mask", "qa_only"])
    ap.add_argument("--train_repr", choices=["step", "path"])
    ap.add_argument("--select_repr", choices=["step", "path"])
    # LLM-baseline args
    ap.add_argument("--llm_backend", help="vllm | together | deepinfra | openai | dry-run")
    ap.add_argument("--llm_model", default="qwen3-8b", help="model name for the LLM backend")
    ap.add_argument("--llm_base_url", help="override API base URL")
    ap.add_argument("--llm_api_key", help="API key (or set env var)")
    ap.add_argument("--include_pairwise", action="store_true", help="also run Pairwise-LLM baseline")
    args = ap.parse_args()

    cfg = build_cfg(args)
    {
        "main": cmd_main, "orm": cmd_orm,
        "leakage": cmd_leakage, "granularity": cmd_granularity,
        "baselines": lambda c: cmd_baselines(c, args),
    }[args.command](cfg)


if __name__ == "__main__":
    main()
