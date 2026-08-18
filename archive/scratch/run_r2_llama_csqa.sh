#!/usr/bin/env bash
# LLaMA / CSQA | Table 1 主表 D-ProtoCoT + ORM + 各基线（run.py orm 一次出全表）
# ⚠️ 8-04 曾崩：trainable questions 0/336（无对/错混合路径）→ 跑前先查 is_correct 标注。
# 用法： nohup bash run_r2_llama_csqa.sh &   然后 tail -f logs/llama_csqa.log
set -e
DATA=newrundata/csqa_llama_flat.jsonl
mkdir -p logs
exec > logs/llama_csqa.log 2>&1
echo "==== START llama_csqa $(date) ===="
python baseline/dprotocot/run.py orm --data_path "$DATA" --epochs 10 --seed 42
echo "==== DONE $(date) ===="
