#!/usr/bin/env bash
# LLaMA / GSM8K | Table 1 主表 D-ProtoCoT + ORM + 各基线（run.py orm 一次出全表）
# 用法： nohup bash run_r2_llama_gsm8k.sh &   然后 tail -f logs/llama_gsm8k.log
set -e
DATA=newrundata/gsm8k_llama_flat.jsonl
mkdir -p logs
exec > logs/llama_gsm8k.log 2>&1
echo "==== START llama_gsm8k $(date) ===="
python baseline/dprotocot/run.py orm --data_path "$DATA" --epochs 10 --seed 42
echo "==== DONE $(date) ===="
