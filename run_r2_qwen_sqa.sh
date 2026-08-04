#!/usr/bin/env bash
# Qwen / StrategyQA | Table 1 主表 D-ProtoCoT + ORM + 各基线（run.py orm 一次出全表）
# 用法： nohup bash run_r2_qwen_sqa.sh &   然后 tail -f logs/qwen_sqa.log
set -e
DATA=newrundata/strategyqa_flat.jsonl
mkdir -p logs
exec > logs/qwen_sqa.log 2>&1
echo "==== START qwen_sqa $(date) ===="
python baseline/dprotocot/run.py orm --data_path "$DATA" --epochs 10 --seed 42
echo "==== DONE $(date) ===="
