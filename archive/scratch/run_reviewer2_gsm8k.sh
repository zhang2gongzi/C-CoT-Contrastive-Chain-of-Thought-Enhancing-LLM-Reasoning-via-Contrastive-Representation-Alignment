#!/usr/bin/env bash
# 种子版重跑 GSM8K/Qwen 三条实验（条 3 ORM / 条 4 granularity / 条 5 leakage）
# 顺序执行：三条共用同一 output_dir，并行会互相覆盖 encoder，必须串行。
# 用法：  nohup bash run_reviewer2_gsm8k.sh > reviewer2_gsm8k.log 2>&1 &
#        然后 tail -f reviewer2_gsm8k.log 看进度；断连也不影响。
set -e

TRAIN=newrundata/gsm8k_merged_flat.jsonl
TEST=newrundata/gsm8k_test_flat.jsonl
EPOCHS=10
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

echo "==================== START $TS ===================="

echo ">>> [1/3] ORM (条3)  $(date)"
python baseline/dprotocot/run.py orm \
    --train_path "$TRAIN" --test_path "$TEST" --epochs "$EPOCHS" \
    2>&1 | tee "logs/orm_${TS}.log"

echo ">>> [2/3] granularity (条4)  $(date)"
python baseline/dprotocot/run.py granularity \
    --train_path "$TRAIN" --test_path "$TEST" --epochs "$EPOCHS" \
    2>&1 | tee "logs/granularity_${TS}.log"

echo ">>> [3/3] leakage (条5)  $(date)"
python baseline/dprotocot/run.py leakage \
    --train_path "$TRAIN" --test_path "$TEST" --epochs "$EPOCHS" \
    2>&1 | tee "logs/leakage_${TS}.log"

echo "==================== DONE $(date) ===================="
echo "汇总结果："
grep -H "D-ProtoCoT\|ORM\|Results \[" logs/*_"${TS}".log
