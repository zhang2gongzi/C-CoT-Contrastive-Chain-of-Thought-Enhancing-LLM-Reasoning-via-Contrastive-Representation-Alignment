#!/usr/bin/env bash
# 重生成 LLaMA/CSQA 干净数据（强制 Final Answer: 字母），修复全 0 标注 → 再跑 orm 填 Table 1
# 用法： nohup bash run_regen_csqa_llama.sh &   然后 tail -f logs/regen_csqa_llama.log
set -e
mkdir -p logs
exec > logs/regen_csqa_llama.log 2>&1
echo "==== [1/2] 重生成 CSQA-LLaMA $(date) ===="
python newrun/regen_csqa_llama.py \
    --model_path ${MODEL_DIR}/Llama-3.1-8B-Instruct \
    --parquet database/commonsenseQA/train-00000-of-00001.parquet \
    --ref_flat newrundata/csqa_llama_flat.jsonl \
    --out newrundata/csqa_llama_flat.jsonl --k 10 --seed 42
echo "==== [2/2] orm 填 Table 1 该格 $(date) ===="
python baseline/dprotocot/run.py orm --data_path newrundata/csqa_llama_flat.jsonl --epochs 10 --seed 42
echo "==== DONE $(date) ===="
