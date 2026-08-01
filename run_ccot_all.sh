#!/bin/bash
cd /home2/zzl/C-CoT

echo "[$(date)] ===== GSM8K (Qwen3-8B) ====="
python newrun/ccot_prompting.py --dataset gsm8k \
    --data_path newrundata/gsm8k_test_flat.jsonl \
    --model_path /home2/zzl/model/Qwen3-8B \
    --output newrun/ccot_gsm8k_qwen.jsonl
echo "[$(date)] GSM8K done."

echo "[$(date)] ===== StrategyQA (LLaMA-3.1-8B) ====="
python newrun/ccot_prompting.py --dataset strategyqa \
    --data_path newrundata/strategyqa_flat.jsonl \
    --model_path /home2/zzl/model/Llama-3.1-8B-Instruct \
    --output newrun/ccot_sqa_llama.jsonl
echo "[$(date)] StrategyQA done."

echo "[$(date)] ===== CSQA (Qwen3-8B) ====="
python newrun/ccot_prompting.py --dataset csqa \
    --data_path newrundata/csqa_500_flat.jsonl \
    --csqa_choices /home2/zzl/C-CoT/database/commonsenseQA/train-00000-of-00001.parquet \
    --model_path /home2/zzl/model/Qwen3-8B \
    --output newrun/ccot_csqa_qwen.jsonl
echo "[$(date)] CSQA done."

echo "[$(date)] ===== ALL THREE DONE ====="
