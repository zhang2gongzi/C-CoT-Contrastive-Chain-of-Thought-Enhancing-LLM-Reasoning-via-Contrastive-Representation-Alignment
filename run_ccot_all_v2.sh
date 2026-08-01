#!/bin/bash
cd /home2/zzl/C-CoT
MT=2048
echo "[$(date)] ===== GSM8K (Qwen3-8B, MT=$MT) ====="
python newrun/ccot_prompting.py --dataset gsm8k --data_path newrundata/gsm8k_test_flat.jsonl \
    --model_path /home2/zzl/model/Qwen3-8B --max_new_tokens $MT --output newrun/ccot_gsm8k_qwen.jsonl
echo "[$(date)] ===== StrategyQA (Qwen3-8B, MT=$MT) ====="
python newrun/ccot_prompting.py --dataset strategyqa --data_path newrundata/strategyqa_flat.jsonl \
    --model_path /home2/zzl/model/Qwen3-8B --max_new_tokens $MT --output newrun/ccot_sqa_qwen.jsonl
echo "[$(date)] ===== CSQA (Qwen3-8B, MT=$MT) ====="
python newrun/ccot_prompting.py --dataset csqa --data_path newrundata/csqa_500_flat.jsonl \
    --csqa_choices /home2/zzl/C-CoT/database/commonsenseQA/train-00000-of-00001.parquet \
    --model_path /home2/zzl/model/Qwen3-8B --max_new_tokens $MT --output newrun/ccot_csqa_qwen.jsonl
echo "[$(date)] ===== ALL DONE (v2) ====="
