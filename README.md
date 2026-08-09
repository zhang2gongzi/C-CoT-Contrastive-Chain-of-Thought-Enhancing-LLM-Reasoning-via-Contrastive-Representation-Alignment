# D-ProtoCoT

Official code for **D-ProtoCoT: Enhancing LLM Reasoning via Contrastive Representation Alignment**.

D-ProtoCoT is an *inference-time* framework for selecting high-quality Chain-of-Thought
reasoning paths. Instead of scoring paths by token-level confidence (logprobs) or by an
external LLM judge, it selects in a **contrastively aligned representation space**: a
`bert-base-uncased` encoder is fine-tuned with a step-level InfoNCE objective, and at
inference a **dynamic, per-question prototype** aggregates the sampled path embeddings so
that the path best aligned with the prototype is chosen. No modification to the backbone
LLM and no annotation beyond gold answers are required.

---

## Repository layout

```
.
├── baseline/dprotocot/      # core method + all training/eval code (see its own README)
│   ├── config.py            # central config (hyper-params, paths, ablation switches)
│   ├── data.py              # jsonl loading, question-grouped splitting, leakage control
│   ├── encoder.py           # MultiGranularEncoder: chunked, multi-granularity embeddings
│   ├── losses.py            # step-level InfoNCE contrastive loss
│   ├── prototype.py         # dynamic prototype construction + path selection
│   ├── train.py             # encoder training loop
│   ├── orm.py               # class-balanced ORM baseline + diagnostics (F1/AUROC)
│   ├── evaluate.py          # unified evaluation of all methods on one shared test set
│   ├── llm_utils.py         # LLM backends for USC / GenSelect (vLLM / OpenAI / ...)
│   └── run.py               # CLI entry point (main / orm / leakage / granularity / baselines)
├── newrun/                  # rebuttal / analysis scripts
│   ├── self_certainty.py            # Self-Certainty (Kang et al.) logprob selector
│   ├── mixed_question_analysis.py   # saturation / mixed-path subset analysis (Q1)
│   ├── token_stats.py               # token-length statistics (Table: token length)
│   ├── fix_labels.py                # unified answer extractor / relabeler
│   └── ...                          # t-SNE, PiCSAR, C-CoT prompting, etc.
├── newrundata/              # generated reasoning-path datasets (see "Data" below)
├── requirements.txt
└── cas-dc-template.tex      # manuscript source
```

> The core method lives entirely under `baseline/dprotocot/`; that folder has a
> module-by-module README documenting the exact I/O of every file.

---

## Installation

```bash
git clone https://github.com/USERNAME/D-ProtoCoT.git
cd D-ProtoCoT
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Core dependencies: `torch>=2.1`, `transformers>=4.44`, `numpy`, `tqdm`. Analysis scripts
under `newrun/` additionally use `pandas`, `scikit-learn`, `matplotlib`, `openTSNE`.

The encoder is `bert-base-uncased`. Pass its location (a local path or the Hugging Face id
`bert-base-uncased`) via `--bert_model`. The reasoning paths were generated with
LLaMA-3.1-8B-Instruct, Qwen3-8B, and Qwen3-14B.

---

## Data

Each dataset is a **flat jsonl**, one sampled reasoning path per line; lines sharing
`raw_example.id` are the K paths of the same question:

```json
{
  "raw_example": {"id": "q_001", "question": "...", "context": "(optional)", "label": 4},
  "cot": "Step 1: ...\nStep 2: ...\nThe answer is 4.",
  "gold_label": 4,
  "is_correct": 1
}
```

Files under `newrundata/` (per benchmark × backbone):

| File | Benchmark / Backbone |
|------|----------------------|
| `gsm8k_merged_flat.jsonl`, `gsm8k_test_flat.jsonl` | GSM8K / Qwen3-8B (official test split) |
| `gsm8k_llama_flat.jsonl` | GSM8K / LLaMA-3.1-8B (official test split) |
| `csqa_500_flat.jsonl` | CommonsenseQA / Qwen3-8B |
| `csqa_llama_flat.jsonl` | CommonsenseQA / LLaMA-3.1-8B |
| `strategyqa_flat.jsonl` | StrategyQA / Qwen3-8B |
| `strategyqa_llama_flat.jsonl` | StrategyQA / LLaMA-3.1-8B |

> **Note:** `*.jsonl` files are excluded from the git history via `.gitignore` because of
> their size. They are released separately; see the release/data link on the repository
> page. Exact train/val/test counts and the splitting protocol for every setting are given
> in the *Data Splits* table of the paper.

---

## Reproducing the main results

All experiments run through `baseline/dprotocot/run.py`. Run from inside that folder:

```bash
cd baseline/dprotocot
```

**1. Main table — one benchmark/backbone column** (Standard CoT / Self-Consistency /
Static-Prototype / Self-Certainty-BERT / D-ProtoCoT):

```bash
# Ratio-split setting (CommonsenseQA, StrategyQA): 8:1:1 grouped by question id
python run.py main \
    --bert_model bert-base-uncased \
    --data_path ../../newrundata/strategyqa_llama_flat.jsonl \
    --use_context --epochs 10

# Official-test setting (GSM8K): explicit train/test files
python run.py main \
    --bert_model bert-base-uncased \
    --train_path ../../newrundata/gsm8k_merged_flat.jsonl \
    --test_path  ../../newrundata/gsm8k_test_flat.jsonl \
    --epochs 10
```

**2. Add the ORM baseline** (same encoder / split / supervision, plus F1 & AUROC
diagnostics):

```bash
python run.py orm \
    --bert_model bert-base-uncased \
    --data_path ../../newrundata/csqa_500_flat.jsonl --epochs 10
```

**3. Answer-leakage ablation** (retrains full / mask / qa_only on the same split):

```bash
python run.py leakage --bert_model bert-base-uncased \
    --data_path ../../newrundata/gsm8k_test_flat.jsonl --epochs 10
```

**4. Representation-granularity ablation** (path/path, step/step, step/path):

```bash
python run.py granularity --bert_model bert-base-uncased \
    --train_path ../../newrundata/gsm8k_merged_flat.jsonl \
    --test_path  ../../newrundata/gsm8k_test_flat.jsonl --epochs 10
```

Each run prints a table like:

```
==== Results [main | input=full] | test questions = 200 ====
  Standard CoT            :  XX.XX%
  Self-Consistency        :  XX.XX%
  Raw-BERT + Centroid     :  XX.XX%   # = Static-Prototype in the paper
  Self-Certainty-BERT     :  XX.XX%
  D-ProtoCoT              :  XX.XX%
```

> **Protocol note:** the paper's main table uses `--epochs 10`. The default in `config.py`
> is 3; always pass `--epochs 10` to match the reported numbers. `--use_context` is used
> for StrategyQA (which has a context field) and omitted for GSM8K/CommonsenseQA.

### Additional-baseline scripts (`newrun/`)

```bash
# Self-Certainty (Kang et al.) — logprob-based selector, needs the generator LLM
python newrun/self_certainty.py --model_path <Qwen3-8B path> \
    --data_path newrundata/gsm8k_test_flat.jsonl --num_paths 10

# Saturation / mixed-path subset analysis (scaling to 14B)
python newrun/mixed_question_analysis.py --bert_model bert-base-uncased \
    --train_path <14B train> --test_path <14B test> --epochs 10

# Token-length statistics
python newrun/token_stats.py
```

LLM-judge baselines (USC / GenSelect / Pairwise) are available via
`run.py baselines --llm_backend vllm ...`; see `baseline/dprotocot/README.md` for the LLM
backend options.

---

## Method at a glance

```
w_i  = softmax( sim(z_q, z_i) )     # weights over K sampled paths, no labels at inference
p_q  = Σ_i w_i · z_i                # dynamic, per-question prototype
a_i  = sim(z_i, p_q)                # alignment score
c*   = argmax_i a_i                 # selected path
```

Training uses a **step-level** InfoNCE loss (each reasoning step is an independent
alignment target, giving |paths|·|steps| positive pairs from outcome-only labels), while
selection operates at the **path level** — the asymmetric train/inference granularity that
is the core design choice.

---

## Hardware

8B experiments were run on a single NVIDIA RTX 3090; the Qwen3-14B experiment was run on an
NVIDIA RTX 5090 (bf16). A single 24 GB GPU is sufficient for all 8B settings.

---

## Citation

```bibtex
@article{dprotocot2026,
  title  = {D-ProtoCoT: Enhancing LLM Reasoning via Contrastive Representation Alignment},
  author = {Zhang, Zhilei and others},
  year   = {2026}
}
```
