# -*- coding: utf-8 -*-
"""
Central configuration for the D-ProtoCoT reimplementation.

Every value here is overridable from the command line (see run.py). Defaults
follow the paper: bert-base-uncased, lr=2e-5, batch=16, epochs=3, tau=0.07,
hierarchical chunking 400/50.

Data schema expected (flat jsonl, one path per line):
    {
      "raw_example": {"id": <str>, "question": <str>, "context": <str, optional>, "label": <int/str>},
      "cot":         <str>,        # full reasoning path text (steps separated by '\n')
      "gold_label":  <int/str>,    # gold answer / label of the QUESTION
      "is_correct":  <0/1 or bool> # whether THIS path's final answer matches gold
    }
Lines sharing raw_example.id belong to the same question (its K sampled paths).
"""

from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class Config:
    # ---- paths (overridable via CLI; see run.py) ----
    bert_model: str = "${MODEL_DIR}/bert-base-uncased"
    # A single flat jsonl that will be split by question, OR a train/test pair.
    data_path: str = "${PROJECT_ROOT}/data/strategyqa_flat_labeled.jsonl"
    train_path: Optional[str] = None   # if set, use official split (train_path + test_path)
    test_path: Optional[str] = None
    output_dir: str = "${PROJECT_ROOT}/dprotocot_runs/run"

    # ---- reasoning-path field names (adjust here if your jsonl differs) ----
    f_raw: str = "raw_example"
    f_id: str = "id"
    f_question: str = "question"
    f_context: str = "context"       # set use_context=False to ignore
    f_cot: str = "cot"
    f_gold: str = "gold_label"
    f_is_correct: str = "is_correct"
    use_context: bool = False        # StrategyQA/PARARULE have context; CSQA/GSM8K usually not

    # ---- data / split ----
    subset_questions: Optional[int] = None  # None = use ALL questions (recommended for the paper fix)
    split_ratio: tuple = (0.8, 0.1, 0.1)    # train/val/test when using a single data_path
    seed: int = 42
    k_paths: int = 10                        # expected sampled paths per question (info only)
    min_paths_for_train: int = 2             # a training question needs >=2 paths (>=1 pos & >=1 neg ideally)

    # ---- encoder / representation ----
    max_seq_len: int = 512
    chunk_size: int = 400
    chunk_overlap: int = 50
    step_delimiter: str = "\n"
    pool: str = "mean"               # question / path pooling over content tokens

    # ---- training ----
    lr: float = 2e-5
    batch_size: int = 16             # questions per batch
    epochs: int = 3
    temperature: float = 0.07
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "cuda"
    num_workers: int = 2

    # ---- ablation controls ----
    # input mode for the reasoning path fed to the encoder:
    #   "full"     -> original CoT (may contain the final answer)   [main setting]
    #   "mask"     -> final answer span replaced with a placeholder [leakage ablation]
    #   "qa_only"  -> only question + extracted final answer, no steps [leakage ablation]
    input_mode: str = "full"
    answer_placeholder: str = "[ANS]"
    # representation granularity for training / selection (granularity ablation):
    #   train_repr in {"step","path"} ; select_repr in {"step","path"}
    train_repr: str = "step"
    select_repr: str = "path"

    def resolve(self):
        assert self.input_mode in {"full", "mask", "qa_only"}
        assert self.train_repr in {"step", "path"}
        assert self.select_repr in {"step", "path"}
        return self
