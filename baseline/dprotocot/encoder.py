# -*- coding: utf-8 -*-
"""
Multi-granular BERT encoder.

Given a reasoning path, we build token-level hidden states for the ENTIRE
sequence (handling >512 tokens via hierarchical chunking, 400-token windows
with 50-token overlap, reassembled by averaging overlaps). From those we derive:

    * step-level embeddings : mean-pool token states within each '\n'-delimited step
    * path-level embedding  : mean of step embeddings

The question is encoded separately and pooled to a single vector z_q.

All BERT parameters are trainable (contrastive alignment fine-tunes them).
Everything is differentiable, so the same code is used at train and eval time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

from config import Config


class MultiGranularEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.bert_model)
        self.bert = BertModel.from_pretrained(cfg.bert_model)
        self.hidden = self.bert.config.hidden_size
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        # usable content length per window (leaving room for [CLS]/[SEP])
        self._win = min(cfg.chunk_size, cfg.max_seq_len - 2)
        self._stride = max(1, self._win - cfg.chunk_overlap)

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------ #
    # token-level hidden states for the whole sequence (with chunking)
    # ------------------------------------------------------------------ #
    def _token_hidden(self, content_ids: torch.Tensor) -> torch.Tensor:
        """
        content_ids: 1-D LongTensor of token ids WITHOUT special tokens, length T.
        returns: [T, H] per-token hidden states, overlaps averaged.
        """
        T = content_ids.size(0)
        H = self.hidden
        if T == 0:
            return torch.zeros(1, H, device=self.device)

        if T <= self._win:
            windows = [(0, T)]
        else:
            windows = []
            start = 0
            while start < T:
                end = min(start + self._win, T)
                windows.append((start, end))
                if end == T:
                    break
                start += self._stride

        buf = torch.zeros(T, H, device=self.device)
        cnt = torch.zeros(T, 1, device=self.device)
        for a, b in windows:
            ids = torch.cat([
                content_ids.new_tensor([self.cls_id]),
                content_ids[a:b],
                content_ids.new_tensor([self.sep_id]),
            ]).unsqueeze(0)  # [1, b-a+2]
            attn = torch.ones_like(ids)
            out = self.bert(input_ids=ids, attention_mask=attn).last_hidden_state[0]
            content = out[1:-1]  # drop [CLS]/[SEP] -> [b-a, H]
            buf[a:b] += content
            cnt[a:b] += 1.0
        return buf / cnt.clamp(min=1.0)

    # ------------------------------------------------------------------ #
    # encode ONE reasoning path -> (step_embs [M,H], path_emb [H])
    # ------------------------------------------------------------------ #
    def encode_path(self, text: str):
        text = text if text and text.strip() else "empty"
        enc = self.tokenizer(text, add_special_tokens=False,
                             return_offsets_mapping=True, truncation=True, max_length=512)
        ids = torch.tensor(enc["input_ids"], device=self.device, dtype=torch.long)
        offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token
        token_hidden = self._token_hidden(ids)  # [T, H]

        # map '\n'-delimited steps to character spans
        step_spans, pos = [], 0
        for seg in text.split(self.cfg.step_delimiter):
            start = pos
            end = pos + len(seg)
            if seg.strip():
                step_spans.append((start, end))
            pos = end + len(self.cfg.step_delimiter)
        if not step_spans:
            step_spans = [(0, len(text))]

        step_embs = []
        for (cs, ce) in step_spans:
            tok_idx = [i for i, (a, b) in enumerate(offsets)
                       if b > cs and a < ce and b > a]
            if not tok_idx:
                continue
            step_embs.append(token_hidden[tok_idx].mean(dim=0))
        if not step_embs:
            step_embs = [token_hidden.mean(dim=0)]

        step_mat = torch.stack(step_embs, dim=0)      # [M, H]
        path_emb = step_mat.mean(dim=0)               # [H]
        return step_mat, path_emb

    # ------------------------------------------------------------------ #
    # encode a question (or any short text) -> pooled [H]
    # ------------------------------------------------------------------ #
    def encode_text_pooled(self, text: str) -> torch.Tensor:
        text = text if text and text.strip() else "empty"
        enc = self.tokenizer(text, add_special_tokens=True, truncation=True,
                             max_length=self.cfg.max_seq_len, return_tensors="pt").to(self.device)
        out = self.bert(**enc).last_hidden_state[0]   # [L, H]
        mask = enc["attention_mask"][0].unsqueeze(-1).float()
        if self.cfg.pool == "cls":
            return out[0]
        return (out * mask).sum(0) / mask.sum().clamp(min=1.0)

    # convenience: encode all paths of a question
    def encode_paths(self, texts):
        steps, paths = [], []
        for t in texts:
            s, p = self.encode_path(t)
            steps.append(s)
            paths.append(p)
        return steps, torch.stack(paths, dim=0)  # list[[Mi,H]], [K,H]

    def save(self, out_dir: str):
        self.bert.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
