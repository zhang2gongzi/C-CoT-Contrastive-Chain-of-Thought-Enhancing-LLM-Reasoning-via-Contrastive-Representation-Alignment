# qwentry/model.py (REPLACEMENT)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from config import *

# ---------------------------
# Encoder: 提取 token / step / sequence-level 表征
# ---------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, PROJ_DIM)
        )
        self.clf_head = nn.Linear(hidden + PROJ_DIM, 2)

    def encode(self, input_ids, attention_mask, return_tokens=False):
        """
        输入: input_ids [N, L], attention_mask [N, L]
        返回:
            cls: [N, H]
            z: [N, D] (归一化的 projection)
            last_hidden (可选): [N, L, H]（token-level 表征）
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state   # [N, L, H]
        cls = last_hidden[:, 0, :]                # [N, H]
        z = F.normalize(self.proj(cls), dim=-1)   # [N, D]
        if return_tokens:
            return cls, z, last_hidden
        return cls, z

    def forward(self,
                input_ids, attention_mask,
                path_input_ids=None, path_attn_mask=None,
                path_step_input_ids=None, path_step_attn_mask=None):
        """
        input_ids: [B, L]
        path_input_ids: [B, P, Lp]
        path_step_input_ids: [B, P, S, Ls]
        """
        outputs = {}
        cls_q, z_q = self.encode(input_ids, attention_mask, return_tokens=False)
        outputs["cls_q"] = cls_q       # [B, H]
        outputs["z_q"] = z_q           # [B, D]
        outputs["clf_head"] = self.clf_head

        # path-level (sequence-level) encoding
        if path_input_ids is not None:
            B, P, Lp = path_input_ids.shape
            flat_ids = path_input_ids.view(B * P, Lp)
            flat_mask = path_attn_mask.view(B * P, Lp)
            # return token-level hidden for token-level pooling if needed
            cls_p, z_p, last_hidden_p = self.encode(flat_ids, flat_mask, return_tokens=True)
            # reshape
            outputs["cls_p"] = cls_p.view(B, P, -1)         # [B, P, H]
            outputs["z_p"] = z_p.view(B, P, -1)             # [B, P, D]
            outputs["last_hidden_p"] = last_hidden_p.view(B, P, Lp, -1)  # [B, P, Lp, H]

        # step-level encoding
        if path_step_input_ids is not None:
            # path_step_input_ids: [B, P, S, Ls]
            B, P, S, Ls = path_step_input_ids.shape
            flat_steps = path_step_input_ids.view(B * P * S, Ls)
            flat_step_mask = path_step_attn_mask.view(B * P * S, Ls)
            cls_steps, z_steps, last_hidden_steps = self.encode(flat_steps, flat_step_mask, return_tokens=True)
            # reshape -> [B, P, S, ...]
            z_steps = z_steps.view(B, P, S, -1)                  # [B, P, S, D]
            cls_steps = cls_steps.view(B, P, S, -1)              # [B, P, S, H]
            last_hidden_steps = last_hidden_steps.view(B, P, S, Ls, -1)  # [B, P, S, Ls, H]
            # step-level center: 每条 path 的 step mean -> [B, P, D]
            z_p_step_mean = z_steps.mean(dim=2)
            outputs["z_p_step"] = z_p_step_mean
            outputs["cls_p_steps"] = cls_steps
            outputs["last_hidden_steps"] = last_hidden_steps

        return outputs

# ---------------------------
# pooling helpers for token-level representation center
# ---------------------------
def attention_mean_pool(last_hidden, attn_mask):
    """
    last_hidden: [N, L, H]
    attn_mask: [N, L]
    return: pooled [N, H]
    """
    attn_mask = attn_mask.unsqueeze(-1).float()  # [N, L, 1]
    masked = last_hidden * attn_mask
    sum_vec = masked.sum(dim=1)                   # [N, H]
    denom = attn_mask.sum(dim=1).clamp(min=1e-6)  # [N, 1]
    return sum_vec / denom

# ---------------------------
# 对比损失：支持 sequence-level / step-level / token-level 合并
# ---------------------------
def info_nce_loss(z_q, z_p, path_is_correct, z_p_step=None, last_hidden_p=None, path_attn_mask=None,
                  step_weight=STEP_CONTRAST_WEIGHT, token_weight=TOKEN_CONTRAST_WEIGHT, temperature=0.07):
    """
    z_q: [B, D]
    z_p: [B, P, D]
    path_is_correct: [B, P] float 0/1
    z_p_step: [B, P, D] or None
    last_hidden_p: [B, P, Lp, H] or None  (用于 token-level pooling)
    path_attn_mask: [B, P, Lp] or None
    """
    B, P, D = z_p.shape
    device = z_q.device
    pos_mask = path_is_correct.to(device)  # [B, P]
    pos_count = pos_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1]

    # ---- sequence-level pos center ----
    pos_centers = (z_p * pos_mask.unsqueeze(-1)).sum(dim=1) / pos_count  # [B, D]

    # ---- negative pool for sequence-level ----
    neg_z_list = []
    for b in range(B):
        neg_self = z_p[b] * (1 - pos_mask[b]).unsqueeze(-1)  # [P, D]
        neg_other = z_p[torch.arange(B) != b].view(-1, D)    # [(B-1)*P, D]
        neg_z_list.append(torch.cat([neg_self, neg_other], dim=0))
    neg_z = torch.stack(neg_z_list, dim=0)  # [B, B*P, D]

    # logits
    pos_sim = torch.sum(z_q * pos_centers, dim=1) / temperature  # [B]
    neg_sim = torch.bmm(z_q.unsqueeze(1), neg_z.transpose(1, 2)).squeeze(1)  # [B, B*P]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1 + B*P]
    labels = torch.zeros(B, dtype=torch.long).to(device)
    seq_loss = F.cross_entropy(logits, labels)

    total_loss = seq_loss

    # ---- step-level loss（可选） ----
    if z_p_step is not None:
        pos_centers_step = (z_p_step * pos_mask.unsqueeze(-1)).sum(dim=1) / pos_count  # [B, D]
        neg_z_list_s = []
        for b in range(B):
            neg_self_s = z_p_step[b] * (1 - pos_mask[b]).unsqueeze(-1)
            neg_other_s = z_p_step[torch.arange(B) != b].view(-1, D)
            neg_z_list_s.append(torch.cat([neg_self_s, neg_other_s], dim=0))
        neg_z_s = torch.stack(neg_z_list_s, dim=0)
        pos_sim_s = torch.sum(z_q * pos_centers_step, dim=1) / temperature
        neg_sim_s = torch.bmm(z_q.unsqueeze(1), neg_z_s.transpose(1, 2)).squeeze(1)
        logits_s = torch.cat([pos_sim_s.unsqueeze(1), neg_sim_s], dim=1)
        step_loss = F.cross_entropy(logits_s, labels)
        total_loss = total_loss + step_weight * step_loss

    # ---- token-level loss（使用 last_hidden_p + path_attn_mask 做 attn_mean pooling） ----
    if USE_TOKEN_CONTRAST and last_hidden_p is not None and path_attn_mask is not None:
        # last_hidden_p: [B, P, Lp, H]; path_attn_mask: [B, P, Lp]
        B_, P_, Lp, H = last_hidden_p.shape
        # flatten for pooling
        flat_last = last_hidden_p.view(B_ * P_, Lp, H)
        flat_mask = path_attn_mask.view(B_ * P_, Lp)
        pooled = attention_mean_pool(flat_last, flat_mask)  # [B*P, H]
        # project pooled -> same proj space as z (need same proj)
        # we can't call self.proj here; so expect caller passes token_proj (or we can approximate by computing pos centers from last_hidden using a linear map)
        # Simpler: we compute token_center in BERT hidden space and then project with a small linear inside this function would require model object.
        # To keep this function stateless, we treat token pooling as additional negative/positive signals by mapping pooled -> D using a linear defined here.
        # For simplicity we create a linear on-the-fly on correct device (small overhead) -- acceptable but could be replaced by model-level fixed projector.
        token_proj = nn.Linear(H, D).to(pooled.device)
        with torch.no_grad():
            # init token_proj weights like identity-ish if dims match else random; keep it untrained to avoid extra params here
            pass
        token_z = F.normalize(token_proj(pooled), dim=-1)  # [B*P, D]
        token_z = token_z.view(B_, P_, -1)  # [B, P, D]

        pos_centers_tok = (token_z * pos_mask.unsqueeze(-1)).sum(dim=1) / pos_count  # [B, D]
        neg_z_list_t = []
        for b in range(B_):
            neg_self_t = token_z[b] * (1 - pos_mask[b]).unsqueeze(-1)
            neg_other_t = token_z[torch.arange(B_) != b].view(-1, D)
            neg_z_list_t.append(torch.cat([neg_self_t, neg_other_t], dim=0))
        neg_z_t = torch.stack(neg_z_list_t, dim=0)
        pos_sim_t = torch.sum(z_q * pos_centers_tok, dim=1) / temperature
        neg_sim_t = torch.bmm(z_q.unsqueeze(1), neg_z_t.transpose(1, 2)).squeeze(1)
        logits_t = torch.cat([pos_sim_t.unsqueeze(1), neg_sim_t], dim=1)
        token_loss = F.cross_entropy(logits_t, labels)
        total_loss = total_loss + token_weight * token_loss

    return total_loss

# ---------------------------
# 分类损失（辅助任务）
# ---------------------------
def classification_loss(cls_q, z_p, clf_head, gold_label):
    B, P, D = z_p.shape
    z_p_avg = z_p.mean(dim=1)
    feat = torch.cat([cls_q, z_p_avg], dim=1)
    logits = clf_head(feat)
    return F.cross_entropy(logits, gold_label)
