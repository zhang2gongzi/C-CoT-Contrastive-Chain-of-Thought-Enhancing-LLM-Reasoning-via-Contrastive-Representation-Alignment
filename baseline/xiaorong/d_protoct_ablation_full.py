# =============================
# 新增：使用本地 BERT 模型进行编码
# =============================
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 设置本地 BERT 路径
LOCAL_BERT_PATH = "/home2/zzl/model/bert-base-uncased"

print("Loading BERT tokenizer and model...")
tokenizer_bert = AutoTokenizer.from_pretrained(LOCAL_BERT_PATH)
model_bert = AutoModel.from_pretrained(LOCAL_BERT_PATH).to(DEVICE)
model_bert.eval()  # 确保是推理模式

def encode_texts_with_bert(texts):
    """
    使用 BERT + 平均池化 生成句向量
    :param texts: List[str]，如 ['reasoning path 1', 'reasoning path 2', ...]
    :return: torch.Tensor [N, 768]
    """
    # 批量编码
    encoded = tokenizer_bert(
        texts,
        padding=True,           # 自动 padding 到最长
        truncation=True,        # 超长截断
        max_length=512,         # 最大长度
        return_tensors="pt"     # 返回 PyTorch 张量
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model_bert(**encoded)  # 运行模型
        last_hidden_state = outputs.last_hidden_state  # [B, L, 768]
        attention_mask = encoded['attention_mask']     # [B, L]

        # Mean Pooling: 对 token 取平均，忽略 padding
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # [B, 768]

        # L2 正则化（便于计算 cosine 相似度）
        pooled = F.normalize(pooled, p=2, dim=1)
    
    return pooled