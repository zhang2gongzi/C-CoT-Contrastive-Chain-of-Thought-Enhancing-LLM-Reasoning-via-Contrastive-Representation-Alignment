import pandas as pd
df = pd.read_parquet("/home2/zzl/C-CoT/database/commonsenseQA/test-00000-of-00001.parquet")
print("数据集列名：", df.columns.tolist())  # 查看所有列，找到标准答案列（如answerKey）
print("示例标准答案：", df.iloc[0]["answerKey"])  # 查看第一个样本的标准答案