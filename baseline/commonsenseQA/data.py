import pandas as pd

path = "/home2/zzl/C-CoT/database/commonsenseQA/train-00000-of-00001.parquet"
df = pd.read_parquet(path)

# 打印前 5 行
print(df.head(5))

# 打印列名
print(df.columns.tolist())
