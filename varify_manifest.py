import json
import os
import pandas as pd

manifest_path = "/p/fastdata/mmlaion/lmdata_2/openwebtext2_tokenized/manifest_train.jsonl"
base_path = "/p/fastdata/mmlaion/lmdata_2/openwebtext2_tokenized/"

def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)

def create_dataframe(manifest_path: str, base_path: str) -> pd.DataFrame:
    data = []
    
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            shard_num = entry["shard"]
            num_sequences = entry["num_sequences"]
            file_path = os.path.join(base_path, f"{shard_num}.tar")
            file_size = get_file_size(file_path) if os.path.exists(file_path) else None
            
            data.append({
                "shard_num": shard_num,
                "num_sequences": num_sequences,
                "file_size": file_size,
                "ratio":  file_size / num_sequences if file_size else None
            })
    df = pd.DataFrame(data)
    df['outliers'] = (df.ratio > (df.ratio.mean() + 2 * df.ratio.std())) | (df.ratio < (df.ratio.mean() - 2 * df.ratio.std()))
    return df

df = create_dataframe(manifest_path, base_path)
df.to_csv("sizes_owt2.csv", index=False)
print(df.ratio.mean(), df.ratio.std())
print(df.outliers.sum())
print(df.query("outliers == True"))