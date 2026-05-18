import pandas as pd

path = "/home/xzhe162/wh_workspace/casual/processed_data/suning/item_multimodal_scalars/item_multimodal_scalars_full_3.parquet"

df = pd.read_parquet(path)

print(df.sample(5))


# python /home/xzhe162/wh_workspace/casual/data_preprocessing/suning_preprocessing/inspect_llm.py