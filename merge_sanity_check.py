import pandas as pd
d = pd.read_csv(r"merged_fod_dataset\dropped.csv"); print(d.groupby(["src","reason"]).size())
m = pd.read_csv(r"merged_fod_dataset\manifest.csv"); print(m.group_size.describe())