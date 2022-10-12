import pandas as pd

df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128.csv")
# print(df["class"].value_counts().to_frame())
print(df.groupby(['cancer', 'class']).size().unstack(fill_value=0))