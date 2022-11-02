import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-8-divergence-knn10-conf0.9/test_performance.csv", index_col=0)
# df = df[df['epoch']==199]
df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
print(df.describe())