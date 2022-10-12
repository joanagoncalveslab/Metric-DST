import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-0/performance.csv", index_col=0)
df = df[df['epoch']==199]
df = df[["train_loss", "test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
print(df.describe())