import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-convergence-BRCA-knn-10-conf-85/performance.csv", index_col=0).reset_index()
# df = df[df['epoch']==199]
df = df.loc[df.groupby('fold', sort=False)['epoch'].idxmax()]
df = df[["validation_loss", "train_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
# print(df.describe())
print(df.mean().to_frame().T)
print(df.std().to_frame().T)

# ---------------------------------

# with open("C:/Users/mathi/git/msc-thesis-2122-mathijs-de-wolf/experiment-6/distances.txt") as file:
#     distances = pd.DataFrame(
#         data=[[float(y) for y in x.split(',')] for x in file.readlines()],
#         columns=['order', 'distances']
#     )
#     # print(distances)
#     sn.boxplot(distances, x="order", y="distances")
#     plt.show()

# ------------------------------------

