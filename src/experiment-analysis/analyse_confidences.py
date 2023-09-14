import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

cancer = 'OV'
conf = 85
pseudolabel = 50
n = 'diversity'
df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/"+n+"_confidences.csv", index_col=0).reset_index(drop=True)

sns.boxplot(data=df, y='confidences', x='loop')
sns.stripplot(data=df, y='confidences', x='loop', color='black', alpha=0.3)

plt.figure()
sns.violinplot(data=df[df['fold']==0], x='loop', y='confidences', cut=0, bw=.2)

plt.figure()
sns.histplot(data=df[(df['fold']==0) & (df['loop']==1)], x='confidences', binwidth=0.1, binrange=(-0.05,1.05))

plt.show()


