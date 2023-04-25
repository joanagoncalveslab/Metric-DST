import pandas as pd
import gc

# df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128.csv")
# # print(df["class"].value_counts().to_frame())
# print(df.groupby(['cancer', 'class']).size().unstack(fill_value=0))

# for c in ['BRCA', 'OV', 'SKCM', 'CESC']:
#     print(c)
#     df_unlabeled = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/unknown_repair_cancer_{c}_seq_128.csv")  
#     gc.collect()
#     print(len(df_unlabeled))
#     df_unlabeled = df_unlabeled[df_unlabeled.cancer == c]
#     print(len(df_unlabeled))
#     df_unlabeled = df_unlabeled[df_unlabeled['seq1']!=-1.0]
#     print(len(df_unlabeled))
#     df_unlabeled = df_unlabeled[df_unlabeled['seq1']!=0.0]
#     print(len(df_unlabeled))

# for c in ['BRCA', 'OV', 'SKCM', 'CESC']:
#     df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_"+c+".csv")
#     df = df[df['seq1']!=0.0]
#     # vc = pd.concat([df['gene1'], df['gene2']]).value_counts()
#     # vc2 = vc.value_counts().sort_index(ascending=False)
#     print(len(set(df['gene1']).union(df['gene2'])))


# Print average precision to number the points in graph
c = 'SKCM'
for n in ['supervised', 'semisupervised', 'diversity']:
    df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-double-holdout/experiment-double-holdout-divergence-pipeline-{c}-knn-10-conf-90-pseudolabels-50/{n}_test_performance.csv")
    print(df['average_precision'])
