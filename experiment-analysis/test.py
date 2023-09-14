import random
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.utils.data
import customDataset
import pandas as pd
from scipy.spatial.distance import squareform, cdist

# --------------- Fix Randomness ------------------------------
# def setup_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True

# def undersample(idx, labels):
#     idx_0 = [id for id in idx if labels[id]==0]
#     idx_1 = [id for id in idx if labels[id]==1]
#     if len(idx_0) < len(idx_1):
#         idx_1 = np.random.choice(idx_1, len(idx_0), replace=False)
#     if len(idx_0) > len(idx_1):
#         idx_0 = np.random.choice(idx_0, len(idx_1), replace=False)
#     return np.concatenate([idx_0, idx_1])

# setup_seed(42)

# data = np.arange(100)
# labels = np.concatenate((np.zeros(25), np.ones(75)))
# splits = StratifiedKFold(5, shuffle=True)

# dataset = customDataset.CustomDataset(torch.Tensor(data), torch.Tensor(labels))

# g = torch.Generator()
# g = g.manual_seed(42)

# for fold, (train_idx, val_idx) in enumerate(splits.split(data, labels)):
#     train_idx = undersample(train_idx, labels)
#     val_idx = undersample(val_idx, labels)
#     if fold == 0:
#         print(len(train_idx)/2)
#         print(sum([labels[id] for id in train_idx]))
#         print(len(val_idx)/2)
#         print(sum([labels[id] for id in val_idx]))
#     #     train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=torch.utils.data.RandomSampler(train_idx, generator=g))
#     #     for batch_idx, (data, labels) in enumerate(train_loader):
#     #         print(batch_idx, data)
#     #     for batch_idx, (data, labels) in enumerate(train_loader):
#     #         print(batch_idx, data)

# -------------------------------------------------------------

fold=0
df = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-noMiner-contrastive/embeddings_train_fold_{fold}.csv', index_col=0)
df['training_genes'] = list(zip(df['gene1'], df['gene2']))
df['coordinates'] = list(zip(df['dim0'], df['dim1']))
df['set'] = 'train'
df.set_index('training_genes', inplace=True)

df_test = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-noMiner-contrastive/embeddings_test_fold_{fold}.csv', index_col=0)
df_test['test_genes'] = list(zip(df_test['gene1'], df_test['gene2']))
df_test['coordinates'] = list(zip(df_test['dim0'], df_test['dim1']))
df_test['set'] = 'test'
df_test.set_index('test_genes', inplace=True)

    # Pairwise similarity matrix
pairwise = pd.DataFrame(
    cdist(df_test[['dim0', 'dim1']], df[['dim0', 'dim1']], metric='euclidean'),
    columns=df.index,
    index=df_test.index
    )

# print(pairwise.iloc[[1]].sort_values(by=pairwise.index[1], axis=1))

idx = np.argsort(pairwise.values, 1)[:, :5]

knn_distances = [[pairwise.iloc[i, j] for j in row] for i, row in enumerate(idx)]
# print(knn_distances)
knn_labels = [[df['label'].iloc[element] for element in row] for row in idx]
print(knn_labels)

knn_distances = np.array(knn_distances)
knn_labels = np.array(knn_labels)
weighted_knn_labels = knn_labels + ((1 - 2 * knn_labels) * knn_distances)
print(weighted_knn_labels)
print(np.equal(weighted_knn_labels.mean(1).round(), df_test['label'].values).astype(float).mean())

# print(res)