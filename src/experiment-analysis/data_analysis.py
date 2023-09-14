import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_BRCA.csv", index_col=0)
df_unlabeled = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/unknown_repair_cancer_BRCA_seq_128.csv")
# for gene in pd.concat([df['gene1'], df['gene2']]).value_counts().index.values:
#     labels = df[(df['gene1']==gene) | (df['gene2']==gene)]['class'].values
#     print(f'{gene}, {len(labels)}, {len(labels)/1963}, {labels.sum()}, {len(labels) - labels.sum()}')
# gene = 'BRCA2'
# df = df[(df['gene1']!=gene) & (df['gene2']!=gene)]
# print(df)


df = df[df.cancer == "BRCA"]
df = df[(df['seq1']!=-1.0) & (df['seq1']!=0.0)]

df_unlabeled = df_unlabeled[df_unlabeled.cancer == "BRCA"]
df_unlabeled = df_unlabeled[(df_unlabeled['seq1']!=-1.0) & (df_unlabeled['seq1']!=0.0)]

X = pd.concat([df.iloc[:, 4:], df_unlabeled.iloc[:, 4:]])
y = pd.concat([df["class"], df_unlabeled["class"]])
# X = df.iloc[:, 4:]
# y = df["class"]
print("data")

pca = PCA(3)
pca_result = pca.fit_transform(X)
print("pca")

n_components = 2
# tsne = TSNE(n_components)
# tsne_result = tsne.fit_transform(X)
# print("tsne")
# print(tsne_result.shape)

umap = UMAP(n_components=n_components, init='random', random_state=0)
umap_result = umap.fit_transform(X)
print("plot")

# result_df = pd.DataFrame({'umap_1': umap_result[:, 0], 'umap_2': umap_result[:, 1], 'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'pca_1': pca_result[:,0], 'pca_2': pca_result[:,1], 'pca_3': pca_result[:,2], 'y': y})


fig, ax = plt.subplots(2, 3)
# sns.scatterplot(x='tsne_1', y='tsne_2', hue='y', data=result_df, ax=ax,s=120)
# lim = (tsne_result.min()-5, tsne_result.max()+5)
# ax.set_xlim(lim)
# ax.set_ylim(lim)
# ax.set_aspect('equal')
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# plt.show()

# plt.figure(figsize=(12,4))
pca_result = pd.DataFrame({'pca_1': pca_result[:,0], 'pca_2': pca_result[:,1], 'y': y})
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca_1", y="pca_2",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=pca_result,
    legend="full",
    alpha=0.3,
    ax=ax1
)
# tsne_result = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'y': y})
# ax2 = plt.subplot(1, 3, 2)
# sns.scatterplot(
#     x="tsne_1", y="tsne_2",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=tsne_result,
#     legend="full",
#     alpha=0.3,
#     ax=ax2
# )
umap_result = pd.DataFrame({'umap_1': umap_result[:,0], 'umap_2': umap_result[:,1], 'y': y})
ax3 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="umap_1", y="umap_2",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=umap_result,
    legend="full",
    alpha=0.3,
    ax=ax3
)
plt.show()

# ------------------------------------

# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=result_df.loc[:,:]["pca_1"], 
#     ys=result_df.loc[:,:]["pca_2"], 
#     zs=result_df.loc[:,:]["pca_3"], 
#     c=result_df.loc[:,:]["y"], 
#     cmap='Dark2'
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
# plt.show()