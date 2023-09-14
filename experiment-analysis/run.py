from matplotlib.colors import to_rgba
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# def f(cancer, ax):
#     df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_"+cancer+".csv", index_col=0)
#     df = df[df.cancer == cancer]
#     df = df[(df['seq1']!=-1.0) & (df['seq1']!=0.0)]
#     X = df.iloc[:, 4:]
#     y = df["class"]

#     n_components = 2
#     umap = UMAP(n_components=n_components, init='random', random_state=0)
#     umap_result = umap.fit_transform(X)

#     palette = {
#         0: to_rgba('tab:blue', 0.3),
#         1: to_rgba('tab:orange', 0.3)
#     }

#     umap_result = pd.DataFrame({'umap_1': umap_result[:,0], 'umap_2': umap_result[:,1], 'y': y})
#     sns.scatterplot(
#         x="umap_1", y="umap_2",
#         hue="y",
#         palette=palette,
#         data=umap_result,
#         legend="full",
#         ax=ax
#     )
#     ax.set_title(cancer)

# fig, ax = plt.subplots(2, 2, figsize=(8,8))
# fig.suptitle("UMAP projections from different cancer types")

# f("BRCA", ax[0, 0])
# f("OV", ax[0, 1])
# f("CESC", ax[1, 0])
# f("SKCM", ax[1, 1])
# plt.show()
