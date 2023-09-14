import platform, gc

from matplotlib.colors import to_rgba
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
from umap import UMAP
import argparse

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run UMAP projections"
    )

    parser.add_argument("--save-umap", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--load-umap", type=str, default=None)
    parser.add_argument("--load-embeddings", type=str, default=None)
    parser.add_argument("--umap-neighbours", type=int, default=15)
    parser.add_argument("-C", "--cancer", choices=["BRCA", "CESC", "SKCM", "OV"], default="BRCA")
    parser.add_argument("--sample", type=int, default=None)

    return parser

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    sns.set()

    if args.load_embeddings is None:
        df_labeled = pd.read_csv(f"{webDriveFolder}train_seq_128_{args.cancer}.csv", index_col=0)
        df_labeled = df_labeled[df_labeled.cancer == args.cancer]
        df_labeled = df_labeled[(df_labeled['seq1']!=-1.0) & (df_labeled['seq1']!=0.0)]

        df_unlabeled = pd.read_csv(f"{webDriveFolder}unknown_repair_cancer_{args.cancer}_seq_128.csv")
        df_unlabeled = df_unlabeled[df_unlabeled.cancer == args.cancer]
        df_unlabeled = df_unlabeled[(df_unlabeled['seq1']!=-1.0) & (df_unlabeled['seq1']!=0.0)]

        if args.sample is not None:
            sample_n = len(df_unlabeled) if args.sample == 1 else args.sample*100
            df_unlabeled = df_unlabeled.sample(sample_n, replace=False, random_state=sample_n).reset_index(drop=True)
        
        df = pd.concat([df_unlabeled, df_labeled])
        X = df.iloc[:, 4:]
        y = df["class"]
        
        gc.collect()

        if args.load_umap is None:
            umap = UMAP(n_components=2, init='random', random_state=0, n_neighbors=args.umap_neighbours, n_jobs=-1)
            umap.fit(X)

            if args.save_umap:
                import joblib
                filename = f'{outputFolder}umaps/umap_state_{args.cancer}_{args.sample}_{args.umap_neighbours}.sav'
                joblib.dump(umap, filename)
        else:
            import joblib
            umap = joblib.load(args.load_umap)

        umap_result = umap.transform(X)
        umap_result = pd.DataFrame({'umap_1': umap_result[:,0], 'umap_2': umap_result[:,1], 'y': y, 'gene1': df["gene1"], 'gene2': df["gene2"]})

        if args.save_embeddings:
            filename = f'{outputFolder}umaps/umap_embedding_{args.cancer}_{args.sample}_{args.umap_neighbours}.csv'
            umap_result.to_csv(filename)
    else:
        umap_result = pd.read_csv(args.load_embeddings, index_col=0)

    palette = {
        0: to_rgba('tab:blue', 1),
        1: to_rgba('tab:orange', 1),
        -1: to_rgba('tab:gray', 0.5)
    }


    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('UMAP projections with different genes highlighted')

    for i, gene in enumerate(['BRCA1', 'PARP1', 'PTEN', 'BRCA2', 'TP53']):
        umap_result['label'] = umap_result['y']
        umap_result.loc[~((umap_result['gene1'] == gene) | (umap_result['gene2'] == gene)), 'label'] = -1
        
        ax:matplotlib.axes.Axes = sns.scatterplot(
            x="umap_1", y="umap_2",
            hue="label",
            hue_order=[-1, 0, 1],
            palette=palette,
            data=umap_result,
            legend="full",
            edgecolor=None,
            size='label',
            sizes=(1,4),
            ax=axs[i%2, i>>1]
        )

        axs[i%2, i>>1].set(xlabel=None, ylabel=None)
        axs[i%2, i>>1].set_title(gene)
        h, l = axs[i%2, i>>1].get_legend_handles_labels()
        axs[i%2, i>>1].legend(h, [f'no {gene}', f'neg {gene}', f'pos {gene}'], loc='upper right')

    ax:matplotlib.axes.Axes = sns.scatterplot(
        x="umap_1", y="umap_2",
        hue="y",
        hue_order=[-1, 0, 1],
        palette=palette,
        data=umap_result,
        legend="full",
        edgecolor=None,
        s=1,
        ax=axs[1, 2],
    )

    axs[1, 2].set(xlabel=None, ylabel=None)
    axs[1, 2].set_title('positive - negative')
    h, l = axs[1, 2].get_legend_handles_labels()
    axs[1, 2].legend(h, ['unlabeled', 'negative', 'positive'], loc='upper right')

    # plt.savefig(f'{outputFolder}umaps/umap_{args.cancer}_{args.sample}_{args.umap_neighbours}.png')
    plt.show()
