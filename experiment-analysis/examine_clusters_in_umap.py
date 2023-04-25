import platform

from matplotlib.colors import to_rgba, Normalize
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import argparse
import pickle

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

    parser.add_argument("--load-embeddings", type=str, default=None)
    parser.add_argument("-C", "--cancer", choices=["BRCA", "CESC", "SKCM", "OV"], default="BRCA")

    return parser

def replace_name(name):
    if name == 'diversity':
        return 'Diversity'
    elif name == 'true_semi_supervised':
        return 'Self training'
    elif name == 'balanced_semi_supervised':
        return 'Balanced self training'

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    palette_added_genes = {
        0: (33/255, 102/255, 172/255, 1),
        1: (178/255, 24/255, 43/255, 1)
    }

    palette_other_genes = {
        0: (103/255, 169/255, 207/255, 1),
        1: (239/255, 138/255, 98/255, 1),
        -1: (218/255, 218/255, 218/255, 1)
    }

    sns.set_theme(style='white')

    # fig, axs = plt.subplots(2, 2, figsize=(10, 11))
    fig = plt.figure(figsize=(10, 10))
    fig.set_dpi(300)
    plt.gca().set_aspect('equal')
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    axs = [ax1, ax2, ax3, ax4]

    umap_result = pd.read_csv(args.load_embeddings, index_col=0)

    vc = umap_result['y'].value_counts()

    umap_result = umap_result[(umap_result['umap_1'].between(13.3, 14.05)) & (umap_result['umap_2'].between(0.55, 1.15))]
    print(umap_result['y'].value_counts())
    print(umap_result[['gene1', 'gene2']].stack().reset_index(drop=True).value_counts())
    # exit()

    ax:matplotlib.axes.Axes = sns.scatterplot(
            x="umap_1", y="umap_2",
            hue="y",
            hue_order=[-1, 0, 1],
            palette=palette_other_genes,
            data=umap_result,
            legend=False,
            edgecolor=None,
            s=1,
            ax=ax1,
        )
    ax1.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    ax1.set_title("Training")

    umap_result = umap_result[umap_result['y'] == -1]
    
    for i, name in enumerate(['true_semi_supervised', 'balanced_semi_supervised', 'diversity'], 1):
    
        added_genes = list()
        labeled_as = list()
        with open(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-pipeline-what-settings/experiment-divergence-pipeline-BRCA-knn-10-conf-90-pseudolabels-20/{name}_iteration_stats_fold_0.pickle", 'rb') as input_file:
            iteration_stats:dict = pickle.load(input_file)
            for key in list(iteration_stats.keys())[1:-1]:
                added_genes.extend(iteration_stats[key]['genes'])
                labeled_as.extend(iteration_stats[key]['pseudolabels'])
        added_genes = np.array(added_genes)
        added_genes = pd.DataFrame(
            added_genes,
            columns=['gene1', 'gene2']
        )
        added_genes['pseudolabel'] = labeled_as

        assert(umap_result.merge(added_genes, how='inner', left_on=['gene1', 'gene2'], right_on=['gene2', 'gene1']).empty)
        added_samples = umap_result.merge(added_genes, how='inner', on=['gene1', 'gene2'])
        other_samples = umap_result.merge(added_genes, how='outer', on=['gene1', 'gene2'], indicator=True).query('_merge=="left_only"')

        other_samples['y'] = -1

        print(name)
        print(added_samples['pseudolabel'].value_counts())

        ax:matplotlib.axes.Axes = sns.scatterplot(
            x="umap_1", y="umap_2",
            hue="y",
            hue_order=[-1],
            palette=palette_other_genes,
            data=other_samples,
            legend=False,
            edgecolor=None,
            s=1,
            ax=axs[i],
        )

        ax:matplotlib.axes.Axes = sns.scatterplot(
            x="umap_1", y="umap_2",
            hue="pseudolabel",
            hue_order=[0, 1],
            palette=palette_added_genes,
            data=added_samples,
            legend=False,
            edgecolor=None,
            s=1,
            ax=axs[i],
        )

        # axs[i].axis('off')
        axs[i].set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        axs[i].set_title(replace_name(name))

    fig.subplots_adjust(wspace=0)

    plt.gca().set_aspect('equal')

    # plt.savefig('W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_square_BRCA_no_title.pdf', dpi=300, format='pdf', bbox_inches='tight')
    # fig.suptitle('UMAP projections with added genes highlighted in self training')
    # plt.savefig('W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_training_BRCA_balanced_semi_supervised_with_title.pdf', dpi=300, format='pdf', bbox_inches='tight')

    # plt.savefig(f'W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_square_BRCA_no_title', dpi=300, bbox_inches='tight')
    # fig.suptitle('UMAP projections with added genes highlighted in self training')
    # plt.savefig(f'W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_training_BRCA_{name}_with_title', dpi=300, bbox_inches='tight')

    # plt.show()
    # plt.close()

