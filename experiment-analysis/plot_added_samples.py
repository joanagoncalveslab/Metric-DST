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
        -1: (218/255, 218/255, 218/255, 0.5)
    }

    sns.set_theme(style='white')

    # fig, axs = plt.subplots(2, 2, figsize=(10, 11))
    fig = plt.figure(figsize=(10, 10))
    fig.set_dpi(300)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    axs = [ax1, ax2, ax3, ax4]
    # fig.suptitle('UMAP projections with added genes highlighted in self training')

    umap_result = pd.read_csv(args.load_embeddings, index_col=0)

    vc = umap_result['y'].value_counts()

    ax:matplotlib.axes.Axes = sns.scatterplot(
            x="umap_1", y="umap_2",
            hue="y",
            hue_order=[-1, 0, 1],
            palette=palette_other_genes,
            data=umap_result,
            legend="full",
            edgecolor=None,
            s=1,
            ax=ax1,
        )
    ax1.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])

    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h, [f'Unlabeled ({vc[-1]:.0f})', f'Negatives ({vc[0]:.0f})', f'Positives ({vc[1]:.0f})'], loc='upper right')
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

        ax:matplotlib.axes.Axes = sns.scatterplot(
            x="umap_1", y="umap_2",
            hue="y",
            hue_order=[-1],
            palette=palette_other_genes,
            data=other_samples,
            legend="full",
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
            legend="full",
            edgecolor=None,
            s=1,
            ax=axs[i],
        )

        axs[i].axis('off')
        h, l = axs[i].get_legend_handles_labels()
        # axs.legend(h, ['unlabeled samples', 'negative training samples', 'positive training samples', 'added as negative', 'added as positive'], loc='upper right')
        # axs.legend(h, ['unlabeled samples', 'negative training samples', 'positive training samples'], loc='upper right')
        axs[i].legend(h, [f'Unlabeled ({len(other_samples):.0f})', f'Pseudonegative ({len(labeled_as) - sum(labeled_as):.0f})', f'Pseudopositive ({sum(labeled_as):.0f})'], loc='upper right')
        axs[i].set_title(replace_name(name))

    fig.subplots_adjust(wspace=0, hspace=0.1)

    plt.gca().set_aspect('equal')

    plt.savefig('W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_square_BRCA_no_title.pdf', dpi=300, format='pdf', bbox_inches='tight')
    # fig.suptitle('UMAP projections with added genes highlighted in self training')
    # plt.savefig('W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_training_BRCA_balanced_semi_supervised_with_title.pdf', dpi=300, format='pdf', bbox_inches='tight')

    plt.savefig(f'W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_square_BRCA_no_title', dpi=300, bbox_inches='tight')
    # fig.suptitle('UMAP projections with added genes highlighted in self training')
    # plt.savefig(f'W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/umap_addedGenes_training_BRCA_{name}_with_title', dpi=300, bbox_inches='tight')

    # plt.show()
    # plt.close()

