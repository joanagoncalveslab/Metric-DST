import sys
sys.path.append('C:/Users/mathi/git/msc-thesis-2122-mathijs-de-wolf')

import platform

from matplotlib.colors import to_rgba
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import argparse
import pickle
import joblib
from scipy.spatial.distance import cdist, pdist


from network import Network
import random
import torch
import torch.utils.data
from metric_learning_utils import create_customDataset

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run UMAP projections"
    )

    parser.add_argument("--load-embeddings", type=str, default=None)
    parser.add_argument("-C", "--cancer", choices=["BRCA", "CESC", "SKCM", "OV"], default="BRCA")

    return parser

def get_training_data():
    dataPath = webDriveFolder + f"train_seq_128_BRCA.csv"
    dataset = pd.read_csv(dataPath, index_col=0).fillna(0)
    dataset = dataset[(dataset['seq1']!=-1.0) & (dataset['seq1']!=0.0)]

    # Undersample dataset
    setup_seed(0)
    idx_0 = dataset[dataset['class'] == 0]
    idx_1 = dataset[dataset['class'] == 1]
    if len(idx_0) < len(idx_1):
        idx_1 = idx_1.sample(len(idx_0))
    if len(idx_0) > len(idx_1):
        idx_0 = idx_0.sample(len(idx_1))
    # Split dataset in validation and train set
    partion = round((len(idx_0)+len(idx_1))/10)
    train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
    validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
    # Shuffle datasets, otherwise the first half is negative and the second half is positive
    setup_seed(0)
    train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)

    train_dataset = create_customDataset(train_dataset)

    train_sampler = torch.utils.data.SequentialSampler(train_dataset)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), sampler=train_sampler)
    return train_data_loader

def get_added_samples_data(name):
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

    unlabeledDatapath = webDriveFolder + f"unknown_repair_cancer_BRCA_seq_128.csv"
    unlabeled_dataset = pd.read_csv(unlabeledDatapath)
    unlabeled_dataset = unlabeled_dataset[unlabeled_dataset['cancer'] == 'BRCA']
    unlabeled_dataset = unlabeled_dataset[(unlabeled_dataset['seq1']!=-1.0) & (unlabeled_dataset['seq1']!=0.0)]

    assert(unlabeled_dataset.merge(added_genes, how='inner', left_on=['gene1', 'gene2'], right_on=['gene2', 'gene1']).empty)
    added_samples = unlabeled_dataset.merge(added_genes, how='inner', on=['gene1', 'gene2'])

    added_samples['class'] = added_samples['pseudolabel']
    added_samples.drop('pseudolabel', axis=1, inplace=True)

    dataset = create_customDataset(added_samples)

    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), sampler=sampler)
    return data_loader

def replace_name(name):
    if name == 'diversity':
        return 'Diverse BST'
    elif name == 'true_semi_supervised':
        return 'Self Training'
    elif name == 'balanced_semi_supervised':
        return 'Balanced ST'

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    sns.set_theme(style='white')
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman', 'font.size':10.5})

    distances = pd.DataFrame()
    
    for name in ['true_semi_supervised', 'balanced_semi_supervised', 'diversity']:
    # for name in ['diversity']:

        network:Network = joblib.load(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-pipeline-what-settings/experiment-divergence-pipeline-BRCA-knn-10-conf-90-pseudolabels-20/{name}_final_embedding_fold_0.sav")
        
        train_data_loader = get_training_data()
        train_embeddings, train_labels, train_genes = network.run(train_data_loader)

        train_data = pd.DataFrame(train_embeddings).astype("float")
        train_data['y'] = train_labels

        added_genes_loader = get_added_samples_data(name)
        added_genes_embeddings, added_genes_labels, added_genes = network.run(added_genes_loader)

        added_genes = pd.DataFrame(added_genes_embeddings).astype("float")
        added_genes['y'] = added_genes_labels

        added_genes_neg = added_genes[added_genes['y'] == 0]
        pairwise_neg = pdist(added_genes_neg[[0, 1]], metric='euclidean')
        pairwise_neg = pd.DataFrame(pairwise_neg, columns=['Distance'])
        pairwise_neg['Method'] = replace_name(name)
        pairwise_neg['class'] = 'Not synthetic lethal'
        distances = pd.concat([distances, pairwise_neg])

        added_genes_pos = added_genes[added_genes['y'] == 1]
        pairwise_pos = pdist(added_genes_pos[[0, 1]], metric='euclidean')
        pairwise_pos = pd.DataFrame(pairwise_pos, columns=['Distance'])
        pairwise_pos['Method'] = replace_name(name)
        pairwise_pos['class'] = 'Synthetic lethal'
        distances = pd.concat([distances, pairwise_pos])

        # palette_added_genes = {
        #     0: to_rgba('red', 1),
        #     1: to_rgba('lime', 1),
        #     -1: to_rgba('tab:gray', 0.5)
        # }

        # palette_other_genes = {
        #     0: to_rgba('tab:blue', 1),
        #     1: to_rgba('tab:orange', 1),
        #     -1: to_rgba('tab:gray', 0.5)
        # }

        palette_added_genes = {
            0: (33/255, 102/255, 172/255, 1),
            1: (178/255, 24/255, 43/255, 1)
        }

        palette_other_genes = {
            0: (103/255, 169/255, 207/255, 1),
            1: (239/255, 138/255, 98/255, 1),
            -1: (218/255, 218/255, 218/255, 0.5)
        }

        # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        # # fig.suptitle('Final embedding with added genes highlighted in self training')

        # ax:matplotlib.axes.Axes = sns.scatterplot(
        #     x=0, y=1,
        #     hue="y",
        #     hue_order=[0, 1],
        #     palette=palette_other_genes,
        #     data=train_data,
        #     legend="full",
        #     edgecolor=None,
        #     s=5,
        #     ax=axs,
        # )

        # ax:matplotlib.axes.Axes = sns.scatterplot(
        #     x=0, y=1,
        #     hue="y",
        #     hue_order=[0, 1],
        #     palette=palette_added_genes,
        #     data=added_genes,
        #     legend="full",
        #     edgecolor=None,
        #     s=5,
        #     ax=axs,
        # )

        # axs.set(xlabel=None, ylabel=None)
        # h, l = axs.get_legend_handles_labels()
        # axs.legend(h, ['negative training samples', 'positive training samples', 'added as negative', 'added as positive'], loc='upper right')
        # axs.legend(h, ['negative training samples', 'positive training samples'], loc='upper right')

        # plt.savefig(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/figures/embedding_addedGenes_BRCA_{name}_no_title', dpi=300, bbox_inches='tight')

        # plt.show()
        # plt.close()

    # fig = plt.figure(figsize=(3.255, 2.441))s
    fig = plt.figure(figsize=(6.4, 4,8))

    ax:plt.Axes = sns.boxplot(distances, x='class', y='Distance', hue='Method')
    ax.set(xlabel=None)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='')

    # plt.savefig('W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/embedding_addedGenes_boxplot_distances_BRCA_no_title.pdf', dpi=300, format='pdf', bbox_inches='tight')
    # plt.savefig('W:/staff-umbrella\JGMasters/2122-mathijs-de-wolf/figures/embedding_addedGenes_boxplot_distances_BRCA_no_title', dpi=300, bbox_inches='tight')
    plt.show()