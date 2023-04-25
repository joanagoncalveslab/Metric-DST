from time import time
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from collections import Counter

from visualize import visualize_gene, visualize_show

def main(fold):
    df = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-noMiner-contrastive/embeddings_train_fold_{fold}.csv', index_col=0)
    df['genes'] = list(zip(df['gene1'], df['gene2']))
    df['coordinates'] = list(zip(df['dim0'], df['dim1']))
    df.set_index('genes', inplace=True)
    
    # Pairwise similarity matrix
    pairwise = pd.DataFrame(
        squareform(pdist(df[['dim0', 'dim1']], metric='euclidean')),
        columns=df.index,
        index=df.index
        )

    # List with samples that are furthest away from the other positives
    positives = df['label']==1
    negatives = df['label']==0
    furthest_positives:pd.DataFrame = (
        pairwise
        .iloc[positives.values, positives.values]
        .mean(axis=1)
        .sort_values(ascending=False)
    )
    genes_in_question = furthest_positives.head(10).index.values

    embeddings = df['coordinates'].values
    labels = df['label'].values
    test = df.loc[genes_in_question]
    visualize_show([list(x) for x in embeddings], labels, [list(x) for x in test['coordinates'].values], test['label'].values)
    close_pairs = []
    for gene in genes_in_question:
        closest = pairwise.loc[negatives.values, [gene]].idxmin()
        test = df.loc[[*closest.values, gene]]
        close_pairs.append([gene, closest.values[0]])
        # visualize_show([list(x) for x in embeddings], labels, [list(x) for x in test['coordinates'].values], test['label'].values)

    df2 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_BRCA.csv", index_col=0).fillna(0)
    df2['genes'] = list(zip(df2['gene1'], df2['gene2']))
    df2.set_index('genes', inplace=True)
    df2 = df2.reindex(index=df.index.values)

    pairwise2 = pd.DataFrame(
        squareform(pdist(df2.iloc[:, 4:], metric='euclidean')),
        columns=df2.index,
        index=df2.index
        )
    # for v in pairwise2.loc[genes_in_question].iloc[:, negatives.values].min(axis=1).values:
    #     print(v)
    g_list = []
    for [g, c], avg_dis_pos, avg_dis_neg in zip(
            close_pairs, 
            pairwise2.loc[genes_in_question].iloc[:, positives.values].mean(axis=1).values, 
            pairwise2.loc[genes_in_question].iloc[:, negatives.values].mean(axis=1).values
            ):
        feature_vectors = df2.loc[[g, c]].iloc[:, 4:]
        # print(f"{fold}; {g}; {c}; {pdist(feature_vectors, metric='correlation')[0]}; {avg_dis_pos}; {avg_dis_neg}")
        # print(f"{pdist(feature_vectors, metric='euclidean')[0]}; {avg_dis_pos}; {avg_dis_neg}")
        g_list.append(g)
    return g_list
        
def main_with_test(fold):
    df = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-noMiner-contrastive/embeddings_train_fold_{fold}.csv', index_col=0)
    df['genes'] = list(zip(df['gene1'], df['gene2']))
    df['coordinates'] = list(zip(df['dim0'], df['dim1']))
    df['set'] = 'train'
    df.set_index('genes', inplace=True)

    df_test = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-noMiner-contrastive/embeddings_test_fold_{fold}.csv', index_col=0)
    df_test['genes'] = list(zip(df_test['gene1'], df_test['gene2']))
    df_test['coordinates'] = list(zip(df_test['dim0'], df_test['dim1']))
    df_test['set'] = 'test'
    df_test.set_index('genes', inplace=True)

    df_combined = pd.concat([df, df_test])

        # Pairwise similarity matrix
    pairwise = pd.DataFrame(
        squareform(pdist(df_combined[['dim0', 'dim1']], metric='euclidean')),
        columns=df_combined.index,
        index=df_combined.index
        )

    # List with samples that are furthest away from the other positives
    positives = (df_combined['label']==1) & (df_combined['set']=='train')
    negatives = (df_combined['label']==0) & (df_combined['set']=='train')
    positives_test = (df_combined['label']==1) & (df_combined['set']=='test')
    negatives_test = (df_combined['label']==0) & (df_combined['set']=='test')
    furthest_positives:pd.DataFrame = (
        pairwise
        .iloc[positives_test.values, positives.values]
        .mean(axis=1)
        .sort_values(ascending=False)
    )
    genes_in_question = furthest_positives.head(10).index.values

    close_pairs = []
    for gene in genes_in_question:
        closest = pairwise.loc[negatives.values, [gene]].idxmin()
        close_pairs.append([gene, closest.values[0]])

    df2 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_BRCA.csv", index_col=0).fillna(0)
    df2['genes'] = list(zip(df2['gene1'], df2['gene2']))
    df2.set_index('genes', inplace=True)
    df2 = df2.reindex(index=df_combined.index.values)

    pairwise2 = pd.DataFrame(
        squareform(pdist(df2.iloc[:, 4:], metric='euclidean')),
        columns=df2.index,
        index=df2.index
        )
    # for v in pairwise2.loc[genes_in_question].iloc[:, negatives.values].min(axis=1).values:
    #     print(v)
    g_list = []
    c_list = []
    for [g, c], avg_dis_pos, avg_dis_neg in zip(
            close_pairs, 
            pairwise2.loc[genes_in_question].iloc[:, positives.values].mean(axis=1).values, 
            pairwise2.loc[genes_in_question].iloc[:, negatives.values].mean(axis=1).values
            ):
        feature_vectors = df2.loc[[g, c]].iloc[:, 4:]
        # print(f"{fold}; {g}; {c}; {pdist(feature_vectors, metric='correlation')[0]}; {avg_dis_pos}; {avg_dis_neg}")
        # print(f"{pdist(feature_vectors, metric='euclidean')[0]}; {avg_dis_pos}; {avg_dis_neg}")
        g_list.append(g)
        c_list.append(c)
    return g_list, c_list

def investigate_gene(fold, gene):
    df = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-noMiner-contrastive/embeddings_train_fold_{fold}.csv', index_col=0)
    df['genes'] = list(zip(df['gene1'], df['gene2']))
    df['coordinates'] = list(zip(df['dim0'], df['dim1']))
    df['set'] = 'train'
    df.set_index('genes', inplace=True)

    df_test = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-0.01-noMiner-contrastive/embeddings_test_fold_{fold}.csv', index_col=0)
    df_test['genes'] = list(zip(df_test['gene1'], df_test['gene2']))
    df_test['coordinates'] = list(zip(df_test['dim0'], df_test['dim1']))
    df_test['set'] = 'test'
    df_test.set_index('genes', inplace=True)

    visualize_gene(list(zip(df['dim0'], df['dim1'])), df['label'].values, list(zip(df['gene1'], df['gene2'])), gene, '', 0, 0)
    visualize_gene(list(zip(df_test['dim0'], df_test['dim1'])), df_test['label'].values, list(zip(df_test['gene1'], df_test['gene2'])), gene, '', 0, 0)


if __name__ == '__main__':
    g_list = []
    c_list = []
    for fold in range(5):
        g_list.extend(main(fold))
    print(len(g_list))
    print(len(set(g_list)))
    print(Counter(g_list))

    # ---------------------------
    # for fold in range(5):
    #     g, c = main_with_test(fold)
    #     g_list.extend(g)
    #     c_list.extend(c)
    # count = 0
    # for g, c in zip(g_list, c_list):
    #     if any(x in c for x in g):
    #         count+=1
    # print(count)
    # flat_list = [item for sublist in g_list for item in sublist]
    # print(Counter(flat_list))


    # investigate_gene(0, 'PARP1')
