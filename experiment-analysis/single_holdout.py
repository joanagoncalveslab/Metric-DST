import pandas as pd
import numpy as np
import random

def single_holdout(cancer):
    print(cancer)
    df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_"+cancer+".csv", index_col=0)
    df = df[df['seq1']!=0.0]

    gene1 = set(df['gene1'])
    gene2 = set(df['gene2'])
    intersect = gene1.intersection(gene2)
    test_genes = gene2.difference(intersect)

    # print(len(gene1.union(gene2)))
    # print(len(gene2))
    # print(len(intersect))
    # print(len(test_genes))

    train_data = df[~df['gene2'].isin(test_genes)]
    # print(train_data)
    print(len(train_data))
    print(train_data.groupby(['class']).size())

    # print(len(df[(df['gene2'].isin(test_genes))]))
    test_data = df[(df['gene2'].isin(test_genes)) & (df['gene1'].isin(train_data['gene1']))]
    print(len(test_data))
    print(test_data.groupby(['class']).size())
    # print(test_data)
    train_data.to_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/train_seq_128_"+cancer+".csv", index=False)
    test_data.to_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/test_seq_128_"+cancer+".csv", index=False)

def single_holdout_OV(cancer, rng):
    print(cancer)
    df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_"+cancer+".csv", index_col=0)
    df = df[df['seq1']!=0.0]

    genes = list(pd.concat([df['gene1'], df['gene2']]).unique())

    adj_genes = pd.DataFrame(
        index=genes,
        columns=genes,
        dtype=np.int8
    )
    for index, row in df.iterrows():
        adj_genes.at[row['gene1'], row['gene2']] = row['class']
        adj_genes.at[row['gene2'], row['gene1']] = row['class']


    test_genes = set()
    test_positive = 0
    test_negative = 0
    while min(test_positive, test_negative) < 40:
        selected_gene = rng.sample(genes, 1)[0]
        test_genes.add(selected_gene)
        genes.remove(selected_gene)
        new_positives =  adj_genes.loc[selected_gene, genes].fillna(0).values.sum()
        new_samples = adj_genes.loc[selected_gene, genes].notna().values.sum()
        positives_to_remove = adj_genes.loc[selected_gene, test_genes].fillna(0).values.sum()
        samples_to_remove = adj_genes.loc[selected_gene, test_genes].notna().values.sum()
        negatives_to_remove = samples_to_remove - positives_to_remove
        test_positive += new_positives - positives_to_remove
        test_negative += new_samples - new_positives - negatives_to_remove

    positives =  adj_genes.loc[test_genes, genes].fillna(0).values.sum()
    samples = adj_genes.loc[test_genes, genes].notna().values.sum()
    negatives = samples - positives
    print(positives)
    print(negatives)
    train_positives =  adj_genes.loc[genes, genes].fillna(0).values.sum()/2
    train_samples = adj_genes.loc[genes, genes].notna().values.sum()/2
    train_negatives = train_samples - train_positives
    print(train_positives)
    print(train_negatives)
    print(test_genes)
    print(genes)

    train_data = df[(df['gene2'].isin(genes)) & (df['gene1'].isin(genes))]
    print(train_data)
    print(len(train_data))
    print(train_data.groupby(['class']).size())

    test_data = df[((df['gene2'].isin(test_genes)) & (df['gene1'].isin(genes))) | ((df['gene1'].isin(test_genes)) & (df['gene2'].isin(genes)))]
    print(len(test_data))
    print(test_data.groupby(['class']).size())
    print(test_data)
    train_data.to_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/train_seq_128_"+cancer+".csv", index=False)
    test_data.to_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/test_seq_128_"+cancer+".csv", index=False)


# single_holdout("BRCA")
# single_holdout("OV")
# single_holdout("SKCM")
# single_holdout("CESC")
rng = random.Random(0)
single_holdout_OV('OV', rng)