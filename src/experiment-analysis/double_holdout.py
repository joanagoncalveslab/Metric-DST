import pandas as pd
import numpy as np
import random
import tqdm
import time

def double_holdout(cancer, fold, rng:random.Random):
    print(f"fold: {fold}")
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
    
    train_size = 0
    test_size = 0
    train_neg_size = 1
    test_neg_size = 1
    training_genes = set()
    testing_genes = set()
    i=0
    # pbar = tqdm.tqdm(total=len(genes))
    while(len(genes) > 0 and i < 15000):
        if((train_size/max(test_size, 1) <= 4) and len(genes) > 0):
            # selected_gene = random.sample(genes, 1)[0]
            selected_gene = rng.sample(genes, 1)[0]
            train_size += adj_genes.loc[selected_gene, training_genes].fillna(0).values.sum()
            training_genes.add(selected_gene)
            genes.remove(selected_gene)
            # train_neg_size = adj_genes.loc[training_genes, training_genes].notna().values.sum()/2 - train_size
            # train_size = adj_genes.loc[training_genes, training_genes].notna().sum()
            # pbar.update()
        if((train_size/max(test_size, 1) > 4) and len(genes) > 0):
            # selected_gene = random.sample(genes, 1)[0]
            selected_gene = rng.sample(genes, 1)[0]
            test_size += adj_genes.loc[selected_gene, testing_genes].fillna(0).values.sum()
            testing_genes.add(selected_gene)
            genes.remove(selected_gene)
            # test_neg_size = adj_genes.loc[testing_genes, testing_genes].notna().values.sum()/2 - test_size
            # test_size = adj_genes.loc[testing_genes, testing_genes].notna().sum()
            # pbar.update()
        i+=1
    # pbar.close()
    # print(f"\titerations needed: {i}")
    # print(f"\tnumber of test genes: {len(testing_genes)}")
    # print(f"\tnumber of training genes: {len(training_genes)}")
    # print(f"\tnumber of test positives: {test_size}")
    # print(f"\tnumber of train positives: {train_size}")
    # print(f"\tnumber of test samples: {adj_genes.loc[testing_genes, testing_genes].notna().values.sum()/2}")
    # print(f"\tnumber of train samples: {adj_genes.loc[training_genes, training_genes].notna().values.sum()/2}")
    # print(f"\tnumber of test negatives: {adj_genes.loc[testing_genes, testing_genes].notna().values.sum()/2 - test_size}")
    # print(f"\tnumber of train negatives: {adj_genes.loc[training_genes, training_genes].notna().values.sum()/2 - train_size}")

    train_set = df[(df['gene1'].isin(training_genes)) & (df['gene2'].isin(training_genes))]
    test_set = df[(df['gene1'].isin(testing_genes)) & (df['gene2'].isin(testing_genes))]

    # print(train_set.groupby(['class']).size())
    # print(test_set.groupby(['class']).size())

    result = pd.concat([train_set.groupby(['class']).size(), test_set.groupby(['class']).size()], axis=1, keys=['train set', 'test set']).reset_index(drop=True).T

    criteria = {
        'BRCA': {
            'train set': 280,
            'test set': 100
        },
        'OV': {
            'train set': 85,
            'test set': 20
        },
        'CESC': {
            'train set': 55,
            'test set': 12
        },
        'LUAD': {
            'train set': 200,
            'test set': 50
        }
    }

    result['TOTAL'] = result.sum(axis=1)
    result.loc['TOTAL'] = result.sum()
    result.index.name = f'fold {fold}'
    result.to_clipboard()
    print('ready')
    # print(pd.concat([train_set.groupby(['class']).size(), test_set.groupby(['class']).size()], axis=1).reset_index().T)

    # if result.min(axis=1).loc['train set'] >= criteria[cancer]['train set'] and result.min(axis=1).loc['test set'] >= criteria[cancer]['test set']:
    #     train_set.to_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/double_holdout/train_seq_128_{cancer}_{fold}.csv", index=False)
    #     test_set.to_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/double_holdout/test_seq_128_{cancer}_{fold}.csv", index=False)
    #     return True
    # return False


def double_holdout_SKCM(fold):
    print(f"fold: {fold}")
    df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_SKCM.csv", index_col=0)
    df = df[df['seq1']!=0.0]

    vc = pd.concat([df['gene1'], df['gene2']]).value_counts()
    popular_gene = vc.index[0]

    other_genes = df[(df['gene1']!=popular_gene) & (df['gene2']!=popular_gene)]
    genes_in_other_genes = set(np.concatenate([other_genes['gene1'], other_genes['gene2']]))
    involving_genes = df[(df['gene1']==popular_gene) | (df['gene2']==popular_gene)]
    train_set = involving_genes[(~involving_genes['gene1'].isin(genes_in_other_genes)) | (~involving_genes['gene2'].isin(genes_in_other_genes))]
    test_set = other_genes
    print(train_set.groupby(['class']).size())
    print(test_set.groupby(['class']).size())

    train_set.to_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/double_holdout/train_seq_128_SKCM_{fold}.csv", index=False)
    test_set.to_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/holdout_experiments/double_holdout/test_seq_128_SKCM_{fold}.csv", index=False)
    
# double_holdout_SKCM(0)

# for cancer in ['LUAD']:
#     print(cancer)
#     for fold in range(30):
#         rng = random.Random(fold)
#         double_holdout(cancer, fold, rng)

# for fold in range(10):
#     double_holdout_SKCM(fold)

# results = {}
# for cancer in ['LUAD']:
#     seed_rng = random.Random(42)
#     print(cancer)
#     selected_seed_values = []
#     while len(selected_seed_values) < 10:
#         seed = seed_rng.randint(0, 10000)
#         rng = random.Random(seed)
#         if double_holdout(cancer, len(selected_seed_values), rng):
#             print(seed)
#             selected_seed_values.append(seed)
#     print(selected_seed_values)
#     results[cancer] = selected_seed_values
# print(results)
