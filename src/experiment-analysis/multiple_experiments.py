import pandas as pd
import gc

for exp1, exp2, cancer in [('isle', 'dsl', 'BRCA'), ('isle', 'dsl', 'LUAD'), ('exp2sl', 'dsl', 'LUAD')]:
    print(f'{exp1}, {exp2}, {cancer}')
    # load both experiments
    df1 = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/{exp1}_seq_128.csv')
    df2 = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/{exp2}_seq_128.csv')

    # select the right cancer
    df1 = df1[df1['cancer']==cancer]
    df2 = df2[df2['cancer']==cancer]

    # remove unwanted samples
    df1 = df1[df1['gene1'] != df1['gene2']]
    df1 = df1[(df1['seq1'] != 0.0) & (df1['seq1'] != 0.0)]
    df2 = df2[df2['gene1'] != df2['gene2']]
    df2 = df2[(df2['seq1'] != 0.0) & (df2['seq1'] != 0.0)]

    # create extra column to make it easier to compare samples
    # this does asume the gene1 and gene2 are ordered the same across the two experiments
    df1['genes'] = list(zip(df1['gene1'], df1['gene2']))
    df2['genes'] = list(zip(df2['gene1'], df2['gene2']))

    # find the overlap between experiment 1 and experiment 2
    x = df1['genes'].isin(df2['genes'])
    common_genes= df1[x==True]['genes'].values
    print(f'overlap between {exp1} and {exp2} for cancer {cancer}: {len(common_genes)}')

    # select the overlap from both experiments
    # since the samples are sorted alphabetically in the both experiments, I can reset the index after selecting them since both of them are in the same order
    df1_common = df1[df1['genes'].isin(common_genes)].reset_index(drop=True)
    df2_common = df2[df2['genes'].isin(common_genes)].reset_index(drop=True)

    # find the saples that share the same label from the overlap
    y = df1_common['class'] == df2_common['class']
    common_genes_with_same_label= df1_common[y]['genes'].values
    print(f'overlap between {exp1} and {exp2} for cancer {cancer} with the same label: {len(common_genes_with_same_label)}')

    # select the samples that are not in the overlap with the same label
    # the samples that are in the overlap and have a different label will be in both the train and test set.
    df1_notcommon = df1[~df1['genes'].isin(common_genes_with_same_label)]
    df2_notcommon = df2[~df2['genes'].isin(common_genes_with_same_label)]

    # load unlabeled data
    unlabeled = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/unknown_repair_cancer_{cancer}_seq_128.csv')

    # remove unwanted unlabeled samples
    unlabeled = unlabeled[(unlabeled['seq1'] != 0.0) & (unlabeled['seq1'] != -1.0)]

    # create temporary column with gene combinations
    # since I don't know if these are alphabetical, I do it both ways
    unlabeled['genes1'] = list(zip(unlabeled['gene1'], unlabeled['gene2']))
    unlabeled['genes2'] = list(zip(unlabeled['gene2'], unlabeled['gene1']))

    # remove all samples that are in the experiments
    unlabeled = unlabeled[~unlabeled['genes1'].isin(df1['genes'])]
    unlabeled = unlabeled[~unlabeled['genes2'].isin(df2['genes'])]

    # remove the temporary columns again
    unlabeled = unlabeled.drop(['genes1', 'genes2'], axis=1)
    df1_notcommon = df1_notcommon.drop(['genes'], axis=1)
    df2_notcommon = df2_notcommon.drop(['genes'], axis=1)
    df1 = df1.drop(['genes'], axis=1)
    df2 = df2.drop(['genes'], axis=1)

    # print stats
    print(f'{exp1}, {cancer}; test size: {len(df1)}, train size: {len(df1_notcommon)}')
    print(f'{exp2}, {cancer}; test size: {len(df2)}, train size: {len(df2_notcommon)}')
    print(f'{exp1}, {exp2}, {cancer}; unlabeled size: {len(unlabeled)}')

    # save the datasets
    df1_notcommon.to_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/feature_sets/{exp1}_{exp2}_{cancer}_train_seq_128.csv', index=False)
    df2_notcommon.to_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/feature_sets/{exp2}_{exp1}_{cancer}_train_seq_128.csv', index=False)
    df1.to_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/feature_sets/{exp2}_{exp1}_{cancer}_test_seq_128.csv', index=False)
    df2.to_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/feature_sets/{exp1}_{exp2}_{cancer}_test_seq_128.csv', index=False)
    unlabeled.to_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/feature_sets/unknown_repair_{exp1}_{exp2}_{cancer}_seq_128.csv', index=False)
    
    gc.collect()
