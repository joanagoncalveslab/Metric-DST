import pickle
import numpy as np
import pandas as pd

df = pd.read_excel('W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/other_datasets/LU-raw-data.xlsx', header=2)
df = df[df['gene1'] != df['gene2']]
# df.groupby(['PMID', 'cancer type tested']).size().unstack(fill_value=0)

print(df.columns)
df = df[df['cancer type tested'] == 'CESC']
print(len(df))

df[['gene1', 'gene2']] = np.sort(df[['gene1', 'gene2']].values)

df_features = pd.read_csv('W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/test_seq_128.csv')
df_features = df_features[df_features['cancer']=='CESC']
df_features = pd.concat([
    pd.read_csv('W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/unknown_repair_cancer_CESC_seq_128.csv'),
    pd.read_csv('W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/train_seq_128_CESC.csv', index_col=0),
    df_features])
print(len(df_features))
print(len(df_features[df_features['gene1']==df_features['gene2']]))

df_features[['gene1', 'gene2']] = np.sort(df_features[['gene1', 'gene2']].values)

df_genes = set(df['gene1']).union(set(df['gene2']))
df_features_genes = set(df_features['gene1']).union(set(df_features['gene2']))

print(len(df_genes))
print(len(df_features_genes))
print(len(df_features_genes.union(df_genes).difference(df_features_genes.intersection(df_genes))))

merged = pd.merge(df, df_features, on=['gene1', 'gene2'], how='inner')

print(merged)
print(len(merged))

merged.to_csv('W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/other_datasets/merged_ISLE_CESC.csv')