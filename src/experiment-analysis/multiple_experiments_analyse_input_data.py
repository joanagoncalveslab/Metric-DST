import pandas as pd

for exp1, exp2, cancer in [('isle', 'dsl', 'LUAD'), ('dsl', 'isle', 'LUAD'), ('isle', 'dsl', 'BRCA'), ('dsl', 'isle', 'BRCA'), ('exp2sl', 'dsl', 'LUAD'), ('dsl', 'exp2sl', 'LUAD')]:
    print(f'{exp1}, {exp2}, {cancer}')
    df_train = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/feature_sets/{exp1}_{exp2}_{cancer}_train_seq_128.csv')
    df_test = pd.read_csv(f'W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/feature_sets/{exp1}_{exp2}_{cancer}_test_seq_128.csv')

    print(df_train['class'].sum())
    print(df_train['class'].count() - df_train['class'].sum())
    print(df_test['class'].sum())
    print(df_test['class'].count() - df_test['class'].sum())