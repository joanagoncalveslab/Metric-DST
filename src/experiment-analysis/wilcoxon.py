import pandas as pd
from scipy.stats import wilcoxon


# cancer = 'LUAD'
# conf = 90
# pseudolabel = 10

# for (cancer, conf, pseudolabel) in [('BRCA', 90, 20), ('OV', 85, 20), ('CESC', 90, 10), ('SKCM', 90, 10), ('LUAD', 90, 10)]:
#     print(cancer)
#     supervised =                pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-pipeline-what-settings/experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/supervised_test_performance.csv", index_col=0).reset_index()
#     true_semi_supervised =      pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-pipeline-what-settings/experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/true_semi_supervised_test_performance.csv", index_col=0).reset_index()
#     balanced_semi_supervised =  pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-pipeline-what-settings/experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/balanced_semi_supervised_test_performance.csv", index_col=0).reset_index()
#     divergence =                pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-pipeline-what-settings/experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/diversity_test_performance.csv", index_col=0).reset_index()

# for (cancer, conf, pseudolabel) in [('BRCA', 85, 6), ('OV', 80, 6), ('CESC', 75, 6), ('SKCM', 75, 20), ('LUAD', 90, 10)]:
#     print(cancer)
#     supervised =                pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-double-holdout/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/supervised_test_performance.csv", index_col=0).reset_index()
#     true_semi_supervised =      pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-double-holdout/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/true_semi_supervised_test_performance.csv", index_col=0).reset_index()
#     balanced_semi_supervised =  pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-double-holdout/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/balanced_semi_supervised_test_performance.csv", index_col=0).reset_index()
#     divergence =                pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-double-holdout/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/diversity_test_performance.csv", index_col=0).reset_index()

for exp, cancer, pseudolabel, conf in [('isle-dsl', 'BRCA', 6, 90), ('dsl-isle', 'BRCA', 6, 90), ('isle-dsl', 'LUAD', 10, 80), ('dsl-isle', 'LUAD', 10, 85), ('exp2sl-dsl', 'LUAD', 6, 85), ('dsl-exp2sl', 'LUAD', 10, 80)]:
    print(f'{exp} {cancer}')
    supervised =                pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/output/experiment-{exp}-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/supervised_test_performance.csv", index_col=0).reset_index()
    true_semi_supervised =      pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/output/experiment-{exp}-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/true_semi_supervised_test_performance.csv", index_col=0).reset_index()
    balanced_semi_supervised =  pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/output/experiment-{exp}-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/balanced_semi_supervised_test_performance.csv", index_col=0).reset_index()
    divergence =                pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/output/experiment-{exp}-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/diversity_test_performance.csv", index_col=0).reset_index()

    measurement = 'auroc'

    supervised = supervised[measurement]
    true_semi_supervised = true_semi_supervised[measurement]
    balanced_semi_supervised = balanced_semi_supervised[measurement]
    divergence = divergence[measurement]

    supst = wilcoxon(x=supervised, y=true_semi_supervised, alternative='two-sided')
    supbst = wilcoxon(x=supervised, y=balanced_semi_supervised, alternative='two-sided')
    supdbst = wilcoxon(x=supervised, y=divergence, alternative='two-sided')

    # print(f"supervised -> ST; statistic: {supst.statistic}, p-value: {supst.pvalue}")
    # print(f"supervised -> BST; statistic: {supbst.statistic}, p-value: {supbst.pvalue}")
    # print(f"supervised -> DBST; statistic: {supdbst.statistic}, p-value: {supdbst.pvalue}")

    print(supst.pvalue)
    print(supbst.pvalue)
    print(supdbst.pvalue)