import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def replace_name(name):
    if name == 'diversity':
        return 'Diverse BST'
    elif name == 'true_semi_supervised':
        return 'Self Training'
    elif name == 'balanced_semi_supervised':
        return 'Balanced ST'
    elif name == 'supervised':
        return 'Supervised'

# for n in ['supervised', 'semisupervised', 'diversity']:
#     df = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-single-holdout-divergence-pipeline-SKCM-knn-10-conf-85/"+n+"_performance.csv", index_col=0).reset_index()
#     # df = df[df['epoch']==199]
#     df = df.loc[df.groupby('fold', sort=False)['validation_loss'].idxmin()]
#     df = df[["validation_loss", "train_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#     # print(df.describe())
#     print(df.mean().to_frame().T)
#     print(df.std().to_frame().T)

# ---------------------------------

# with open("C:/Users/mathi/git/msc-thesis-2122-mathijs-de-wolf/experiment-6/distances.txt") as file:
#     distances = pd.DataFrame(
#         data=[[float(y) for y in x.split(',')] for x in file.readlines()],
#         columns=['order', 'distances']
#     )
#     # print(distances)
#     sn.boxplot(distances, x="order", y="distances")
#     plt.show()

# ------------------------------------

# combined_df = pd.DataFrame()
# for cancer in ['OV', 'CESC', 'SKCM']:
#     for n in ['diversity']:
#         for i in [1,2,3]:
#             # df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-pipeline-{cancer}-knn-10-conf-85-pseudolabels-50{'' if i==1 else f'-{i}'}/{n}_test_performance.csv", index_col=0).reset_index()
#             df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-pipeline-{cancer}-knn-10-conf-85-pseudolabels-50-4/{n}_{i}_test_performance.csv", index_col=0).reset_index()
#             # df = df.loc[df.groupby('fold', sort=False)['validation_loss'].idxmin()]
#             df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#             df['hue'] = n
#             df['class'] = f"{cancer}, {i}"
#             combined_df = pd.concat([combined_df, df])

# sn.boxplot(combined_df, x='class', y='average_precision', hue='hue')
# plt.show()

# combined_df = pd.DataFrame()
# # for pseudolabel, conf in [(10, 65), (10,75), (10, 85), (20, 75), (50, 75), (50, 85)]:
# for pseudolabel, conf in [(10, 65), (10,75), (10, 85), (20, 75), (50, 75), (50, 85)]:
#     for n in ['supervised', 'semisupervised', 'diversity']:
#         df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-pipeline-OV-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/"+n+"_performance.csv", index_col=0).reset_index()
#         df = df.loc[df.groupby('fold', sort=False)['validation_loss'].idxmin()]
#         df = df[["validation_loss", "train_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#         df['hue'] = n
#         df['class'] = f'{pseudolabel}, {conf}'
#         combined_df = pd.concat([combined_df, df])

# sn.boxplot(combined_df, x='class', y='average_precision', hue='hue')
# plt.show()

# combined_df = pd.DataFrame()
# cancer = "SKCM"
# for pseudolabel, conf in [(10, 75), (20, 75), (10, 90), (20, 90), (50, 90)]:
#     for n in ['supervised', 'true_semi_supervised', 'balanced_semi_supervised', 'diversity']:
#         df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/"+n+"_test_performance.csv", index_col=0).reset_index()
#         # df = df.loc[df.groupby('fold', sort=False)['validation_loss'].idxmin()]
#         df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#         df['hue'] = n
#         df['class'] = f'{pseudolabel}, {conf}'
#         combined_df = pd.concat([combined_df, df])

# sn.boxplot(combined_df, x='class', y='average_precision', hue='hue')

# combined_df = pd.DataFrame()
# for cancer, pseudolabel, conf in [('BRCA', 10, 75), ('BRCA', 10, 85), ('BRCA', 20, 75), ('CESC', 10, 70), ('CESC', 6, 70), ('OV', 10, 70), ('OV', 10, 75), ('SKCM', 10, 70)]:
#     for n in ['supervised', 'semisupervised', 'diversity']:
#         print(f'{cancer}, {pseudolabel}, {conf}, {n}')
#         df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/"+n+"_test_performance.csv", index_col=0).reset_index()
#         # df = df.loc[df.groupby('fold', sort=False)['validation_loss'].idxmin()]
#         df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#         df['hue'] = n
#         df['class'] = f'{cancer}, {pseudolabel}, {conf}'
#         combined_df = pd.concat([combined_df, df])
#         print(combined_df[combined_df['average_precision'] > 1])
# sn.boxplot(combined_df, x='class', y='average_precision', hue='hue')
# plt.show()

# cancer = 'LUAD'
# combined_df = pd.DataFrame()
# for pseudolabel in [6, 10, 20]:
#     for conf in [75, 80, 85, 90]:
#         # if (pseudolabel, conf) not in [(10, 90), (10, 95), (20, 95)]:
#             for n in ['supervised', 'balanced_semi_supervised', 'diversity']:
#                 # df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-pipeline-{cancer}-knn-10-conf-85-pseudolabels-50{'' if i==1 else f'-{i}'}/{n}_test_performance.csv", index_col=0).reset_index()
#                 df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/{n}_test_performance.csv", index_col=0).reset_index()
#                 # df = df.loc[df.groupby('fold', sort=False)['validation_loss'].idxmin()]
#                 df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#                 df['hue'] = n
#                 df['class'] = f"{pseudolabel}, {conf}"
#                 combined_df = pd.concat([combined_df, df])

# combined_df = pd.DataFrame()
# # for (cancer, conf, pseudolabel) in [('BRCA', 85, 6), ('LUAD', 90, 10), ('SKCM', 75, 20)]:
# cancer = "OV"
# for conf in [80, 85]:
#     for pseudolabel in [6, 10, 20]:
#         if not (conf == 85 and pseudolabel == 6):
#             for n in ['supervised', 'true_semi_supervised', 'balanced_semi_supervised', 'diversity']:
#                 df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-double-holdout/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/{n}_test_performance.csv", index_col=0).reset_index()
#                 df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#                 df['hue'] = n
#                 df['cancer'] = f"{cancer}, {pseudolabel}, {conf}"
#                 combined_df = pd.concat([combined_df, df])

# combined_df = pd.DataFrame()
# for (cancer, conf, pseudolabel) in [('BRCA', 90, 20), ('CESC', 85, 20), ('OV', 85, 20), ('SKCM', 75, 20)]:
#     for n in ['supervised', 'semisupervised', 'diversity']:
#         df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-single-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/{n}_test_performance.csv", index_col=0).reset_index()
#         df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#         df['hue'] = n
#         df['class'] = f"{cancer}, {pseudolabel}, {conf}"
#         combined_df = pd.concat([combined_df, df])

# sn.boxplot(combined_df, x='class', y='average_precision', hue='hue')
# plt.show()

# ----------- Random split ------------
combined_df = pd.DataFrame()
for (cancer, conf, pseudolabel) in [('BRCA', 90, 20), ('OV', 85, 20), ('CESC', 90, 10), ('SKCM', 90, 10), ('LUAD', 90, 10)]:
    for n in ['supervised', 'true_semi_supervised', 'balanced_semi_supervised', 'diversity']:
        df = pd.read_csv(f"/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-pipeline-what-settings/experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/{n}_test_performance.csv", index_col=0).reset_index()
        df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
        # print(f"{cancer} {replace_name(n)}: {df['average_precision'].median()}")
        df['hue'] = replace_name(n)
        df['Cancer'] = f"{cancer}"
        df['AUPRC'] = df['average_precision']
        df['AUROC'] = df['auroc']
        combined_df = pd.concat([combined_df, df])
print(combined_df)


# ----------- Double holdout ------------
# combined_df = pd.DataFrame()
# for (cancer, conf, pseudolabel) in [('BRCA', 85, 6), ('OV', 80, 6), ('CESC', 75, 6), ('SKCM', 75, 20), ('LUAD', 90, 10)]:
#     print(cancer)
#     for n in ['supervised', 'true_semi_supervised', 'balanced_semi_supervised', 'diversity']:
#         df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiments-double-holdout/experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/{n}_test_performance.csv", index_col=0).reset_index()
#         df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#         print(df['average_precision'].median())
#         df['hue'] = replace_name(n)
#         df['cancer'] = f"{cancer}"
#         combined_df = pd.concat([combined_df, df])


# combined_df = pd.DataFrame()
# for exp, cancer, pseudolabel, conf in [('isle-dsl', 'BRCA', 6, 90), ('dsl-isle', 'BRCA', 6, 90), ('isle-dsl', 'LUAD', 10, 80), ('dsl-isle', 'LUAD', 10, 85), ('exp2sl-dsl', 'LUAD', 6, 85), ('dsl-exp2sl', 'LUAD', 10, 80)]:
# # exp = 'dsl-isle'
# # cancer = 'LUAD'
# # for pseudolabel in [6, 10, 20]:
# #     for conf in [80, 85]:
# #         if (pseudolabel, conf) not in [(10, 80)]:
#             print(f'{exp} {cancer}')
#             for n in ['supervised', 'true_semi_supervised', 'balanced_semi_supervised', 'diversity']:
#                 df = pd.read_csv(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/multiple_experiments/output/experiment-{exp}-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}/{n}_test_performance.csv", index_col=0).reset_index()
#                 # df = df.loc[df.groupby('fold', sort=False)['validation_loss'].idxmin()]
#                 df = df[["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
#                 print(df['average_precision'].median())
#                 df['hue'] = replace_name(n)
#                 # df['cancer'] = f"{exp}, {cancer}, {pseudolabel}, {conf}"
#                 df['experiment'] = f"{exp}, {cancer}"
#                 combined_df = pd.concat([combined_df, df])



# color_palette = {
#         0: (103/255, 169/255, 207/255, 1),
#         1: (239/255, 138/255, 98/255, 1),
#         -1: (218/255, 218/255, 218/255, 0.5)
#     }

color_palette = [
    (0/255,      118/255,   194/255),
    (237/255,    104/255,   66/255),
    (0/255,      155/255,   119/255),
    (165/255,    0/255,     52/255)
]

# for metric in ["test_loss", "accuracy", "f1_score", "average_precision", "auroc"]:
for metric in ["test_loss", "average_precision", "auroc"]:
    plt.close()
    fig = plt.figure(figsize=(10,5))
    sn.boxplot(data=combined_df, y=metric, x='cancer', hue='hue', showfliers=False, palette=color_palette)
    sn.stripplot(data=combined_df, y=metric, x='cancer', hue='hue', palette='dark:black', alpha=1, dodge=True, legend=False)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='')
    # plt.savefig(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/figures/multiple-experiments-final-performance-{metric}-with-dots.pdf", dpi=300, format='pdf', bbox_inches='tight')
    # plt.savefig(f"W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/figures/multiple-experiments-final-performance-{metric}-with-dots", dpi=300, bbox_inches='tight')
    # plt.show()

    # plt.close()
    # plt.figure(figsize=(10,5))
    # sn.boxplot(combined_df, x='class', y=metric, hue='hue')
    # # plt.savefig(f"C:/Users/mathi/Downloads/updated-pipeline-performance-{metric}")
    # # plt.show()