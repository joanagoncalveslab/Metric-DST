import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def create_convergence_graph(location):
    df = pd.read_csv(location, index_col=0)
    print(df.head())
    sn.set_theme()
    sn.lineplot(data=df)
    # plt.savefig('performance-logistic-regression-bce.png')
    plt.show()

def create_confusion_matrix(path):
    df = pd.read_csv(path, index_col=0)
    sn.set_theme()
    sn.heatmap(df, annot=True, fmt='d')
    # plt.savefig('output.png')
    plt.show()

def create_fold_convergence_graph(location, folder):
    df = pd.read_csv(location, index_col=0)
    df = df[["epoch", "train_loss", "validation_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
    df = df.melt('epoch', var_name='cols', value_name='vals')
    sn.set_theme()
    sn.lineplot(data=df, x="epoch", y='vals', hue='cols', ci='sd')
    plt.savefig(location+".png")
    # plt.show()

def create_fold_convergence_graph_individual_folds(location, folder):
    df = pd.read_csv(location, index_col=0)
    df = df[["epoch", "fold", "train_loss", "validation_loss", "accuracy", "f1_score", "average_precision", "auroc"]]
    df = df.melt(['epoch', 'fold'], var_name='cols', value_name='vals')
    sn.set_theme()
    g=sn.FacetGrid(df, col='fold', col_wrap=3)
    g.map(sn.lineplot, 'epoch', 'vals', 'cols')
    # sn.lineplot(data=df, x="epoch", y='vals', hue='cols', style='fold')
    # plt.savefig(folder+'performance.png')
    plt.show()

def calculate(location):
    df=pd.read_csv(location, index_col=0)
    df = df[df['epoch']==100]
    print(df.mean())

def near(y_label, existing_labels):
    for label in existing_labels:
        if abs(y_label - label) < 0.05:
            return True
    return False

def create_facet_grid(data:pd.DataFrame, outputfile:str):
    sn.set_theme()
    # Use relplot
    g=sn.FacetGrid(data, col='hidden nodes', col_wrap=2)
    g.map(sn.lineplot, 'epoch', 'vals', 'cols')

    # gr = data[data['epoch']==100].groupby(['hidden nodes', 'cols'])
    # means = gr.std()
    # stds = gr.mean()
    # print(means)
    # print(stds)

    # for hidden_node, axis in g.axes_dict.items():
    #     present_labels = []
    #     for l in axis.lines:
    #         if l.get_label() in ['accuracy', 'average_precision', 'auroc']:
    #             label_y = gr.mean().loc[(hidden_node, l.get_label()),('vals')]
    #             print(near(label_y, present_labels))
    #             present_labels.append(label_y)
    #             axis.annotate(
    #                 f"{means.loc[(hidden_node, l.get_label()),('vals')]:.2f}({stds.loc[(hidden_node, l.get_label()),('vals')]:.2f})", 
    #                 xy=(1,label_y), 
    #                 xycoords=('axes fraction', 'data'),
    #                 ha='left', 
    #                 va='center', 
    #                 color=l.get_color()
    #             )


    g.add_legend()
    plt.savefig(outputfile+".png")
    plt.show()

if __name__=="__main__":
    # create_fold_convergence_graph_individual_folds("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-retrain-knn-10-conf-8/performance.csv", "")
    create_fold_convergence_graph("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-divergence-pipeline-BRCA-knn-10-conf-85/diversity_performance.csv", "")
    exit()
    # create_fold_convergence_graph("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-64-lr-0.01-ml/performance.csv")

    # experiment = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA-64-0.01-LossMargin-0.4/"
    # create_fold_convergence_graph(experiment+'performance8.csv', '')

    # create_confusion_matrix('experiment-5/confusion_matrix.csv')
    # experiment="OV-64"

    for experiment in ["BRCA-64-0.01-Lp-power-2"]:
        print(experiment)
        file = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-"+experiment+"/"
        # file = "experiment-"+experiment+"/"

    #     # LEARNING RATE
    #     # df1 = pd.read_csv(file+"2-ml/performance8.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    #     # df1.insert(3, 'learning rate', '0.5')
    #     # df2 = pd.read_csv(file+"0.2/performance8.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    #     # df2.insert(3, 'loss margin', '0.1')
    #     # df3 = pd.read_csv(file+"0.3/performance8.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    #     # df3.insert(3, 'loss margin', '0.2')
    #     # df4 = pd.read_csv(file+"0.4/performance8.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    #     # df4.insert(3, 'loss margin', '0.3')
    #     # df5 = pd.read_csv(file+"0.5/performance8.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    #     # df5.insert(3, 'loss margin', '0.4')
    #     # df6 = pd.read_csv(file+"1000-ml/performance8.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    #     # df6.insert(3, 'learning rate', '0.001')

    #     # MULTIPLE LAYERS
        df2 = pd.read_csv(file+"performance2.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
        df2.insert(3, 'hidden nodes', '2, 2')
        df3 = pd.read_csv(file+"performance4.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
        df3.insert(3, 'hidden nodes', '4, 2')
        df4 = pd.read_csv(file+"performance8.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
        df4.insert(3, 'hidden nodes', '8, 2')
        df5 = pd.read_csv(file+"performance16.csv")[["epoch", "train_loss", "test_loss", "accuracy", "f1_score", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
        df5.insert(3, 'hidden nodes', '16, 2')
        conc = pd.concat([df2, df3, df4, df5])
        create_facet_grid(conc, "png-"+experiment)
    print("DONE")
