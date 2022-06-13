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

def create_fold_convergence_graph(location):
    df = pd.read_csv(location, index_col=0)
    df = df[["epoch", "train_loss", "test_loss", "auroc", "auprc", "f1", "accuracy", "average_precision"]]
    df = df.melt('epoch', var_name='cols', value_name='vals')
    sn.set_theme()
    sn.lineplot(data=df, x="epoch", y='vals', hue='cols', ci=95)
    plt.savefig('performance-l-16-0.png')
    # plt.show()

def calculate(location):
    df=pd.read_csv(location, index_col=0)
    df = df[df['epoch']==75]
    print(df.mean())

def create_facet_grid(data):
    sn.set_theme()
    g=sn.FacetGrid(data, col='hidden nodes', col_wrap=3)
    g.map(sn.lineplot, 'epoch', 'vals', 'cols')
    plt.savefig('png.png')
    plt.show()

if __name__=="__main__":
    # create_fold_convergence_graph("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-l-16-0/performance.csv")
    # create_fold_convergence_graph('experiment-6/performance.csv')
    # create_confusion_matrix('experiment-5/confusion_matrix.csv')
    df1 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA/performance0.csv")[["epoch", "train_loss", "test_loss", "auroc", "auprc", "f1", "accuracy", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    df1.insert(3, 'hidden nodes', 0)
    df2 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA/performance2.csv")[["epoch", "train_loss", "test_loss", "auroc", "auprc", "f1", "accuracy", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    df2.insert(3, 'hidden nodes', 2)
    df3 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA/performance4.csv")[["epoch", "train_loss", "test_loss", "auroc", "auprc", "f1", "accuracy", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    df3.insert(3, 'hidden nodes', 4)
    df4 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA/performance8.csv")[["epoch", "train_loss", "test_loss", "auroc", "auprc", "f1", "accuracy", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    df4.insert(3, 'hidden nodes', 8)
    df5 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA/performance16.csv")[["epoch", "train_loss", "test_loss", "auroc", "auprc", "f1", "accuracy", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    df5.insert(3, 'hidden nodes', 16)
    df6 = pd.read_csv("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-BRCA/performance32.csv")[["epoch", "train_loss", "test_loss", "auroc", "auprc", "f1", "accuracy", "average_precision"]].melt('epoch', var_name='cols', value_name='vals')
    df6.insert(3, 'hidden nodes', 32)
    conc = pd.concat([df1, df2, df3, df4, df5, df6])
    create_facet_grid(conc)
