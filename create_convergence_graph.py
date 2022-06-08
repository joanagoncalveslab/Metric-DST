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

if __name__=="__main__":
    # create_fold_convergence_graph("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-l-16-0/performance.csv")
    # create_fold_convergence_graph('experiment-6/performance.csv')
    # create_confusion_matrix('experiment-5/confusion_matrix.csv')
    calculate("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-0/performance.csv")
    calculate("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-l-2-0/performance.csv")
    calculate("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-l-4-0/performance.csv")
    calculate("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-l-8-0/performance.csv")
    calculate("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/experiment-l-16-0/performance.csv")
