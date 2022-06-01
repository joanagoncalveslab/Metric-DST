import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def create_convergence_graph(location):
    df = pd.read_csv(location, index_col=0)
    print(df.head())
    sn.set_theme()
    sn.lineplot(data=df)
    plt.savefig('performance-logistic-regression-bce.png')
    plt.show()

if __name__=="__main__":
    create_convergence_graph("W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/performance-logistic-regression-bce-2.csv")