import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

def visualize(embeddings, labels, test_embeddings, test_labels, output, fold, epoch):
    embeddings = pd.DataFrame(embeddings).astype("float")
    embeddings['labels'] = ['train negative' if label==0 else 'train positive' for label in labels]
    test_embeddings = pd.DataFrame(test_embeddings).astype("float")
    test_embeddings['labels'] = ['test negative' if label==0 else 'test positive' for label in test_labels]

    ax = sn.scatterplot(data=embeddings, x=0, y=1, hue='labels', alpha=0.2)
    sn.scatterplot(data=test_embeddings, x=0, y=1, hue='labels', alpha=1, ax=ax, marker="X")

    # concatenated = pd.concat([embeddings.assign(dataset='train'), test_embeddings.assign(dataset='test')])
    # sn.scatterplot(data=concatenated, x=0, y=1, hue='labels', style='dataset')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(embeddings[0], embeddings[1], embeddings[2], c='skyblue')

    plt.savefig(output+f'visualization-fold-{fold}-epoch-{epoch}.png')
    # plt.show()
    plt.clf()