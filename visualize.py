import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

def visualize(embeddings, labels, test_embeddings, test_labels, output, fold, epoch):
    embeddings = pd.DataFrame(embeddings).astype("float")
    embeddings['labels'] = ['train negative' if label==0 else 'train positive' for label in labels]
    test_embeddings = pd.DataFrame(test_embeddings).astype("float")
    test_embeddings['labels'] = ['test negative' if label==0 else 'test positive' for label in test_labels]

    palette = {
        'train negative': to_rgba('tab:blue', 0.2),
        'train positive': to_rgba('tab:orange', 0.2),
        'test negative' : to_rgba('tab:blue', 1),
        'test positive': to_rgba('tab:orange', 1)
    }

    ax = sn.scatterplot(data=embeddings, x=0, y=1, hue='labels', palette=palette)
    sn.scatterplot(data=test_embeddings, x=0, y=1, hue='labels', ax=ax, marker="X", palette=palette)

    # concatenated = pd.concat([embeddings.assign(dataset='train'), test_embeddings.assign(dataset='test')])
    # sn.scatterplot(data=concatenated, x=0, y=1, hue='labels', style='dataset')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(embeddings[0], embeddings[1], embeddings[2], c='skyblue')

    plt.savefig(output+f'visualization-fold-{fold}-epoch-{epoch}.png')
    # plt.show()
    plt.clf()

def visualize_gene(embeddings, labels, genes, test_embeddings, test_genes, gene, outputFolder, fold, epoch):
    embeddings = pd.DataFrame(embeddings).astype("float")
    embeddings['colour'] = [gene if gene in g else 'other' for g in genes]
    test_embeddings = pd.DataFrame(test_embeddings).astype("float")
    test_embeddings['colour'] = ['test_'+str(gene) if gene in g else 'other' for g in test_genes]
    embeddings = pd.concat([embeddings, test_embeddings])

    palette = {
        gene: to_rgba('tab:blue', 1),
        'test_'+str(gene): to_rgba('tab:blue', 0.5),
        'other': to_rgba('gray', 0.2) 
    }
    # https://matplotlib.org/stable/gallery/color/named_colors.html

    sn.scatterplot(data=embeddings, x=0, y=1, hue='colour', style=labels, palette=palette)
    plt.savefig(outputFolder+f'visualization-gene-{gene}-fold-{fold}-epoch-{epoch}.png')
    # plt.show()
    plt.clf()

def visualize_show(embeddings, labels, test_embeddings, test_labels):
    embeddings = pd.DataFrame(embeddings).astype("float")
    embeddings['labels'] = ['train negative' if label==0 else 'train positive' for label in labels]
    test_embeddings = pd.DataFrame(test_embeddings).astype("float")
    test_embeddings['labels'] = ['test negative' if label==0 else '10 worst train positives' for label in test_labels]

    palette = {
        'train negative': to_rgba('tab:blue', 0.2),
        'train positive': to_rgba('tab:orange', 0.2),
        'test negative' : to_rgba('tab:blue', 1),
        '10 worst train positives': to_rgba('tab:orange', 1)
    }

    ax = sn.scatterplot(data=embeddings, x=0, y=1, hue='labels', palette=palette)
    sn.scatterplot(data=test_embeddings, x=0, y=1, hue='labels', ax=ax, marker="X", palette=palette)

    plt.show()
    plt.clf()