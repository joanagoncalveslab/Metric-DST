import platform
import numpy as np
import torch
import torch.nn as nn

import customDataset, customAccuracyCalculator
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from pytorch_metric_learning import distances, losses, miners, reducers

import visualize
import create_convergence_graph

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

flags = {}

def layer(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_f, out_f, *args, **kwargs),
        nn.Sigmoid()
    )

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([layer(layers[i], layers[i+1], bias=True) for i in range(len(layers)-1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_epoch(model, data_loader, device, optimizer, loss_func, mining_func, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, labels, _) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        if flags['miner']:
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(torch.squeeze(embeddings), torch.squeeze(labels), indices_tuple)
        else:
            loss = loss_func(torch.squeeze(embeddings), torch.squeeze(labels))
        total_loss += loss.item()*data.size(0)
        loss.backward()
        optimizer.step()
    return total_loss/len(data_loader.sampler)

def test_epoch_dml(model, train_loader, test_loader, accuracy_calculator, device, loss_func, visualize_bool, fold, epoch, final_epoch):
    model.eval()
    test_loss = 0
    test_labels = torch.zeros(0, dtype=torch.long, device='cpu')
    test_embeddings = torch.zeros(0, dtype=torch.long, device='cpu')
    test_genes = []
    with torch.no_grad():
        for data, target, (gene1, gene2) in test_loader:
            data, target = data.to(device), target.to(device)
            test_genes.extend(list(zip(gene1, gene2)))
            test_labels = torch.cat([test_labels, target.cpu()])
            output = model(data)
            test_loss += loss_func(torch.squeeze(output), torch.squeeze(target)).item()*data.size(0)
            test_embeddings = torch.cat([test_embeddings, output.cpu()])
    test_loss /= len(test_loader.sampler)
    train_labels = torch.zeros(0, dtype=torch.long, device='cpu')
    train_embeddings = torch.zeros(0, dtype=torch.long, device='cpu')
    train_genes = []
    with torch.no_grad():
        for data, target, (gene1, gene2) in train_loader:
            data, target = data.to(device), target.to(device)
            train_genes.extend(list(zip(gene1, gene2)))
            train_labels = torch.cat([train_labels, target.cpu()])
            output = model(data)
            train_embeddings = torch.cat([train_embeddings, output.cpu()])
    if visualize_bool and flags['visualize_embeddings']:
        visualize.visualize(train_embeddings.numpy(), train_labels.numpy(), test_embeddings.numpy(), test_labels.numpy(), outputFolder, fold, epoch)
    if visualize_bool and flags['visualize_genes']:
        for gene in ['PARP1', 'BRCA1', 'PTEN', 'TP53', 'BRCA2']:
            visualize.visualize_gene(train_embeddings.numpy(), torch.cat([train_labels, test_labels]), train_genes, test_embeddings.numpy(), test_genes, gene, outputFolder, fold, epoch)

    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )

    if final_epoch and flags['save_embeddings']:
        columns = ['label']
        columns.extend(['dim'+str(x) for x in range(len(list(zip(*test_embeddings.numpy()))))])
        columns.extend(['gene1', 'gene2'])
        test_df = pd.DataFrame(
            data=list(zip(
                test_labels.numpy(), 
                *list(zip(*test_embeddings.numpy())), 
                *list(zip(*test_genes))
            )),
            columns=columns
        )
        train_df = pd.DataFrame(
            data=list(zip(
                train_labels.numpy(), 
                *list(zip(*train_embeddings.numpy())), 
                *list(zip(*train_genes))
            )),
            columns=columns
        )
        test_df.to_csv(outputFolder+f'embeddings_test_fold_{fold}.csv')
        train_df.to_csv(outputFolder+f'embeddings_train_fold_{fold}.csv')

    return [test_loss, accuracies["accuracy"], accuracies["f1_score"], accuracies["average_precision"], accuracies['auroc']]

def train(model, train_loader, test_loader, device, optimizer, loss_func, num_epochs, fold, mining_func, accuracy_calculator):
    metrics = []
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, device, optimizer, loss_func, mining_func, epoch)
        test_metrics = test_epoch_dml(model, train_loader=train_loader, test_loader=test_loader, accuracy_calculator=accuracy_calculator, device=device, loss_func=loss_func, visualize_bool=(epoch in [5,10,num_epochs]), fold=fold, epoch=epoch, final_epoch=epoch==num_epochs)
        # test_metrics = test_epoch_dml(model, train_loader=train_loader, test_loader=test_loader, accuracy_calculator=accuracy_calculator, device=device, loss_func=loss_func, visualize_bool=epoch==num_epochs, fold=fold, epoch=epoch, final_epoch=epoch==num_epochs)
        metrics.append([epoch, fold, tr_loss, *test_metrics])
    return pd.DataFrame(data=metrics, columns=['epoch', 'fold', 'train_loss', 'test_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc'])

def create_customDataset(data:pd.DataFrame):
    features = data.iloc[:, 4:]
    labels = data["class"].astype('long')
    genes = data.iloc[:, :2]
    return customDataset.CustomDataset(
        torch.Tensor(features.values.astype(np.float64)), 
        torch.Tensor(labels.values.astype(np.float64)), 
        genes.values.tolist())

def undersample(idx, labels):
    idx_0 = [id for id in idx if labels[id]==0]
    idx_1 = [id for id in idx if labels[id]==1]
    if len(idx_0) < len(idx_1):
        idx_1 = np.random.choice(idx_1, len(idx_0), replace=False)
    if len(idx_0) > len(idx_1):
        idx_0 = np.random.choice(idx_0, len(idx_1), replace=False)
    return np.concatenate([idx_0, idx_1])

def crossvalidation(layers, device, loss_func, num_epochs, dataset, test_dataset, batch_size, seed, learning_rate, mining_func, accuracy_calculator):
    df = pd.DataFrame()
    splits = StratifiedKFold(5, shuffle=True, random_state=seed)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)), dataset.labels)):
            print("fold: {}".format(fold))

            train_idx = undersample(train_idx, dataset.labels)
            val_idx = undersample(val_idx, dataset.labels)

            g_train = torch.Generator().manual_seed(seed+fold)
            g_val = torch.Generator().manual_seed(seed+fold)
            train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx, generator=g_train))
            val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx, generator=g_val))

            model = Net(layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            result = train(model, train_data_loader, val_data_loader, device, optimizer, loss_func, num_epochs, fold, mining_func, accuracy_calculator=accuracy_calculator)
            df = pd.concat([df, result], ignore_index=True, sort=False)
            performance_from_test_set = test_epoch_dml(model, train_data_loader, test_data_loader, accuracy_calculator, device, loss_func, True, fold, num_epochs+1, True)
            pd.DataFrame(
                data=[[fold, *performance_from_test_set]],
                columns=['fold', 'test_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
            ).to_csv(outputFolder+"performance-testset.csv", mode='a', header=fold==0)

    df.to_csv(outputFolder + "performance.csv")
    create_convergence_graph.create_fold_convergence_graph(outputFolder + "performance.csv", outputFolder)

def main(outputPath: str, dataset:pd.DataFrame, layers: list, testset:pd.DataFrame=None, batch_size=128, num_epochs=1, seed=42, learning_rate=0.01, flags_in={}, knn=5):
    global outputFolder
    outputFolder = outputPath
    global flags
    flags = flags_in


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()

    if flags['contrastive']:
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)
    else:
        loss_func = losses.TripletMarginLoss(margin=0.4, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all")

    accuracy_calculator = customAccuracyCalculator.CustomCalculator(include=("accuracy", "f1_score", "average_precision", "auroc"), k=knn)

    tr_dataset = create_customDataset(dataset)
    test_dataset=create_customDataset(testset)
    layers = [tr_dataset.data.shape[1]] + layers

    crossvalidation(layers, device, loss_func, num_epochs, tr_dataset, test_dataset, batch_size, seed, learning_rate, mining_func, accuracy_calculator)

if __name__ == "__main__":
    main()
