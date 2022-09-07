import platform
import numpy as np
import torch
import torch.nn as nn

import customDataset, customAccuracyCalculator
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import visualize
import create_convergence_graph

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

def layer(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_f, out_f, *args, **kwargs),
        nn.Sigmoid(),
        nn.Dropout(p=0.5)
    )

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([layer(layers[i], layers[i+1], bias=True) for i in range(len(layers)-1)])
        self.fc2 = nn.Linear(128, 8, bias=True)
        self.fc1 = nn.Linear(8, 2, bias=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

def train_epoch(model, data_loader, device, optimizer, loss_func, mining_func, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(torch.squeeze(embeddings), torch.squeeze(labels), indices_tuple)
        # loss = loss_func(torch.squeeze(embeddings), torch.squeeze(labels))
        total_loss += loss.item()*data.size(0)
        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print(
        #         "Epoch {} Iteration {}: Loss = {}".format(
        #             epoch, batch_idx, loss
        #         )
        #     )
    return total_loss/len(data_loader.sampler)

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester(dataloader_num_workers=0)
    return tester.get_all_embeddings(dataset, model)

def test_epoch_dml(model, train_loader, test_loader, accuracy_calculator, device, loss_func, visualize_bool, fold, epoch):
    model.eval()
    test_loss = 0
    y_true = torch.zeros(0, dtype=torch.long, device='cpu')
    y_prob = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_true = torch.cat([y_true, target.cpu()])
            output = model(data)
            test_loss += loss_func(torch.squeeze(output), torch.squeeze(target)).item()*data.size(0)
            y_prob = torch.cat([y_prob, output.cpu()])
    test_loss /= len(test_loader.sampler)
    y_true_train = torch.zeros(0, dtype=torch.long, device='cpu')
    y_prob_train = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            y_true_train = torch.cat([y_true_train, target.cpu()])
            output = model(data)
            y_prob_train = torch.cat([y_prob_train, output.cpu()])
    if visualize_bool:
        visualize.visualize(y_prob_train.numpy(), y_true_train.numpy(), y_prob.numpy(), y_true.numpy(), outputFolder, fold, epoch)
    train_embeddings, train_labels = y_prob_train, y_true_train
    test_embeddings, test_labels = y_prob, y_true
    # train_labels = train_labels.squeeze(1)
    # test_labels = test_labels.squeeze(1)
    # print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )

    accuracies_test_reference = accuracy_calculator.get_accuracy(
        test_embeddings, test_embeddings, test_labels, test_labels, True
    )
    # print(
    #     "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    # print("Test set loss = {}".format(test_loss))
    return [test_loss, accuracies["mean_average_precision_at_r"], accuracies_test_reference["mean_average_precision_at_r"]]


def test_epoch(model, device, test_loader, last_round, loss_func):
    model.eval()
    test_loss = 0
    y_pred = torch.zeros(0, dtype=torch.long, device='cpu')
    y_true = torch.zeros(0, dtype=torch.long, device='cpu')
    y_prob = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_true = torch.cat([y_true, target.view(-1).cpu()])
            output = model(data)
            y_prob = torch.cat([y_prob, output.view(-1).cpu()])
            test_loss += loss_func(torch.squeeze(output), torch.squeeze(target)).item()*data.size(0)
            pred = output.round()
            y_pred = torch.cat([y_pred, pred.view(-1).cpu()])

    test_loss /= len(test_loader.sampler)
    # if last_round:
    #     classes = ['negative', 'positive']
    #     cf_matrix = confusion_matrix(y_true, y_pred)
    #     # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=classes, columns=classes)
    #     df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
    #     df_cm.to_csv(outputFolder + 'confusion_matrix.csv')
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_prob)

    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     test_loss, y_true.eq(y_pred).sum().item(), len(test_loader.sampler), 100*accuracy))
    return [test_loss, auprc, auroc, accuracy, f1, average_precision]

def train(model, train_loader, test_loader, device, optimizer, loss_func, num_epochs, fold, mining_func, accuracy_calculator):
    metrics = []
    # test_epoch_dml(model, train_loader, test_loader, accuracy_calculator, device, loss_func, True, fold, 0)
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, device, optimizer, loss_func, mining_func, epoch)
        test_metrics = test_epoch_dml(model, train_loader=train_loader, test_loader=test_loader, accuracy_calculator=accuracy_calculator, device=device, loss_func=loss_func, visualize_bool=epoch in [1,2,3,4,5,6,7,8,9,10,num_epochs], fold=fold, epoch=epoch)
        metrics.append([epoch, fold, tr_loss, *test_metrics])
        # metrics.append([epoch, fold, tr_loss])
        # test_metrics = test_epoch(model, device, test_loader, num_epochs==epoch, loss_func)
        # metrics.append([epoch, fold, tr_loss, *test_metrics])
    # df = pd.DataFrame(data=metrics, columns=['epoch', 'fold', 'train_loss', 'test_loss', 'auprc', 'auroc', 'accuracy', 'f1', 'average_precision'])
    df = pd.DataFrame(data=metrics, columns=['epoch', 'fold', 'train_loss', 'test_loss', 'mean_average_precision_at_r', 'test_reference'])
    return df

def get_data(path):
    file = pd.read_csv(path).fillna(0)
    data = file.iloc[:, 4:]
    labels = file["class"]
    dataset = customDataset.CustomDataset(torch.Tensor(data.values.astype(np.float64)), torch.Tensor(labels.values.astype(np.float64)))
    return dataset

def create_customDataset(data:pd.DataFrame):
    features = data.iloc[:, 4:]
    labels = data["class"].astype('long')
    dataset = customDataset.CustomDataset(torch.Tensor(features.values.astype(np.float64)), torch.Tensor(labels.values.astype(np.float64)))
    return dataset

def undersample(idx, labels):
    idx_0 = [id for id in idx if labels[id]==0]
    idx_1 = [id for id in idx if labels[id]==1]
    if len(idx_0) < len(idx_1):
        idx_1 = np.random.choice(idx_1, len(idx_0), replace=False)
    if len(idx_0) > len(idx_1):
        idx_0 = np.random.choice(idx_0, len(idx_1), replace=False)
    return np.concatenate([idx_0, idx_1])

def crossvalidation(layers, device, loss_func, num_epochs, dataset, batch_size, seed, learning_rate, mining_func, accuracy_calculator):
    df = pd.DataFrame()
    splits = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)), dataset.labels)):
            print("fold: {}".format(fold))

            train_idx = undersample(train_idx, dataset.labels)
            val_idx = undersample(val_idx, dataset.labels)

            g_train = torch.Generator().manual_seed(seed+fold)
            g_val = torch.Generator().manual_seed(seed+fold)
            train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx, generator=g_train))
            test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx, generator=g_val))

            model = Net(layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            result = train(model, train_data_loader, test_data_loader, device, optimizer, loss_func, num_epochs, fold, mining_func, accuracy_calculator=accuracy_calculator)
            df = pd.concat([df, result], ignore_index=True, sort=False)

    df.to_csv(outputFolder + "performance" + (str(layers[1]) if len(layers) > 1 else str(0)) + ".csv")
    create_convergence_graph.create_fold_convergence_graph(outputFolder + "performance" + (str(layers[1]) if len(layers) > 1 else str(0)) + ".csv", outputFolder)

def noCrossvalidation(layers, device, loss_func, num_epochs, dataset, batch_size):
    train_dataset = get_data(webDriveFolder + "feature_sets/train_seq_128.csv")
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = get_data(webDriveFolder + "feature_sets/test_seq_128.csv")
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Net(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, train_data_loader, test_data_loader, device, optimizer, loss_func, num_epochs, 0)

def main(outputPath: str, dataset:pd.DataFrame, layers: list, testset=None, batch_size=128, num_epochs=1, seed=42, learning_rate=0.01):
    global outputFolder
    outputFolder = outputPath

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    # distance = distances.CosineSimilarity()

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.AvgNonZeroReducer()
    loss_func = losses.TripletMarginLoss(margin=0.4, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all")
    # loss_func = losses.ContrastiveLoss(0, 1, distance=distance, reducer=reducer)

    # loss_func = losses.CosFaceLoss(2, 2, margin=0.35, scale=64)

    accuracy_calculator = customAccuracyCalculator.CustomCalculator(include=("mean_average_precision_at_r",), k=None)

    tr_dataset = create_customDataset(dataset)
    layers = [tr_dataset.data.shape[1]] + layers

    # noCrossvalidation(layers, device, loss_func, num_epochs, tr_dataset, batch_size)
    crossvalidation(layers, device, loss_func, num_epochs, tr_dataset, batch_size, seed, learning_rate, mining_func, accuracy_calculator)

if __name__ == "__main__":
    main()
