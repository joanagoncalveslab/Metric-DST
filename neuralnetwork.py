import platform
import numpy as np
import torch
import torch.nn as nn

import customDataset
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

class EarlyStop():
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.min_val_loss = np.Inf
        self.counter = 0
        self.early_stop = False
        self.model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.min_val_loss - self.delta:
            self.min_val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter+=1
            if self.counter > self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        self.model_state = model.state_dict()

    def load_checkpoint(self):
        return self.model_state

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
        self.finalLayer = nn.Linear(layers[-1], 1, bias=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.finalLayer(x)
        x = torch.sigmoid(x)
        return x

def train_epoch(model, data_loader, device, optimizer, loss_func, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(torch.squeeze(embeddings), torch.squeeze(labels))
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

def train(model, train_loader, test_loader, device, optimizer, loss_func, num_epochs, fold):
    metrics = []
    # early_stop = EarlyStop(10)
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, device, optimizer, loss_func, epoch)
        test_metrics = test_epoch(model, device, test_loader, num_epochs==epoch, loss_func)
        metrics.append([epoch, fold, tr_loss, *test_metrics])
        # early_stop(test_metrics[0], model)
        # if early_stop.early_stop:
        #     print(f"Early stop after epoch: {epoch}")
        #     break
    df = pd.DataFrame(data=metrics, columns=['epoch', 'fold', 'train_loss', 'test_loss', 'auprc', 'auroc', 'accuracy', 'f1', 'average_precision'])
    return df

def get_data(path):
    file = pd.read_csv(path).fillna(0)
    data = file.iloc[:, 4:]
    labels = file["class"]
    dataset = customDataset.CustomDataset(torch.Tensor(data.values.astype(np.float64)), torch.Tensor(labels.values.astype(np.float64)))
    return dataset

def create_customDataset(data):
    features = data.iloc[:, 4:]
    labels = data["class"]
    dataset = customDataset.CustomDataset(torch.Tensor(features.values.astype(np.float64)), torch.Tensor(labels.values.astype(np.float64)))
    return dataset

def crossvalidation(layers, device, loss_func, num_epochs, dataset, batch_size, seed, learning_rate):
    df = pd.DataFrame()
    splits = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)), dataset.labels)):
        print("fold: {}".format(fold))

        # TODO: Sample here!
        # Just undersample the training and validation set

        g_train = torch.Generator().manual_seed(seed+fold)
        g_val = torch.Generator().manual_seed(seed+fold)
        train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx, generator=g_train))
        test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx, generator=g_val))

        model = Net(layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        result = train(model, train_data_loader, test_data_loader, device, optimizer, loss_func, num_epochs, fold)
        df = pd.concat([df, result], ignore_index=True, sort=False)

    df.to_csv(outputFolder + "performance" + (str(layers[1]) if len(layers) > 1 else str(0)) + ".csv")

def noCrossvalidation(layers, device, loss_func, num_epochs, dataset, batch_size):
    train_dataset = get_data(webDriveFolder + "feature_sets/train_seq_128.csv")
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = get_data(webDriveFolder + "feature_sets/test_seq_128.csv")
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Net(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, train_data_loader, test_data_loader, device, optimizer, loss_func, num_epochs, 0)

def main(outputPath: str, dataset:pd.DataFrame, layers: list, testset=None, batch_size=128, num_epochs=1, seed=42, learning_rate=0.1):
    global outputFolder
    outputFolder = outputPath

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    loss_func = torch.nn.BCELoss()

    tr_dataset = create_customDataset(dataset)
    layers = [tr_dataset.data.shape[1]] + layers

    # noCrossvalidation(layers, device, loss_func, num_epochs, tr_dataset, batch_size)
    crossvalidation(layers, device, loss_func, num_epochs, tr_dataset, batch_size, seed, learning_rate)

if __name__ == "__main__":
    main()
