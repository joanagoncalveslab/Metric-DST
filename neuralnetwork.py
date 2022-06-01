import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import customDataset
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score, average_precision_score

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 1, bias=True)
        # self.fc2 = nn.Linear(128, 128, bias=True)
        # self.fc3 = nn.Linear(128, 1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        # x = self.fc2(x)
        # x = torch.sigmoid(x)
        # x = self.fc3(x)
        # x = torch.sigmoid(x)
        return x

def train_round(model, data_loader, device, optimizer, loss_func, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(torch.squeeze(embeddings), labels)
        total_loss += loss.item()
        loss = loss/len(labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )
    return total_loss/len(data_loader.dataset)

def test(model, device, test_loader, last_round):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = torch.zeros(0, dtype=torch.long, device='cpu')
    y_true = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_true = torch.cat([y_true, target.view(-1).cpu()])
            output = model(data)
            test_loss += F.binary_cross_entropy(torch.squeeze(output), target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            y_pred = torch.cat([y_pred, pred.view(-1).cpu()])
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if last_round:
        classes = ['negative', 'positive']
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=classes, columns=classes)
        df_cm.to_csv(outputFolder + 'confusion_matrix.csv')
    precission, recall, threshholds = precision_recall_curve(y_true, y_pred)
    auprc = auc(precission, recall)
    auroc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return [test_loss, auprc, auroc, accuracy, f1, average_precision]

def train(model, train_loader, test_loader, device, optimizer, loss_func, num_epochs):
    metrics = []
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_round(model, train_loader, device, optimizer, loss_func, epoch)
        test_metrics = test(model, device, test_loader, num_epochs==epoch)
        metrics.append([tr_loss, *test_metrics])
    df = pd.DataFrame(data=metrics, columns=['train_loss', 'test_loss', 'auprc', 'auroc', 'accuracy', 'f1', 'average_precision'])
    df.to_csv(outputFolder + "performance.csv")

def get_data(path):
    file = pd.read_csv(path).fillna(0)
    data = file.iloc[:, 4:]
    labels = file["class"]
    dataset = customDataset.CustomDataset(torch.Tensor(data.values.astype(np.float64)), torch.Tensor(labels.values.astype(np.float64)))
    return dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    batch_size = 256
    num_epochs = 1

    train_dataset = get_data(webDriveFolder + "feature_sets/train_seq_128.csv")
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = get_data(webDriveFolder + "feature_sets/test_seq_128.csv")
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Net(train_dataset.data.shape[1]).to(device)
    loss_func = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train(model, train_data_loader, test_data_loader, device, optimizer, loss_func, num_epochs)

if __name__ == "__main__":
    main()
