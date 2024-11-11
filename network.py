from typing import Tuple
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners

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

class Network():
    model:Net
    mining_func:miners.BaseMiner
    loss_func:losses.BaseMetricLossFunction
    optimizer: torch.optim.Optimizer
    device:torch.device


    def __init__(self, layers:list, loss_func:losses.BaseMetricLossFunction, learning_rate:float, device:torch.device, mining_func:miners.BaseMiner=None):
        self.model = Net(layers).to(device)
        self.mining_func = mining_func
        self.loss_func = loss_func
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device

    def train(self, data_loader:DataLoader)->float:
        self.model.train()
        total_loss = 0
        for batch_idx, (data, labels, _, _) in enumerate(data_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            embeddings = self.model(data)
            if self.mining_func is not None:
                indices_tuple = self.mining_func(embeddings, labels)
                loss = self.loss_func(torch.squeeze(embeddings), torch.squeeze(labels), indices_tuple)
            else:
                try:
                    loss = self.loss_func(torch.squeeze(embeddings), torch.squeeze(labels))
                except:
                    print(torch.squeeze(embeddings[0]))
                    print(torch.squeeze(labels))
            total_loss += loss.item()*data.size(0)
            loss.backward()
            self.optimizer.step()
        return total_loss/len(data_loader.sampler)

    def evaluate(self, data_loader:DataLoader)->Tuple[float, torch.Tensor, torch.Tensor, list]:
        self.model.eval()
        loss = 0
        labels = torch.zeros(0, dtype=torch.long, device='cpu')
        embeddings = torch.zeros(0, dtype=torch.long, device='cpu')
        genes = []
        with torch.no_grad():
            for data, target, (gene1, gene2), _ in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                genes.extend(list(zip(gene1, gene2)))
                labels = torch.cat([labels, target.cpu()])
                output = self.model(data)
                loss += self.loss_func(torch.squeeze(output), torch.squeeze(target)).item()*data.size(0)
                embeddings = torch.cat([embeddings, output.cpu()])
        loss /= len(data_loader.sampler)
        return loss, embeddings, labels, genes
    
    def run(self, data_loader:DataLoader)->Tuple[torch.Tensor, torch.Tensor, list]:
        self.model.eval()
        labels = torch.zeros(0, dtype=torch.long, device='cpu')
        embeddings = torch.zeros(0, dtype=torch.long, device='cpu')
        genes = []
        with torch.no_grad():
            for data, target, (gene1, gene2), _ in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                genes.extend(list(zip(gene1, gene2)))
                labels = torch.cat([labels, target.cpu()])
                output = self.model(data)
                embeddings = torch.cat([embeddings, output.cpu()])
        return embeddings, labels, genes

    def run_with_idx(self, data_loader:DataLoader)->Tuple[torch.Tensor, torch.Tensor, list, list]:
        self.model.eval()
        labels = torch.zeros(0, dtype=torch.long, device='cpu')
        embeddings = torch.zeros(0, dtype=torch.long, device='cpu')
        genes = []
        idx = []
        with torch.no_grad():
            for data, target, (gene1, gene2), id in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                genes.extend(list(zip(gene1, gene2)))
                idx.extend(id)
                labels = torch.cat([labels, target.cpu()])
                output = self.model(data)
                embeddings = torch.cat([embeddings, output.cpu()])
        return embeddings, labels, genes, idx
