from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    data:Tensor
    labels:Tensor
    labels: list

    def __init__(self, data:Tensor, labels:Tensor, genes:list):
        self.data = data
        self.labels = labels
        self.genes = genes

    def __len__(self)->int:
        return len(self.labels)

    def __getitem__(self, idx:int)->Tuple[Tensor, Tensor, list, int]:
        return self.data[idx], self.labels[idx], self.genes[idx], idx
