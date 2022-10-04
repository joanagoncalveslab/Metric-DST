from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, genes):
        self.data = data
        self.labels = labels
        self.genes = genes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.genes[idx]
