from random import sample
import customDataset
import torch
import pandas as pd


file = pd.read_csv("W:\\staff-umbrella\\JGMasters\\2122-mathijs-de-wolf\\feature_sets\\test_seq_128.csv")
print(file[file['class']==1].shape)

data = file.iloc[:, 4:]
labels = file["class"]

dataset = customDataset.CustomDataset(data.to_numpy(), labels.to_numpy())
print(dataset.data.dtype)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# print(labels.head())

sample_idx = torch.randint(len(dataset), size=(1,)).item()
data, labels = dataset[sample_idx]
print(data)
