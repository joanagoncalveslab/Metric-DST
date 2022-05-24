from random import sample
import customDataset
import torch
import pandas as pd


file = pd.read_csv("W:\\staff-umbrella\\JGMasters\\2122-mathijs-de-wolf\\feature_sets\\train_tissue.csv")
data = file.iloc[:, 4:]
labels = file["class"]

dataset = customDataset.CustomDataset(data.to_numpy(), labels.to_numpy())

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# print(labels.head())

sample_idx = torch.randint(len(dataset), size=(1,)).item()
data, labels = dataset[sample_idx]
print(data)
