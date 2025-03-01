import pandas as pd
import numpy as np
import platform
import os
import network
import metric_learning
from pytorch_metric_learning import distances, losses, miners, reducers
import torch
from sklearn.model_selection import StratifiedKFold
import create_convergence_graph

def undersample(idx, labels):
    idx_0 = [id for id in idx if labels[id]==0]
    idx_1 = [id for id in idx if labels[id]==1]
    if len(idx_0) < len(idx_1):
        idx_1 = np.random.choice(idx_1, len(idx_0), replace=False)
    if len(idx_0) > len(idx_1):
        idx_0 = np.random.choice(idx_0, len(idx_1), replace=False)
    return np.concatenate([idx_0, idx_1])


if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = "/home/nfs/ytepeli/python_projects/msc-thesis-2122-mathijs-de-wolf/output/"

dataPath = "train_seq_128.csv"
if not os.path.exists(dataPath):
    dataPath = webDriveFolder + dataPath
    if not os.path.exists(dataPath):
        raise FileNotFoundError('The dataset does not exist')

outputFile = "experiment-"
i = 0
while os.path.exists(outputFolder + outputFile + str(i)):
    i += 1
outputFolder = outputFolder + outputFile + str(i) + '/'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

dataset = pd.read_csv(dataPath).fillna(0)
dataset = dataset[dataset['cancer']=="BRCA"]
dataset = dataset[(dataset['seq1']!=-1.0) & (dataset['seq1']!=0.0)]

idx_0 = dataset[dataset['class'] == 0]
idx_1 = dataset[dataset['class'] == 1]
if len(idx_0) < len(idx_1):
    idx_1 = idx_1.sample(len(idx_0))
if len(idx_0) > len(idx_1):
    idx_0 = idx_0.sample(len(idx_1))
dataset = pd.concat([idx_0, idx_1])
portion = round(len(dataset)/10)
train_dataset = pd.concat([dataset[dataset['class'] == 0].iloc[portion:, :], dataset[dataset['class'] == 1].iloc[portion:, :]])
test_dataset = pd.concat([dataset[dataset['class'] == 0].iloc[:portion, :], dataset[dataset['class'] == 1].iloc[:portion, :]])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
device = torch.device(device)

train_dataset = metric_learning.create_customDataset(train_dataset)
test_dataset = metric_learning.create_customDataset(test_dataset)

distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
reducer = reducers.MeanReducer()

splits = StratifiedKFold(5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)), train_dataset.labels)):
    print(fold)

    train_idx = undersample(train_idx, train_dataset.labels)
    val_idx = undersample(val_idx, train_dataset.labels)

    loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

    ntwrk = network.Network([128,8,2], loss_func, 0.01, device)
    ml = metric_learning.MetricLearning(ntwrk, test_dataset, train_dataset, train_idx, val_idx, outputFolder)
    ml.train(fold)

create_convergence_graph.create_fold_convergence_graph(outputFolder + "performance.csv", outputFolder)