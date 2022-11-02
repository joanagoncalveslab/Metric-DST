import platform
import os, random
from re import T

import pandas as pd
import numpy as np
import torch

from pytorch_metric_learning import distances, losses, miners, reducers

from network import Network
from divergence import SelfTraining
import create_convergence_graph

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"


def undersample(idx, labels):
    idx_0 = [id for id in idx if labels[id]==0]
    idx_1 = [id for id in idx if labels[id]==1]
    if len(idx_0) < len(idx_1):
        idx_1 = np.random.choice(idx_1, len(idx_0), replace=False)
    if len(idx_0) > len(idx_1):
        idx_0 = np.random.choice(idx_0, len(idx_1), replace=False)
    return np.concatenate([idx_0, idx_1])

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    dataPath = "train_seq_128_BRCA.csv"
    if not os.path.exists(dataPath):
        dataPath = webDriveFolder + dataPath
        if not os.path.exists(dataPath):
            raise FileNotFoundError('The dataset does not exist')
    unlabeledDatapath = webDriveFolder + "unknown_repair_cancer_BRCA_seq_128.csv"
    testDatapath = webDriveFolder + "test_seq_128.csv"

    outputFile = "experiment-"
    i = 0
    while os.path.exists(outputFolder + outputFile + str(i)):
        i += 1
    outputFolder = outputFolder + outputFile + str(i) + '/'

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    setup_seed(42)

    dataset = pd.read_csv(dataPath, index_col=0).fillna(0)
    # dataset = dataset[dataset['cancer']=="BRCA"]
    dataset = dataset[(dataset['seq1']!=-1.0) & (dataset['seq1']!=0.0)]

    unlabeled_dataset = pd.read_csv(unlabeledDatapath)
    unlabeled_dataset = unlabeled_dataset[unlabeled_dataset['cancer'] == "BRCA"]
    unlabeled_dataset = unlabeled_dataset[(unlabeled_dataset['seq1']!=-1.0) & (unlabeled_dataset['seq1']!=0.0)]

    test_dataset = pd.read_csv(testDatapath)
    test_dataset = test_dataset[test_dataset['cancer'] == "BRCA"]
    test_dataset = test_dataset[(test_dataset['seq1']!=-1.0) & (test_dataset['seq1']!=0.0)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    
    for fold in range(5):
        print(f'fold: {fold}')

        # Undersample dataset
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if len(idx_0) < len(idx_1):
            idx_1 = idx_1.sample(len(idx_0))
        if len(idx_0) > len(idx_1):
            idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if len(idx_0) < len(idx_1):
            idx_1 = idx_1.sample(len(idx_0))
        if len(idx_0) > len(idx_1):
            idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)
        
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        ntwrk = Network([128,8,2], loss_func, 0.01, device)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, outputFolder, 10)
        ml.train(fold)
        
    create_convergence_graph.create_fold_convergence_graph(outputFolder + "performance.csv", outputFolder)