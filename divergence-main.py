import argparse
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

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run self training model"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument('-C', '--cancer', choices=['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'SKCM', 'OV'], default='BRCA')
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--num-pseudolabels", type=int, default=50)

    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--early-stop-pseudolabeling", action='store_true')
    parser.add_argument("--single-fold", action='store_true')

    return parser

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    dataPath = f"train_seq_128_{args.cancer}.csv"
    if not os.path.exists(dataPath):
        dataPath = webDriveFolder + dataPath
        if not os.path.exists(dataPath):
            raise FileNotFoundError('The dataset does not exist')
    unlabeledDatapath = webDriveFolder + f"unknown_repair_cancer_{args.cancer}_seq_128.csv"
    testDatapath = webDriveFolder + f"test_seq_128.csv"

    if args.output_file:
        outputFolder = outputFolder + args.output_file + "/"
    else:
        outputFile = "experiment-"
        i = 0
        while os.path.exists(outputFolder + outputFile + str(i)):
            i += 1
        outputFolder = outputFolder + outputFile + str(i) + '/'

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    if args.lr > 1:
        args.lr = 1 / args.lr

    while args.conf > 1:
        args.conf = args.conf / 10

    with open(outputFolder + 'settings.txt', 'a') as f:
        f.write('\n'.join([
            outputFolder,
            'batch size: '+str(args.batch_size),
            'cancer: '+str(args.cancer),
            'learning rate: '+str(args.lr),
            'seed: '+str(args.seed),
            'knn: '+str(args.knn),
            'confidence: '+str(args.conf),
            'number of pseudolabels added per round: '+str(args.num_pseudolabels),
            'retrain model from scratch during pseudolabeling: '+str(args.retrain),
            'add pseudolabels until convergence: '+str(args.early_stop_pseudolabeling),
            ''
        ]))

    setup_seed(args.seed)

    dataset = pd.read_csv(dataPath, index_col=0).fillna(0)
    # dataset = dataset[dataset['cancer']=="BRCA"]
    dataset = dataset[(dataset['seq1']!=-1.0) & (dataset['seq1']!=0.0)]

    unlabeled_dataset = pd.read_csv(unlabeledDatapath)
    unlabeled_dataset = unlabeled_dataset[unlabeled_dataset['cancer'] == args.cancer]
    unlabeled_dataset = unlabeled_dataset[(unlabeled_dataset['seq1']!=-1.0) & (unlabeled_dataset['seq1']!=0.0)]

    test_dataset = pd.read_csv(testDatapath)
    test_dataset = test_dataset[test_dataset['cancer'] == args.cancer]
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

        ntwrk = Network([128,8,2], loss_func, args.lr, device)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, outputFolder, args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, retrain=args.retrain, add_samples_to_convergence=args.early_stop_pseudolabeling)
        if args.single_fold:
            break
    create_convergence_graph.create_fold_convergence_graph(outputFolder + "performance.csv", outputFolder)