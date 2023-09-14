import argparse
import platform
import os, sys, random, gc
project_path = '.'
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'dssl4sl':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)

import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing

from pytorch_metric_learning import distances, losses, reducers

from network import Network
from divergence import SelfTraining
from metric_learning import MetricLearning
import create_convergence_graph

from src import config
from src import load_dataset as ld
from src import bias_techniques as bt

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = config.RESULT_DIR / 'bias'
else:
    webDriveFolder = "/home/nfs/ytepeli/python_projects/msc-thesis-2122-mathijs-de-wolf/data"
    outputFolder = config.RESULT_DIR / 'bias'


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
        description="Run metric learning pipeline"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument('-D', '--dataset', choices=['fire', 'pistachio', 'pumpkin', 'raisin', 'rice', 'abalone', 'yeast', 
                                                    'spam', 'adult', 'drug', 'breast_cancer'], default='breast_cancer')
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--experiment_id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--num-pseudolabels", type=int, default=6)

    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--early_stop_pseudolabeling", action='store_true')
    parser.add_argument("--single-fold", action='store_true')
    parser.add_argument("--model", choices=['supervised', 'balanced_semi_supervised', 'true_semi_supervised', 'diversity', 'all'], default='diversity')
    parser.add_argument("--bias", default='hierarchyy_0.9')
    parser.add_argument("--balance", default='none')

    return parser

def main(dataset, test_dataset, unlabeled_dataset, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10

    # --------------- Divergence ---------------
    for fold in range(total_number_of_folds):
        print(f'fold: {fold}')

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)
        
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, args.lr, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "diversity", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, retrain=args.retrain, add_samples_to_convergence=args.early_stop_pseudolabeling, pseudolabel_method='divergence')
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "diversity_performance.csv"), outputFolder)

    # ------------ Balanced Semi supervised -------------
    for fold in range(total_number_of_folds):
        print(f'fold: {fold}')

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)
        
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, args.lr, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "balanced_semi_supervised", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, retrain=args.retrain, add_samples_to_convergence=args.early_stop_pseudolabeling, pseudolabel_method='balanced_semi_supervised')
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "balanced_semi_supervised_performance.csv"), outputFolder)

    # ------------ True Semi supervised -------------
    for fold in range(total_number_of_folds):
        print(f'fold: {fold}')

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)
        
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, args.lr, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "true_semi_supervised", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, retrain=args.retrain, add_samples_to_convergence=args.early_stop_pseudolabeling, pseudolabel_method='semi_supervised')
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "true_semi_supervised_performance.csv"), outputFolder)

    # --------------- Supervised ---------------
    for fold in range(total_number_of_folds):
        print(fold)

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)

        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, 0.01, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "supervised", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, supervised=True)
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "supervised_performance.csv"), outputFolder)

def main_divergence(dataset, test_dataset, unlabeled_dataset, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10

    # --------------- Divergence ---------------
    for fold in range(total_number_of_folds):
        print(f'fold: {fold}')

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)
        
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, args.lr, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "diversity", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, retrain=args.retrain, add_samples_to_convergence=args.early_stop_pseudolabeling, pseudolabel_method='divergence')
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "diversity_performance.csv"), outputFolder)

def main_balanced_semi_supervised(dataset, test_dataset, unlabeled_dataset, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10
    # ------------ Balanced Semi supervised -------------
    for fold in range(total_number_of_folds):
        print(f'fold: {fold}')

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)
        
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, args.lr, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "balanced_semi_supervised", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, retrain=args.retrain, add_samples_to_convergence=args.early_stop_pseudolabeling, pseudolabel_method='balanced_semi_supervised')
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "balanced_semi_supervised_performance.csv"), outputFolder)

def main_true_semi_supervised(dataset, test_dataset, unlabeled_dataset, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10
    # ------------ True Semi supervised -------------
    for fold in range(total_number_of_folds):
        print(f'fold: {fold}')

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)
        
        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, args.lr, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "true_semi_supervised", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, retrain=args.retrain, add_samples_to_convergence=args.early_stop_pseudolabeling, pseudolabel_method='semi_supervised')
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "true_semi_supervised_performance.csv"), outputFolder)

def main_supervised(dataset, test_dataset, unlabeled_dataset, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10
    # --------------- Supervised ---------------
    for fold in range(total_number_of_folds):
        print(fold)

        # Undersample dataset
        setup_seed(fold)
        idx_0 = dataset[dataset['class'] == 0]
        idx_1 = dataset[dataset['class'] == 1]
        if 'train' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        # Split dataset in validation and train set
        partion = round((len(idx_0)+len(idx_1))/10)
        train_dataset = pd.concat([idx_0.iloc[partion:, :], idx_1.iloc[partion:, :]], ignore_index=True)
        validation_dataset = pd.concat([idx_0.iloc[:partion, :], idx_1.iloc[:partion, :]], ignore_index=True)
        # Shuffle datasets, otherwise the first half is negative and the second half is positive
        setup_seed(fold)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
        setup_seed(fold)
        validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

        # Undersample test set
        setup_seed(fold)
        idx_0 = test_dataset[test_dataset['class'] == 0]
        idx_1 = test_dataset[test_dataset['class'] == 1]
        if 'test' in args.balance:
            if len(idx_0) < len(idx_1):
                idx_1 = idx_1.sample(len(idx_0))
            if len(idx_0) > len(idx_1):
                idx_0 = idx_0.sample(len(idx_1))
        fold_test_dataset = pd.concat([idx_0, idx_1])
        setup_seed(fold)
        fold_test_dataset = fold_test_dataset.sample(frac=1).reset_index(drop=True)

        loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)

        setup_seed(fold)
        ntwrk = Network([dataset.shape[1]-3,8,2], loss_func, args.lr, device)
        setup_seed(fold)
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "supervised", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, supervised=True)
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "supervised_performance.csv"), outputFolder)

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    print(args)

    #dataPath = webDriveFolder + f"train_seq_128_{args.cancer}.csv"
    #unlabeledDatapath = webDriveFolder + f"unknown_repair_cancer_{args.cancer}_seq_128.csv"
    #testDatapath = webDriveFolder + f"test_seq_128.csv"

    if args.output_file:
        outputFolder = outputFolder / args.bias / args.dataset / args.output_file 
    elif args.experiment_id != -1:
        outputFolder = outputFolder / args.bias / args.dataset / f'experiment-{args.experiment_id}'
    else:
        outputFile = "experiment-"
        i = 0
        while os.path.exists(outputFolder / args.bias /  args.dataset / f'{outputFile}{i}'):
            i += 1
        outputFolder = outputFolder / args.bias / args.dataset / f'{outputFile}{i}'

    config.ensure_dir(outputFolder)
    config.safe_create_dir(outputFolder)

    if args.lr > 1:
        args.lr = 1 / args.lr

    while args.conf > 1:
        args.conf = args.conf / 10

    if args.bias == 'none':
        bias = {'name': None}
    if 'hierarchyy_' in args.bias:
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': 30, 'prob': float(args.bias.split('_')[1])}
        bias_params = {key: val for key, val in bias.items() if ('name' not in key) and ('y' != key)}
    
    with open(outputFolder / 'settings.txt', 'a+') as f:
        f.write('\n'.join([
            str(outputFolder),
            'experiment id: '+str(args.experiment_id),
            'batch size: '+str(args.batch_size),
            'dataset: '+str(args.dataset),
            'bias_name: '+str(args.bias),
            'balance_strategy: '+str(args.balance),
            'learning rate: '+str(args.lr),
            'seed: '+str(args.seed),
            'knn: '+str(args.knn),
            'confidence: '+str(args.conf),
            'number of pseudolabels added per round: '+str(args.num_pseudolabels),
            'retrain model from scratch during pseudolabeling: '+str(args.retrain),
            'add pseudolabels until convergence: '+str(args.early_stop_pseudolabeling),
            f'bias: {bias}',
            f'args: {args}',
            'Note: This is biased by DBaST. Test set is not balanced.'
            ''
        ]))

    
    setup_seed(args.seed)
    #dataset = pd.read_csv(dataPath, index_col=0).fillna(0)
    # dataset = dataset[dataset['cancer']=="BRCA"]
    #dataset = dataset[(dataset['seq1']!=-1.0) & (dataset['seq1']!=0.0)]
    X, y, X_test, y_test = ld.load_dataset(args.dataset, test=True)
    X_train, X_unlabeled, y_train, y_unlabeled = ld.split_dataset(X, y, 0.3, 0.7, r_seed=args.seed)
    if bias['name'] is None:
        X_b_train, y_b_train = X_train.copy(), y_train.copy()
    else:
        mask = np.ones_like(y_train, bool)
        selected_ids = bt.get_bias(bias['name'], X_train, y=y_train,
                                                   **bias_params).astype(int)
        print(selected_ids)
        print(X_train)
        print(y_train)
        X_b_train, y_b_train = X_train[selected_ids, :], y_train[selected_ids]
    
    
    zv_cols = np.var(X_b_train, axis=0)==0 #zero variance columns
    X_b_train, X_test, X_unlabeled = X_b_train[:, ~zv_cols], X_test[:, ~zv_cols], X_unlabeled[:, ~zv_cols]
    scaler = preprocessing.StandardScaler().fit(X_b_train)
    # scaler = MinMaxScaler().fit(x_b_train)
    X_b_train = scaler.transform(X_b_train)
    X_test = scaler.transform(X_test)
    X_unlabeled = scaler.transform(X_unlabeled)
    
    dataset = pd.DataFrame(X_b_train)
    base_gi = 0
    dataset['gene1'] = [f'gene1_{base_gi+gi}' for gi in range(dataset.shape[0])]
    dataset['gene2'] = [f'gene2_{base_gi+gi}' for gi in range(dataset.shape[0])]
    dataset['class'] = y_b_train
    test_dataset = pd.DataFrame(X_test)
    base_gi = dataset.shape[1]
    test_dataset['gene1'] = [f'gene1_{base_gi+gi}' for gi in range(test_dataset.shape[0])]
    test_dataset['gene2'] = [f'gene2_{base_gi+gi}' for gi in range(test_dataset.shape[0])]
    test_dataset['class'] = y_test
    unlabeled_dataset = pd.DataFrame(X_unlabeled)
    base_gi = test_dataset.shape[1]
    unlabeled_dataset['gene1'] = [f'gene1_{base_gi+gi}' for gi in range(unlabeled_dataset.shape[0])]
    unlabeled_dataset['gene2'] = [f'gene2_{base_gi+gi}' for gi in range(unlabeled_dataset.shape[0])]
    unlabeled_dataset['class'] = y_unlabeled

    gc.collect()

    if args.model == 'all': 
        main(dataset, test_dataset, unlabeled_dataset, args)
    elif args.model == 'diversity':
        main_divergence(dataset, test_dataset, unlabeled_dataset, args)
    elif args.model == 'balanced_semi_supervised':
        main_balanced_semi_supervised(dataset, test_dataset, unlabeled_dataset, args)
    elif args.model == 'true_semi_supervised':
        main_true_semi_supervised(dataset, test_dataset, unlabeled_dataset, args)
    elif args.model == 'supervised':
        main_supervised(dataset, test_dataset, unlabeled_dataset, args)
