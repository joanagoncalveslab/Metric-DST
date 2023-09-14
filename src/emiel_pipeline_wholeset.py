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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import collections
import matplotlib.pyplot as plt
import pickle as pkl

from pytorch_metric_learning import distances, losses, reducers

from network import Network
from divergence import SelfTraining
from metric_learning import MetricLearning
import create_convergence_graph

from src import config
from src.andrei import biases as a_bias
from src.emiel import bias as e_bias
from src.dbast import bias_techniques as y_bias
from src.andrei import dataset_generator as a_dg
from src.custom import dataset_generator as c_dg
from src.dbast import load_dataset as y_dg

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = config.RESULT_DIR / 'bias' / 'emiel'
else:
    webDriveFolder = "/home/nfs/ytepeli/python_projects/msc-thesis-2122-mathijs-de-wolf/data"
    outputFolder = config.RESULT_DIR / 'bias' / 'emiel'


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
    parser.add_argument("--batch-size", "-bs", type=int, default=64)
    parser.add_argument('--dataset', '-d', default='emiel_10000|5|3')
    parser.add_argument("--output-file", "-of", type=str, default=None)
    parser.add_argument("--experiment_id", "-eid", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--conf", "-c", type=float, default=0.9)
    parser.add_argument("--num-pseudolabels", type=int, default=-2)

    parser.add_argument("--retrain", "-rt", action='store_true')
    parser.add_argument("--early_stop_pseudolabeling", "-esp", action='store_true')
    parser.add_argument("--single-fold", action='store_true')
    parser.add_argument("--model", "-m", choices=['supervised', 'balanced_semi_supervised', 'true_semi_supervised', 'diversity', 'all', 'visualize'], default='all')
    parser.add_argument("--bias", "-b", default='emiel_4|4_1000|1000|1000')
    parser.add_argument("--balance", "-bal", default='none')

    return parser

def main_visualize(data_dict, args):
    import umap

    total_number_of_folds = 10
    # ------------ Visualize -------------
    for fold, fold_dict in data_dict.items():
        plt.clf()
        plt.figure()
        reducer = umap.UMAP()
        print(f'fold: {fold}')

        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        
        all_data = np.concatenate([X_b_train, X_unlabeled, X_test])
        if all_data.shape[1]==2:
            embedding = all_data.copy()
        else:
            embedding = reducer.fit_transform(all_data)
        tr_emb = embedding[:len(X_b_train),:]
        unl_emb = embedding[len(X_b_train):len(X_b_train)+len(X_unlabeled),:]
        te_emb = embedding[len(X_b_train)+len(X_unlabeled):,:]
        
        plt.scatter(unl_emb[:, 0], unl_emb[:,1], c="gray", s=15, label='Unlabeled', marker = '.')
        #plt.scatter(unl_emb[y_unlabeled == 0, 0], unl_emb[y_unlabeled == 0, 1], c="#EE99AA", s=15, label='Negative', marker = '.')
        #plt.scatter(unl_emb[y_unlabeled == 1, 0], unl_emb[y_unlabeled == 1, 1], c="#6699CC", s=15, label='Positive', marker = '.')
        
        plt.scatter(tr_emb[y_b_train == 0, 0], tr_emb[y_b_train == 0, 1], c="#EE99AA", s=15, label='Train-Negative', marker = 'o')
        plt.scatter(tr_emb[y_b_train == 1, 0], tr_emb[y_b_train == 1, 1], c="#6699CC", s=15, label='Train-Positive', marker = 'o')
        
        plt.scatter(te_emb[y_test == 0, 0], te_emb[y_test == 0, 1], c="#EE99AA", s=15, label='Test-Negative', marker = 'x')
        plt.scatter(te_emb[y_test == 1, 0], te_emb[y_test == 1, 1], c="#6699CC", s=15, label='Test-Positive', marker = 'x')
        
        out_loc = outputFolder / 'bias_vis' / f'{fold}'
        config.ensure_dir(out_loc)
        #vis2d(embedding, y_train, x_ind, out_loc)
        plt.axis('off')
        plt.legend()
        png_loc = f'{out_loc}.png'
        config.ensure_dir(png_loc)
        plt.savefig(png_loc, dpi=300, bbox_inches='tight')

        

def main_divergence(data_dict, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()

    total_number_of_folds = 10
    # ------------ Divergence -------------
    #skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    for fold, fold_dict in data_dict.items():
        print(f'fold: {fold}')

        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        
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

def main_balanced_semi_supervised(data_dict, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()

    total_number_of_folds = 10
    # ------------ Divergence -------------
    #skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    for fold, fold_dict in data_dict.items():
        print(f'fold: {fold}')

        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
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

def main_true_semi_supervised(data_dict, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()

    total_number_of_folds = 10
    # ------------ Divergence -------------
    #skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    for fold, fold_dict in data_dict.items():
        print(f'fold: {fold}')
            
        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
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

def main_supervised(data_dict, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()

    total_number_of_folds = 10
    # ------------ Divergence -------------
    #skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    for fold, fold_dict in data_dict.items():
        print(f'fold: {fold}')
            
        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        
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

        '''
            Begin RFClassifier
        '''
        clf = RandomForestClassifier(n_estimators=400, random_state=0)
        clf.fit(train_dataset.drop(columns=['gene1', 'gene2', 'class']).values, train_dataset['class'].values)
        train_acc = clf.score(train_dataset.drop(columns=['gene1', 'gene2', 'class']).values, 
              train_dataset['class'].values)
        val_acc = clf.score(validation_dataset.drop(columns=['gene1', 'gene2', 'class']).values, 
              validation_dataset['class'].values)
        test_acc = clf.score(fold_test_dataset.drop(columns=['gene1', 'gene2', 'class']).values, 
              fold_test_dataset['class'].values)
        print(f'Acc -> Train: {train_acc}\tVal: {val_acc}\tTest: {test_acc}')
        
        '''End'''
        
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
    experiment_args = {'bs': args.batch_size, 'rs':args.seed, 'lr': args.lr, 'knn':args.knn, 'c':args.conf, 
                       'kb': args.num_pseudolabels, 'bal': args.balance}
    if args.retrain:
        experiment_args['rt']=''
    if args.early_stop_pseudolabeling:
        experiment_args['esp']=''
    experiment_str = "_".join([f'{exp_k}{exp_v}' for exp_k, exp_v in experiment_args.items()])
    
    
    custom_args_dict = {'ns': 'n_samples', 'nc': 'n_clusters', 'nf': 'n_features', 'ri': 'r_informative', 'rs': 'random_state', 'fy': 'flip_y', 'cs':'class_sep'}
    if 'custom' in args.dataset:
        custom_ds_args = {}
        for data_arg in args.dataset.split('_')[1:]:
            data_abbr = data_arg[:2]
            data_val = float(data_arg[2:]) if '.' in data_arg[2:] else int(data_arg[2:])
            custom_ds_args[custom_args_dict[data_abbr]] = data_val
        X, y = c_dg.custom_mdimension(**custom_ds_args)
    
    bias_name, bias_sizes, bias_arg_params = args.bias.split('_')
    if bias_name == 'concept':
        bias = {'name': bias_name, 
                'n_global': bias_sizes.split('|')[0], 'n_source': bias_sizes.split('|')[1], 'n_target': bias_sizes.split('|')[2],
                'n_domains': bias_arg_params.split('|')[0], 'trans': bias_arg_params.split('|')[1]}
    elif bias_name == 'covariate':
        bias = {'name': bias_name, 
                'n_global': bias_sizes.split('|')[0], 'n_source': bias_sizes.split('|')[1], 'n_target': bias_sizes.split('|')[2],
                'source_scale': bias_arg_params.split('|')[0], 'target_scale': bias_arg_params.split('|')[0], 
                'bias_dist': bias_arg_params.split('|')[1]}
    bias_params = collections.OrderedDict()
    for key, val in bias.items():
        if 'name' not in key and 'y' != key:
            if val.isnumeric():
                bias_params[key] = int(val)
            elif val.replace(".", "").isnumeric():
                bias_params[key] = float(val)
            else:
                bias_params[key] = val
    
    #bias_param_str = "|".join([str(i) for i in bias_params.values()]).replace(" ", "").replace(",", "_")
    #bias_folder_name = f'{bias["name"]}_{bias_param_str}'
    bias_folder_name = args.bias
    
    if args.output_file:   
        outputFolder = outputFolder / bias_folder_name / f'{args.dataset}' / args.output_file 
    elif args.experiment_id != -1:
        outputFolder = outputFolder / bias_folder_name / f'{args.dataset}' / f'experiment-{args.experiment_id}'
    else:
        outputFile = f"experiment-{experiment_str}-"
        i = 0
        #while os.path.exists(outputFolder / bias_folder_name /  f'{args.dataset}' / f'{outputFile}{i}'):
        #    i += 1
        outputFolder = outputFolder / bias_folder_name / f'{args.dataset}' / f'{outputFile}{i}'

    config.ensure_dir(outputFolder)
    config.safe_create_dir(outputFolder)

    if args.lr > 1:
        args.lr = 1 / args.lr

    while args.conf > 1:
        args.conf = args.conf / 10

    dataset_name = args.dataset
    
    if args.num_pseudolabels < -2:
        provisional_no = np.power(bias['n_source'], -1/args.num_pseudolabels) 
        args.num_pseudolabels = provisional_no - provisional_no%2 
    
    with open(outputFolder / 'settings.txt', 'a+') as f:
        f.write('\n'.join([
            str(outputFolder),
            'experiment id: '+str(args.experiment_id),
            'batch size: '+str(args.batch_size),
            'dataset: '+str(args.dataset),
            'bias_name: '+str(bias_name),
            'balance_strategy: '+str(args.balance),
            'learning rate: '+str(args.lr),
            'seed: '+str(args.seed),
            'knn: '+str(args.knn),
            'confidence: '+str(args.conf),
            'number of pseudolabels added per round: '+str(args.num_pseudolabels),
            'retrain model from scratch during pseudolabeling: '+str(args.retrain),
            'add pseudolabels until convergence: '+str(args.early_stop_pseudolabeling),
            f'bias: {bias}',
            f'bias_params: {bias_params}',
            f'args: {args}',
            'Note: This is biased by DBaST. CV is used.'
            '\n'
        ]))

    
    setup_seed(args.seed)
    total_number_of_folds = 10
    fold_dict = collections.OrderedDict()
    fold_loc = outputFolder / 'data' / 'data_folds.pkl'
    config.ensure_dir(fold_loc)
    if os.path.exists(fold_loc):
        with open(fold_loc, 'rb') as f:
            fold_dict = pkl.load(f)
    else:
        for fold_i in range(total_number_of_folds):
            X_unlabeled, y_unlabeled, X_b_train, y_b_train, X_test, y_test = e_bias.call_bias(X.copy(), y.copy(), bias, bias_params)
            fold_dict[fold_i] = {'xg': X_unlabeled.copy(), 'yg': y_unlabeled.copy(),
                                'xs': X_b_train.copy(), 'ys': y_b_train.copy(),
                                'xt': X_test.copy(), 'yt': y_test.copy()}
            with open(fold_loc, 'wb') as f:
                pkl.dump(fold_dict, f)
            
    setup_seed(args.seed)
    gc.collect()

    if args.model == 'visualize' or args.model == 'all':
        try:
            main_visualize(fold_dict, args)
        except Exception as e:
            print(f'main_visualize could not run! Exception {e} occured!')
    if args.model == 'diversity' or args.model == 'all':
        try:
            main_divergence(fold_dict, args)
        except Exception as e:
            print(f'main_divergence could not run! Exception {e} occured!')
    if args.model == 'balanced_semi_supervised' or args.model == 'all':
        try:
            main_balanced_semi_supervised(fold_dict, args)
        except Exception as e:
            print(f'main_balanced_semi_supervised could not run! Exception {e} occured!')
    if args.model == 'true_semi_supervised' or args.model == 'all':
        try:
            main_true_semi_supervised(fold_dict, args)
        except Exception as e:
            print(f'main_true_semi_supervised could not run! Exception {e} occured!')
    if args.model == 'supervised' or args.model == 'all':
        try:
            main_supervised(fold_dict, args)
        except Exception as e:
            print(f'main_supervised could not run! Exception {e} occured!')
