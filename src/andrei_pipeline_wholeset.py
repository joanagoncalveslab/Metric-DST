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

from pytorch_metric_learning import distances, losses, reducers

from network import Network
from divergence import SelfTraining
from metric_learning import MetricLearning
import create_convergence_graph

from src import config
from src.andrei import biases as anbias
from src.andrei import dataset_generator as dg
from src import load_dataset as ld

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
    parser.add_argument("--batch-size", "-bs", type=int, default=64)
    parser.add_argument('--dataset', '-d', choices=['three_unbalanced', 'overlapped', 'rotated_moons', 'multidimension'], default='rotated_moons')
    parser.add_argument('-dd', '--dataset_dimension', type=int, default=50)
    parser.add_argument('-ss', '--sample_size', type=int, default=1000)
    parser.add_argument("--output-file", "-of", type=str, default=None)
    parser.add_argument("--experiment_id", "-eid", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--conf", "-c", type=float, default=0.8)
    parser.add_argument("--num-pseudolabels", type=int, default=6)

    parser.add_argument("--retrain", "-rt", action='store_true')
    parser.add_argument("--early_stop_pseudolabeling", "-esp", action='store_true')
    parser.add_argument("--single-fold", action='store_true')
    parser.add_argument("--model", "-m", choices=['supervised', 'balanced_semi_supervised', 'true_semi_supervised', 'diversity', 'all', 'visualize'], default='diversity')
    parser.add_argument("--bias", "-b", choices=['class_imbalance', 'reduce_samples', 'bias_2features', 'bias_most_important_feature', 'bias_multi_features', 'bias_balanced_multi_features', 'none'], default='bias_balanced_multi_features')
    parser.add_argument("--balance", "-bal", default='none')

    return parser


def array_row_intersection(a,b):
   tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
   bool_lst = np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)
   return np.arange(len(bool_lst))[bool_lst]

def vis2d(X, y, selected_indices, out_loc):
    import matplotlib.pyplot as plt
    plt.figure()
    class_size = len(np.unique(y))
    y_color = y.copy()
    for class_tmp in np.unique(y):
        selected_class = selected_indices[y[selected_indices] == class_tmp]
        y_color[selected_class] = class_tmp+class_size
    plt.scatter(X[y_color == 0, 0], X[y_color == 0, 1], c="#EE99AA", s=15, label='Negative')
    plt.scatter(X[y_color == 1, 0], X[y_color == 1, 1], c="#6699CC", s=15, label='Positive')
    plt.scatter(X[y_color == 2, 0], X[y_color == 2, 1], c="#994455", s=15, label=f'Negative_chosen')
    plt.scatter(X[y_color == 3, 0], X[y_color == 3, 1], c="#004488", s=15, label=f'Positive_chosen')
    
    plt.axis('off')
    plt.legend()
    png_loc = f'{out_loc}.png'
    config.ensure_dir(png_loc)
    # plt.show()
    plt.savefig(png_loc, dpi=300, bbox_inches='tight')


def main_visualize(X, y, bias, bias_params, args):
    import umap
    total_number_of_folds = 10
    skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    # --------------- Divergence ---------------
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'fold: {fold}')
        
        X_rest, y_rest, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
        X_train, X_unlabeled, y_train, y_unlabeled = ld.split_dataset(X_rest, y_rest, 0.3, 0.7, r_seed=args.seed+fold)
        reducer = umap.UMAP()
        #X_train = StandardScaler().fit_transform(X_train)
        if bias['name'] is None:
            X_b_train, y_b_train = X_train.copy(), y_train.copy()
        else:
            X_b_train, y_b_train = anbias.call_bias(X_train.copy(), y_train.copy(), bias['name'], **bias_params)
        X_train_tup = [tuple(arow) for arow in X_train]
        X_b_train_tup = [tuple(arow) for arow in X_b_train]
        X_diff_train = np.setdiff1d(X_train_tup,X_b_train_tup)
        #intersect, x_ind, x_b_ind = np.intersect1d(X_train_tup, X_b_train_tup, return_indices=True)
        x_ind = array_row_intersection(X_train, X_b_train)
        print(f'X_train({X_train.shape}) - X_b_train({X_b_train.shape}) - Diff({X_diff_train.shape}), Intsec({len(x_ind)})')
        if X_train.shape[1]==2:
            embedding = X_train.copy()
        else:
            embedding = reducer.fit_transform(X_train)
        out_loc = outputFolder / 'bias_vis' / f'{fold}'
        config.ensure_dir(out_loc)
        vis2d(embedding, y_train, x_ind, out_loc)
    

def main_divergence(X, y, bias, bias_params, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10
    skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    # --------------- Divergence ---------------
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'fold: {fold}')
        
        X_rest, y_rest, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
        X_train, X_unlabeled, y_train, y_unlabeled = ld.split_dataset(X_rest, y_rest, 0.3, 0.7, r_seed=args.seed+fold)
        if bias['name'] is None:
            X_b_train, y_b_train = X_train.copy(), y_train.copy()
        else:
            X_b_train, y_b_train = anbias.call_bias(X_train.copy(), y_train.copy(), bias['name'], **bias_params)
        
        
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

def main_balanced_semi_supervised(X, y, bias, bias_params, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10
    # ------------ Balanced Semi supervised -------------
    skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'fold: {fold}')
        
        X_rest, y_rest, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
        X_train, X_unlabeled, y_train, y_unlabeled = ld.split_dataset(X_rest, y_rest, 0.3, 0.7, r_seed=args.seed+fold)
        if bias['name'] is None:
            X_b_train, y_b_train = X_train.copy(), y_train.copy()
        else:
            X_b_train, y_b_train = anbias.call_bias(X_train.copy(), y_train.copy(), bias['name'], **bias_params)
        
        
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

def main_true_semi_supervised(X, y, bias, bias_params, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10
    # ------------ True Semi supervised -------------
    skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'fold: {fold}')
        
        X_rest, y_rest, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
        X_train, X_unlabeled, y_train, y_unlabeled = ld.split_dataset(X_rest, y_rest, 0.3, 0.7, r_seed=args.seed+fold)
        if bias['name'] is None:
            X_b_train, y_b_train = X_train.copy(), y_train.copy()
        else:
            X_b_train, y_b_train = anbias.call_bias(X_train.copy(), y_train.copy(), bias['name'], **bias_params)
        
        
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

def main_supervised(X, y, bias, bias_params, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    total_number_of_folds = 10
    # --------------- Supervised ---------------
    skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'fold: {fold}')
        
        X_rest, y_rest, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
        X_train, X_unlabeled, y_train, y_unlabeled = ld.split_dataset(X_rest, y_rest, 0.3, 0.7, r_seed=args.seed+fold)
        if bias['name'] is None:
            X_b_train, y_b_train = X_train.copy(), y_train.copy()
        else:
            X_b_train, y_b_train = anbias.call_bias(X_train.copy(), y_train.copy(), bias['name'], **bias_params)
        
        
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

    if args.dataset == 'three_unbalanced':
        X, y = dg.three_unbalanced_clusters()
        if args.bias == 'bias_2features':
            bias = {'name': args.bias, 'i':5, 'n_class_partitions':10, 'b_1':0.1, 'b_2':5, 'delta_1': (-7, 6), 'delta_2': (5, 5)} 
    elif args.dataset == 'overlapped':
        X, y = dg.overlapped_linear_clusters(args.sample_size, args.sample_size)
        if args.bias == 'bias_2features':
            bias = {'name': args.bias, 'i':5, 'n_class_partitions':10, 'b_1':0.5, 'b_2':0.1, 'delta_1': (0,0), 'delta_2': (-25, 0)}
    elif args.dataset == 'rotated_moons':
        X, y = dg.rotated_moons(args.sample_size)
        if args.bias == 'bias_2features':
            bias = {'name': args.bias, 'i':5, 'n_class_partitions':10, 'b_1':1.5, 'b_2':1.5, 'delta_1': (1, 0.5), 'delta_2':(0, 0)} 
    elif args.dataset == 'multidimension':
        X, y = dg.multidimensional_set(args.sample_size, args.dataset_dimension, args.seed)
        bias = {'name': args.bias, 'size':50, 'a':3, 'b':4} 
    else:
        print(f'The choice {args.dataset} does not exist!!!')
        raise Exception(f'The choice {args.dataset} does not exist!!!')


    if args.bias == 'none':
        bias = {'name': None}
    elif 'hierarchyy_' in args.bias:
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': 30, 'prob': float(args.bias.split('_')[1])}
    bias_params = {key: val for key, val in bias.items() if ('name' not in key) and ('y' != key)}
    bias_param_str = "|".join([str(i) for i in bias_params.values()]).replace(" ", "").replace(",", "_")
    bias_folder_name = f'andrei_{bias["name"]}_{bias_param_str}'
    if args.output_file:   
        outputFolder = outputFolder / bias_folder_name / f'{args.dataset}_{args.sample_size}|{args.dataset_dimension}' / args.output_file 
    elif args.experiment_id != -1:
        outputFolder = outputFolder / bias_folder_name / f'{args.dataset}_{args.sample_size}|{args.dataset_dimension}' / f'experiment-{args.experiment_id}'
    else:
        outputFile = "experiment-"
        i = 0
        while os.path.exists(outputFolder / bias_folder_name /  f'{args.dataset}_{args.sample_size}|{args.dataset_dimension}' / f'{outputFile}{i}'):
            i += 1
        outputFolder = outputFolder / bias_folder_name / f'{args.dataset}_{args.sample_size}|{args.dataset_dimension}' / f'{outputFile}{i}'

    config.ensure_dir(outputFolder)
    config.safe_create_dir(outputFolder)

    if args.lr > 1:
        args.lr = 1 / args.lr

    while args.conf > 1:
        args.conf = args.conf / 10

    
    with open(outputFolder / 'settings.txt', 'a+') as f:
        f.write('\n'.join([
            str(outputFolder),
            'experiment id: '+str(args.experiment_id),
            'batch size: '+str(args.batch_size),
            'dataset: '+str(args.dataset),
            'bias_name: '+str(bias['name']),
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
            'Note: This is biased by DBaST. CV is used.'
            ''
        ]))

    
    setup_seed(args.seed)

    gc.collect()

    if args.model == 'diversity' or args.model == 'all':
        try:
            main_divergence(X, y, bias, bias_params, args)
        except Exception as e:
            print(f'diversity could not run! Exception {e} occured!')
    if args.model == 'balanced_semi_supervised' or args.model == 'all':
        try:
            main_balanced_semi_supervised(X, y, bias, bias_params, args)
        except Exception as e:
            print(f'balanced_semi_supervised could not run! Exception {e} occured!')
    if args.model == 'true_semi_supervised' or args.model == 'all':
        try:
            main_true_semi_supervised(X, y, bias, bias_params, args)
        except Exception as e:
            print(f'true_semi_supervised could not run! Exception {e} occured!')
    if args.model == 'supervised' or args.model == 'all':
        try:
            main_supervised(X, y, bias, bias_params, args)
        except Exception as e:
            print(f'supervised could not run! Exception {e} occured!')
    if args.model == 'visualize':
        try:
            main_visualize(X, y, bias, bias_params, args)
        except Exception as e:
            print(f'Visualize could not run! Exception {e} occured!')
