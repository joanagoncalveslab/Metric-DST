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
from src.dbast import bias_techniques as y_bias
from src.custom import biases as c_bias
from src.andrei import dataset_generator as a_dg
from src.custom import dataset_generator as c_dg
from src.dbast import load_dataset as y_dg

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = config.RESULT_DIR / 'bias' / 'yasin'
else:
    webDriveFolder = "/home/nfs/ytepeli/python_projects/msc-thesis-2122-mathijs-de-wolf/data"
    outputFolder = config.RESULT_DIR / 'bias' / 'yasin'


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
    parser.add_argument('--dataset', '-d', default='custom_')
    parser.add_argument("--output-file", "-of", type=str, default=None)
    parser.add_argument("--experiment_id", "-eid", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--conf", "-c", type=float, default=0.9)
    parser.add_argument("--num-pseudolabels", type=int, default=-2)

    parser.add_argument("--retrain", "-rt", action='store_true')
    parser.add_argument("--early_stop_pseudolabeling", "-esp", action='store_true')
    parser.add_argument("--single-fold", action='store_true')
    parser.add_argument("--model", "-m", default='diversity')
    parser.add_argument("--bias", "-b", default='bias_balanced_multi_features')
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
    pdf_loc = f'{out_loc}.pdf'
    config.ensure_dir(png_loc)
    # plt.show()
    plt.savefig(pdf_loc, dpi=300, bbox_inches='tight')
    plt.savefig(png_loc, dpi=300, bbox_inches='tight')


def main_visualize(data_dict, args):
    import umap
    # --------------- Divergence ---------------
    for fold, fold_dict in data_dict.items():
        plt.clf()
        print(f'fold: {fold}')
        X_train, y_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        x_ind = fold_dict['chosen_ids']
        
        reducer = umap.UMAP()
        if X_train.shape[1]==2:
            embedding = X_train.copy()
        else:
            embedding = reducer.fit_transform(X_train)
        out_loc = outputFolder / 'bias_vis' / f'{fold}'
        config.ensure_dir(out_loc)
        vis2d(embedding, y_train, x_ind, out_loc)
    

def main_divergence(data_dict, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    
    # --------------- Divergence ---------------
    for fold, fold_dict in data_dict.items():
        plt.clf()
        print(f'fold: {fold}')

        X_train, y_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        x_ind = fold_dict['chosen_ids']        
        
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
    
    for fold, fold_dict in data_dict.items():
        plt.clf()
        print(f'fold: {fold}')

        X_train, y_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        x_ind = fold_dict['chosen_ids']
        
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
    
    for fold, fold_dict in data_dict.items():
        plt.clf()
        print(f'fold: {fold}')

        X_train, y_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        x_ind = fold_dict['chosen_ids']
        
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
    
    for fold, fold_dict in data_dict.items():
        plt.clf()
        print(f'fold: {fold}')

        X_train, y_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_b_train, y_b_train = fold_dict['xs'].copy(), fold_dict['ys'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        x_ind = fold_dict['chosen_ids']
        
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

def main_supervised_none(data_dict, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    
    for fold, fold_dict in data_dict.items():
        plt.clf()
        print(f'fold: {fold}')

        #X_train, y_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_b_train, y_b_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        x_ind = fold_dict['chosen_ids']
        
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
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "supervised_none", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, supervised=True)
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "supervised_none_performance.csv"), outputFolder)

def main_supervised_random(data_dict, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    device = torch.device(device)

    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
    reducer = reducers.MeanReducer()
    
    for fold, fold_dict in data_dict.items():
        plt.clf()
        print(f'fold: {fold}')

        #X_train, y_train = fold_dict['xs_nobias'].copy(), fold_dict['ys_nobias'].copy()
        X_b_train, y_b_train = fold_dict['xs_random'].copy(), fold_dict['ys_random'].copy()
        X_unlabeled, y_unlabeled = fold_dict['xg'].copy(), fold_dict['yg'].copy()
        X_test, y_test = fold_dict['xt'].copy(), fold_dict['yt'].copy()
        x_ind = fold_dict['chosen_ids']
        
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
        ml = SelfTraining(ntwrk, fold_test_dataset, train_dataset, unlabeled_dataset, validation_dataset, str(outputFolder)+'/', "supervised_random", args.knn, args.conf, args.num_pseudolabels)
        ml.train(fold, supervised=True)
        if args.single_fold:
            break
    gc.collect()
    create_convergence_graph.create_fold_convergence_graph(str(outputFolder / "supervised_random_performance.csv"), outputFolder)

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    print(args)

    #dataPath = webDriveFolder + f"train_seq_128_{args.cancer}.csv"
    #unlabeledDatapath = webDriveFolder + f"unknown_repair_cancer_{args.cancer}_seq_128.csv"
    #testDatapath = webDriveFolder + f"test_seq_128.csv"
    #f'ns{n_samples}_nc{n_clusters}_nf{n_features}_ri{r_informative}_rs{random_state}_fy{flip_y}_cs{class_sep}.png'
    potential_bias_size = 36
    custom_args_dict = {'ns': 'n_samples', 'nc': 'n_clusters', 'nf': 'n_features', 'ri': 'r_informative', 'rs': 'random_state', 'fy': 'flip_y', 'cs':'class_sep'}
    if 'custom' in args.dataset:
        custom_ds_args = {}
        for data_arg in args.dataset.split('_')[1:]:
            data_abbr = data_arg[:2]
            data_val = float(data_arg[2:]) if '.' in data_arg[2:] else int(data_arg[2:])
            custom_ds_args[custom_args_dict[data_abbr]] = data_val
        X, y = c_dg.custom_mdimension(**custom_ds_args)
    elif 'moon_' in args.dataset:
        X, y = a_dg.rotated_moons(int(args.dataset.split('_')[1]))
        if args.bias == 'andrei_bias_2features':
            bias = {'name': args.bias, 'i':5, 'n_class_partitions':10, 'b_1':1.5, 'b_2':1.5, 'delta_1': (1, 0.5), 'delta_2':(0, 0)} 
            potential_bias_size = np.round(((len(y)*0.9)*0.3)/2)
    elif args.dataset == 'andrei_three_unbalanced':
        X, y = a_dg.three_unbalanced_clusters()
        if args.bias == 'andrei_bias_2features':
            bias = {'name': args.bias, 'i':5, 'n_class_partitions':10, 'b_1':0.1, 'b_2':5, 'delta_1': (-7, 6), 'delta_2': (5, 5)} 
            potential_bias_size = np.round(((len(y)*0.9)*0.3)/2)
    elif args.dataset == 'andrei_overlapped':
        X, y = a_dg.overlapped_linear_clusters(args.sample_size, args.sample_size)
        if args.bias == 'andrei_bias_2features':
            bias = {'name': args.bias, 'i':5, 'n_class_partitions':10, 'b_1':0.5, 'b_2':0.1, 'delta_1': (0,0), 'delta_2': (-25, 0)}
            potential_bias_size = np.round(((len(y)*0.9)*0.3)/2)
    elif args.dataset == 'andrei_rotated_moons':
        X, y = a_dg.rotated_moons(args.sample_size)
        if args.bias == 'andrei_bias_2features':
            bias = {'name': args.bias, 'i':5, 'n_class_partitions':10, 'b_1':1.5, 'b_2':1.5, 'delta_1': (1, 0.5), 'delta_2':(0, 0)} 
            potential_bias_size = np.round(((len(y)*0.9)*0.3)/2)
    elif args.dataset == 'andrei_multidimension':
        X, y = a_dg.multidimensional_set(args.sample_size, args.dataset_dimension, args.seed)
        bias = {'name': args.bias, 'size':50, 'a':3, 'b':4} 
        potential_bias_size = bias['size']*2
    else:
        print(f'The choice {args.dataset} does not exist, Looking for others!!!')
        X, y = y_dg.load_dataset(args.dataset, test=False)
        #raise Exception(f'The choice {args.dataset} does not exist!!!')


    if args.bias == 'none':
        bias = {'name': None}
        potential_bias_size = np.round((len(y)*0.9)*0.3) #custom_ns4000_nc3_nf16_ri1.0_rs0_fy0_cs1.5
    elif 'hierarchyy_' in args.bias:
        bias = {'name': 'hierarchyy', 'y': True, 'prob': float(args.bias.split('_')[1]), 'max_size': float(args.bias.split('_')[2])}
        potential_bias_size = bias['max_size']*2
    elif 'delta_' in args.bias:
        b_par_args = args.bias.split('_')[1]
        delta1 = b_par_args.split('|')[2].split(',')
        delta2 = b_par_args.split('|')[3].split(',')
        bias = {'name': 'delta', 'size': int(b_par_args.split('|')[0]), 'b': float(b_par_args.split('|')[1]), 'delta1': (float(delta1[0]), float(delta1[1])), 'delta2': (float(delta2[0]), float(delta2[1]))}
        potential_bias_size = bias['size']*2
    bias_params = {key: val for key, val in bias.items() if ('name' not in key) and ('y' != key)}
    bias_param_str = "|".join([str(i) for i in bias_params.values()]).replace(" ", "").replace(",", "_")
    bias_folder_name = args.bias#f'{bias["name"]}_{bias_param_str}'
    
    
    experiment_args = {'bs': args.batch_size, 'rs':args.seed, 'lr': args.lr, 'knn':args.knn, 'c':args.conf, 
                       'kb': args.num_pseudolabels, 'bal': args.balance}
    if args.retrain:
        experiment_args['rt']=''
    if args.early_stop_pseudolabeling:
        experiment_args['esp']=''
    
    experiment_str = "_".join([f'{exp_k}{exp_v}' for exp_k, exp_v in experiment_args.items()])
    
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

    if args.num_pseudolabels < -2:
        provisional_no = np.power(potential_bias_size, -1/args.num_pseudolabels) 
        args.num_pseudolabels = provisional_no - provisional_no%2 
    
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

    total_number_of_folds = 10
    fold_dict = collections.OrderedDict()
    fold_loc = outputFolder / 'data' / 'data_folds.pkl'
    config.ensure_dir(fold_loc)
    if os.path.exists(fold_loc):
        with open(fold_loc, 'rb') as f:
            fold_dict = pkl.load(f)
        if 'xs_random' not in fold_dict[0].keys():
            for fold_i in fold_dict.keys():
                X_train, y_train = fold_dict[fold_i]['xs_nobias'].copy(), fold_dict[fold_i]['ys_nobias'].copy()
                x_r_ind = y_bias.get_bias('random', X_train, y=y_train, size=int(potential_bias_size/2), r_seed=args.seed+fold_i).astype(int)
                X_r_train, y_r_train = X_train[x_r_ind, :], y_train[x_r_ind]
                fold_dict[fold_i]['xs_random'] = X_r_train.copy()
                fold_dict[fold_i]['ys_random'] = y_r_train.copy()
            with open(fold_loc, 'wb') as f:
                pkl.dump(fold_dict, f)
    else:
        skf = StratifiedKFold(n_splits=total_number_of_folds, shuffle=True, random_state=args.seed)
        for fold_i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f'fold: {fold_i}')
            
            X_rest, y_rest, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
            X_train, X_unlabeled, y_train, y_unlabeled = y_dg.split_dataset(X_rest, y_rest, 0.3, 0.7, r_seed=args.seed+fold_i)
            
            if bias['name'] is None:
                X_b_train, y_b_train = X_train.copy(), y_train.copy()
            elif 'andrei' in bias['name']:
                X_b_train, y_b_train = a_bias.call_bias(X_train.copy(), y_train.copy(), bias['name'], **bias_params)
                X_train_tup = [tuple(arow) for arow in X_train]
                X_b_train_tup = [tuple(arow) for arow in X_b_train]
                X_diff_train = np.setdiff1d(X_train_tup,X_b_train_tup)
                x_ind = array_row_intersection(X_train, X_b_train)
            elif 'hierarchy' in bias['name']: 
                x_ind = y_bias.get_bias(bias['name'], X_train, y=y_train, **bias_params).astype(int)
                X_b_train, y_b_train = X_train[x_ind, :], y_train[x_ind]
            elif 'delta' in bias['name']: 
                x_ind = c_bias.call_bias(X_train, y_train, bias['name'],  **bias_params).astype(int)
                X_b_train, y_b_train = X_train[x_ind, :], y_train[x_ind]
                
            x_r_ind = y_bias.get_bias('random', X_train, y=y_train, size=int(potential_bias_size/2), r_seed=args.seed+fold_i).astype(int)
            X_r_train, y_r_train = X_train[x_r_ind, :], y_train[x_r_ind]
                
            fold_dict[fold_i] = {'xg': X_unlabeled.copy(), 'yg': y_unlabeled.copy(),
                                'xs': X_b_train.copy(), 'ys': y_b_train.copy(),
                                'xs_nobias': X_train.copy(), 'ys_nobias': y_train.copy(),
                                'xs_random': X_r_train.copy(), 'ys_random': y_r_train.copy(),
                                'xt': X_test.copy(), 'yt': y_test.copy(),
                                'chosen_ids': x_ind.copy()}
            with open(fold_loc, 'wb') as f:
                pkl.dump(fold_dict, f)

    setup_seed(args.seed)
    gc.collect()

    if args.model == 'visualize' or args.model == 'all':
        try:
            main_visualize(fold_dict, args)
        except Exception as e:
            print(f'Visualize could not run! Exception {e} occured!')
    if args.model == 'supervised' or args.model == 'all':
        try:
            main_supervised(fold_dict, args)
        except Exception as e:
            print(f'supervised could not run! Exception {e} occured!')
    if args.model == 'diversity' or args.model == 'all':
        try:
            main_divergence(fold_dict, args)
        except Exception as e:
            print(f'diversity could not run! Exception {e} occured!')
    if args.model == 'true_semi_supervised' or args.model == 'all':
        try:
            main_true_semi_supervised(fold_dict, args)
        except Exception as e:
            print(f'true_semi_supervised could not run! Exception {e} occured!')
    if args.model == 'balanced_semi_supervised' or args.model == 'all':
        try:
            main_balanced_semi_supervised(fold_dict, args)
        except Exception as e:
            print(f'balanced_semi_supervised could not run! Exception {e} occured!')
    if args.model == 'nobias' or args.model == 'all' or args.model == 'comparison':
        try:
            main_supervised_none(fold_dict, args)
        except Exception as e:
            print(f'main_supervised_none could not run! Exception {e} occured!')
    if args.model == 'random' or args.model == 'all' or args.model == 'comparison':
        try:
            main_supervised_random(fold_dict, args)
        except Exception as e:
            print(f'main_supervised_random could not run! Exception {e} occured!')
