import platform
import os
import argparse
import random
import time

import pandas as pd
import numpy as np
import torch

import neuralnetwork

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

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
        description="Run neuralnetwork"
    )
    parser.add_argument("-t", "--testset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("-l", "--layers", type=int, nargs='*', default=[])
    parser.add_argument('-C', '--cancer', choices=['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'SKCM', 'OV'])
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--miner", action='store_true')
    parser.add_argument("--contrastive", action='store_true')
    parser.add_argument("--exclude", type=str, default='')

    parser.add_argument("--vis-genes", action='store_true')
    parser.add_argument("--vis-embeddings", action='store_true')
    parser.add_argument("--save-embeddings", action='store_true')

    parser.add_argument("file")
    return parser

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    dataPath = args.file
    if not os.path.exists(dataPath):
        dataPath = webDriveFolder + dataPath
        if not os.path.exists(dataPath):
            raise FileNotFoundError('The dataset does not exist')
    
    layers = [x for x in args.layers if x > 0]

    if args.output_file:
        outputFolder = outputFolder + args.output_file + "/"
    else:
        outputFile = "experiment-"
        i = 0
        while os.path.exists(outputFolder + outputFile + str(i)):
            i += 1
        outputFolder = outputFolder + outputFile + str(i) + '/'

    time.sleep(random.random())

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    if args.lr > 1:
        args.lr = 1 / args.lr

    with open(outputFolder + 'settings.txt', 'a') as f:
        f.write('\n'.join([
            outputFolder,
            'number of epochs: '+str(args.num_epochs),
            'batch size: '+str(args.batch_size),
            'trainingset: '+args.file,
            'testset: '+str(args.testset),
            'cancer: '+str(args.cancer),
            'layers: '+' '.join([str(x) for x in layers]),
            'learning rate: '+str(args.lr),
            'seed: '+str(args.seed),
            'knn: '+str(args.knn),
            'miner: '+str(args.miner),
            'loss: '+('contrastive' if args.contrastive else 'triplet'),
            'excluded gene: '+str(args.exclude),
            ''
        ]))

    setup_seed(args.seed)
    
    dataset = pd.read_csv(dataPath).fillna(0)
    if args.cancer is not None:
        dataset = dataset[dataset['cancer']==args.cancer]
    dataset = dataset[(dataset['seq1']!=-1.0) & (dataset['seq1']!=0.0)]
    train_dataset = dataset[(dataset['gene1']!=args.exclude) & (dataset['gene2']!=args.exclude)]
    test_dataset = dataset[(dataset['gene1']==args.exclude) | (dataset['gene2']==args.exclude)]

    idx_0 = train_dataset[train_dataset['class'] == 0]
    idx_1 = train_dataset[train_dataset['class'] == 1]
    if len(idx_0) < len(idx_1):
        idx_1 = idx_1.sample(len(idx_0))
    if len(idx_0) > len(idx_1):
        idx_0 = idx_0.sample(len(idx_1))
    train_dataset = pd.concat([idx_0, idx_1])

    train_dataset = pd.concat([train_dataset, test_dataset[test_dataset['class'] == 0].iloc[:40, :], test_dataset[test_dataset['class'] == 1].iloc[:10, :]])
    test_dataset = pd.concat([test_dataset[test_dataset['class'] == 0].iloc[40:, :], test_dataset[test_dataset['class'] == 1].iloc[10:, :]])

    flags = {
        'visualize_genes': args.vis_genes,
        'visualize_embeddings': args.vis_embeddings,
        'save_embeddings': args.save_embeddings,
        'miner': args.miner,
        'contrastive': args.contrastive
    }

    neuralnetwork.main(outputPath=outputFolder, dataset=train_dataset, testset=test_dataset, num_epochs=args.num_epochs, batch_size=args.batch_size, layers=layers, seed=args.seed, learning_rate=args.lr, flags_in=flags, knn=args.knn)
