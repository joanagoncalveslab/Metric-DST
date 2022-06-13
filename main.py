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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("-l", "--layers", type=int, nargs='*', default=[])
    parser.add_argument('-C', '--cancer', choices=['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'SKCM', 'OV'])
    parser.add_argument("-B", "--balance", action="store_true")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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

    time.sleep(random.randint(10, 100))

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    with open(outputFolder + 'settings.txt', 'a') as f:
        f.write('\n'.join([
            outputFolder,
            'number of epochs: '+str(args.num_epochs),
            'batch size: '+str(args.batch_size),
            'trainingset: '+args.file,
            'testset: '+str(args.testset),
            'cancer: '+str(args.cancer),
            'layers: '+' '.join([str(x) for x in layers]),
            'balanced: '+str(args.balance),
            'with dropout of 0.5',
            ''
        ]))

    setup_seed(args.seed)

    # iter_csv = pd.read_csv(dataPath, chunksize=1000)
    # dataset = pd.concat([chunk[chunk['cancer']==args.cancer] for chunk in iter_csv]).fillna(0)
    
    dataset = pd.read_csv(dataPath).fillna(0)
    if args.cancer is not None:
        dataset = dataset[dataset['cancer']==args.cancer]

    if args.balance:
        index = dataset['class'].value_counts()

        major_class = dataset[dataset['class']==index.index[0]].index
        minor_class = dataset[dataset['class']==index.index[1]].index

        sampled_major_class = np.random.choice(major_class, index.values[1], replace=False)
        idx = np.concatenate([sampled_major_class, minor_class])
        np.random.shuffle(idx)
        dataset = dataset.loc[idx]

    neuralnetwork.main(outputPath=outputFolder, dataset=dataset, num_epochs=args.num_epochs, batch_size=args.batch_size, layers=layers, seed=args.seed)
