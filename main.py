import platform
import os
import argparse

import pandas as pd

import neuralnetwork

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = ""
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run neuralnetwork"
    )
    parser.add_argument("-t", "--testset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("-l", "--layers", type=int, nargs='*', default=[])
    parser.add_argument('-C', '--cancer', choices=['BRCA', 'CESC', 'COAD', 'KIRC', 'LAML', 'LUAD', 'SKCM', 'OV'])
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

    i = 0
    while os.path.exists(outputFolder + "experiment-%s" % i):
        i += 1
    outputFolder = outputFolder + "experiment-%s/" % i
    os.mkdir(outputFolder)

    with open(outputFolder + 'settings.txt', 'a') as f:
        f.write('\n'.join([
            'experiment-'+str(i),
            'number of epochs: '+str(args.num_epochs),
            'batch size: '+str(args.batch_size),
            'trainingset: '+args.file,
            'testset: '+str(args.testset),
            'cancer: '+str(args.cancer),
            'layers: '+' '.join([str(x) for x in layers])
        ]))
    
    
    
    dataset = pd.read_csv(dataPath).fillna(0)
    if args.cancer is not None:
        dataset = dataset[dataset['cancer']==args.cancer]

    neuralnetwork.main(outputPath=outputFolder, dataset=dataset, num_epochs=args.num_epochs, batch_size=args.batch_size, layers=args.layers)
