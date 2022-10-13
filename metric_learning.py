import platform
import numpy as np
import torch
import torch.nn as nn

from scipy.spatial.distance import squareform, cdist
import customDataset, customAccuracyCalculator
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score, average_precision_score


from sklearn.model_selection import StratifiedKFold
from pytorch_metric_learning import distances, losses, miners, reducers

import create_convergence_graph
import neuralnetwork
from network import Network
from typing import Sequence, Iterator

from visualize import visualize, visualize_gene

def create_customDataset(data:pd.DataFrame) -> customDataset.CustomDataset:
        features = data.iloc[:, 4:]
        labels = data["class"].astype('long')
        genes = data.iloc[:, :2]
        return customDataset.CustomDataset(
            torch.Tensor(features.values.astype(np.float64)), 
            torch.Tensor(labels.values.astype(np.float64)), 
            genes.values.tolist())

class MetricLearning():

    network:Network = None

    def __init__(self, network:Network, test_data:customDataset.CustomDataset, training_data:customDataset.CustomDataset, training_idx: Sequence[int], validation_idx: Sequence[int], outputFolder: str):
        self.network = network
        self.accuracyCalculator = self.AccuracyCalculator()
        self.outputFolder = outputFolder

        g_test = torch.Generator().manual_seed(43)
        g_train = torch.Generator().manual_seed(42)
        g_validation = torch.Generator().manual_seed(42)
        test_sampler = torch.utils.data.RandomSampler(test_data, generator=g_test)
        self.training_sampler = self.MySubsetRandomSampler(training_idx, g_train)
        self.validation_sampler = self.MySubsetRandomSampler(validation_idx, g_validation)

        self.test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, sampler=test_sampler)
        self.training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=64, sampler=self.training_sampler)
        self.validation_data_loader = torch.utils.data.DataLoader(training_data, batch_size=64, sampler=self.validation_sampler)
    
    def train(self, fold):
        metrics = []
        for epoch in range(100):
            training_loss = self.network.train(self.training_data_loader)
            validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
            train_embeddings, train_labels, train_genes = self.network.run(self.training_data_loader)
            if fold==0 and epoch in [0,5,10,99]:
                visualize(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy(), self.outputFolder, fold, epoch)
                for gene in ['PARP1', ' BRCA1', 'PTEN', 'TP53', 'BRCA2']:
                    visualize_gene(train_embeddings.numpy(), torch.cat([train_labels, validation_labels]), train_genes, validation_embeddings.numpy(), validation_genes, gene, self.outputFolder, fold, epoch)
            accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy())
            if epoch==99:
                self.save_embedding(train_embeddings, train_labels, train_genes, validation_embeddings, validation_labels, validation_genes, fold)
            metrics.append([epoch, fold, training_loss, validation_loss, *accuracies])

        pd.DataFrame(
            data=metrics, 
            columns=['epoch', 'fold', 'train_loss', 'test_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"performance.csv", mode='a', header=fold==0)


        test_embeddings, test_labels, test_genes = self.network.run(self.test_data_loader)
        if fold==0:
            for gene in ['PARP1', ' BRCA1', 'PTEN', 'TP53', 'BRCA2']:
                    visualize_gene(train_embeddings, torch.cat([train_labels, test_labels]), train_genes, test_embeddings, test_genes, gene, self.outputFolder, fold, epoch+1)
        accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), test_embeddings.numpy(), test_labels.numpy())
        pd.DataFrame(
                data=[[fold, *accuracies]],
                columns=['fold', 'accuracy', 'f1_score', 'average_precision', 'auroc']
            ).to_csv(self.outputFolder+"performance-testset.csv", mode='a', header=fold==0)

    def save_embedding(self, train_embeddings, train_labels, train_genes, test_embeddings, test_labels, test_genes, fold):
        columns = ['label']
        columns.extend(['dim'+str(x) for x in range(len(list(zip(*test_embeddings.numpy()))))])
        columns.extend(['gene1', 'gene2'])
        test_df = pd.DataFrame(
            data=list(zip(
                test_labels.numpy(), 
                *list(zip(*test_embeddings.numpy())), 
                *list(zip(*test_genes))
            )),
            columns=columns
        )
        train_df = pd.DataFrame(
            data=list(zip(
                train_labels.numpy(), 
                *list(zip(*train_embeddings.numpy())), 
                *list(zip(*train_genes))
            )),
            columns=columns
        )
        test_df.to_csv(self.outputFolder+f'embeddings_test_fold_{fold}.csv')
        train_df.to_csv(self.outputFolder+f'embeddings_train_fold_{fold}.csv')

    class MySubsetRandomSampler(torch.utils.data.Sampler[int]):
        r"""Samples elements randomly from a given list of indices, without replacement.

        Args:
            indices (sequence): a sequence of indices
            generator (Generator): Generator used in sampling.
        """
        indices: Sequence[int]

        def __init__(self, indices: Sequence[int], generator=None) -> None:
            self.indices = indices
            self.generator = generator

        def __iter__(self) -> Iterator[int]:
            for i in torch.randperm(len(self.indices), generator=self.generator):
                yield self.indices[i]

        def __len__(self) -> int:
            return len(self.indices)

        def add_index(self, index: int) -> None:
            indices = list(self.indices)
            indices.append(index)
            self.indices = indices

        def add_indices(self, indices: Sequence[int]) -> None:
            indices = list(self.indices)
            indices.extend(indices)
            self.indices = indices
        
        def remove_index(self, index: int) -> None:
            indices = list(self.indices)
            indices.remove(index)
            self.indices = indices

        def remove_indices(self, indices: Sequence[int]) -> None:
            for index in indices:
                self.remove_index(index)

    class AccuracyCalculator():
        def __init__(self) -> None:
            pass

        def get_accuracies(self, train_embeddings, train_labels, test_embeddings, test_labels):
            # Pairwise distances between test/validation embeddings as the rows and training embeddings as the columns
            pairwise = pd.DataFrame(
                cdist(test_embeddings, train_embeddings, metric='euclidean')
            )
            # Get the indices of the columns with the largest distance to each row
            sorted_pairwise = np.argsort(pairwise.values, 1)
            idx = sorted_pairwise[:, :5]
            # Get the 5 nearest distances and labels
            knn_distances = np.array([[pairwise.iloc[i, j] for j in row] for i, row in enumerate(idx)])

            max_distance = np.array([pairwise.iloc[i, j] for i, j in enumerate(sorted_pairwise[:, -1])])
            max_distance[max_distance<1]=1
            reshaped_distances = knn_distances / max_distance[:, np.newaxis]

            knn_labels = np.array([[train_labels[element] for element in row] for row in idx])
            # Calculate the weighted knn where the distance between the sample and nearest neighbour is subtracted from the neighbour label according to:
            # Label_k * (1 - |distance to k|) + (1 - label_k) * |distance to k|, where k represents the nearest neighbour k
            weighted_knn_labels = knn_labels + ((1 - 2 * knn_labels) * reshaped_distances)

            accuracy = np.equal(weighted_knn_labels.mean(1).round(), test_labels).astype(float).mean()

            y_pred = weighted_knn_labels.mean(1).round()
            f1 = f1_score(test_labels, y_pred)
            y_prob = weighted_knn_labels.mean(1)
            ap = average_precision_score(test_labels, y_prob)
            auroc = roc_auc_score(test_labels, y_prob)
            return [accuracy, f1, ap, auroc]