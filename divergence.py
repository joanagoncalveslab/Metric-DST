import numpy as np
import torch
import time

from scipy.spatial.distance import cdist
from customDataset import CustomDataset
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

import torch.utils.data

from network import Network
from typing import Sequence, Iterator

def create_customDataset(data:pd.DataFrame) -> CustomDataset:
        features = data.iloc[:, 4:]
        labels = data["class"].astype('long')
        genes = data.iloc[:, :2]
        return CustomDataset(
            torch.Tensor(features.values.astype(np.float64)), 
            torch.Tensor(labels.values.astype(np.float64)), 
            genes.values.tolist())

class EarlyStop():
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.min_val_loss = np.Inf
        self.counter = 0
        self.early_stop = False
        self.model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.min_val_loss - self.delta:
            self.min_val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter+=1
            if self.counter > self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        self.model_state = model.state_dict()

    def load_checkpoint(self):
        return self.model_state


class SelfTraining():

    network:Network = None

    def __init__(self, network:Network, validation_data:CustomDataset, training_data:CustomDataset, training_idx: Sequence[int], unlabeled_idx: Sequence[int], outputFolder: str):
        self.network = network
        self.accuracyCalculator = self.AccuracyCalculator()
        self.outputFolder = outputFolder

        g_validation = torch.Generator().manual_seed(43)
        g_train = torch.Generator().manual_seed(42)
        g_unlabeled = torch.Generator().manual_seed(42)
        validation_sampler = torch.utils.data.RandomSampler(validation_data, generator=g_validation)
        self.training_sampler = self.MySubsetRandomSampler(training_idx, g_train)
        self.unlabeled_sampler = self.MySubsetRandomSampler(unlabeled_idx, g_unlabeled)

        self.validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, sampler=validation_sampler)
        self.training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=64, sampler=self.training_sampler)
        self.unlabeled_data_loader = torch.utils.data.DataLoader(training_data, batch_size=64, sampler=self.unlabeled_sampler)
    
    def train(self, fold) -> None:
        training_rounds = 100
        early_stop = EarlyStop(10)
        metrics = []
        # Train initial model
        for epoch in range(training_rounds):
            training_loss = self.network.train(self.training_data_loader)
            validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
            train_embeddings, train_labels, train_genes = self.network.run(self.training_data_loader)
            accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy())
            metrics.append([epoch, fold, training_loss, validation_loss, *accuracies])
            early_stop(validation_loss, self.network.model)
            if early_stop.early_stop:
                print(f"Early stop after epoch: {epoch}")
                break

        # Loop and add new samples
        for loop in range(10):
            self.find_sample_to_add()
            for epoch in range(10):
                training_loss = self.network.train(self.training_data_loader)
                validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
                train_embeddings, train_labels, train_genes = self.network.run(self.training_data_loader)
                accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy())
                metrics.append([epoch+training_rounds+(loop*10), fold, training_loss, validation_loss, *accuracies])
            # Evaluate if we made any progress for x number of rounds

        pd.DataFrame(
            data=metrics, 
            columns=['epoch', 'fold', 'train_loss', 'test_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"performance.csv", mode='a', header=fold==0)

    def find_sample_to_add(self):
        training_embeddings, training_labels, training_genes = self.network.run(self.training_data_loader)
        unlabeled_embeddings, _, unlabeled_genes, unlabeled_idx = self.network.run_with_idx(self.unlabeled_data_loader)
        unlabeled_idx  = np.array(unlabeled_idx)
        # Find dimension of output space
        embeddings = list(zip(*training_embeddings))
        dimensions = []
        for dimension in zip(*training_embeddings):
            dimensions.append((min(dimension), max(dimension)))

        # Find samples to add
        pseudolabels = []
        samples_to_add = []
        while len(pseudolabels) < 10:
            # Random point in space
            random_coordinate = []
            for low, high in dimensions:
                random_coordinate.append((np.random.random() * (high-low)) + low)

            # Find closest unlabeled sample
            selected_gene_id = pd.DataFrame(
                cdist(unlabeled_embeddings, [random_coordinate], metric='euclidean')
            ).idxmin().values[0]

            # Find confidence of selected gene
            closest_samples = pd.DataFrame(
                cdist(training_embeddings, [np.array(unlabeled_embeddings[selected_gene_id])], metric='euclidean')
            ).sort_values(by=0)
            knn_labels = np.array([training_labels[sample] for sample in closest_samples.index.values[:5]])
            knn_distances = np.array(closest_samples.values[:5, :].T[0])
            weighted_knn_labels = knn_labels + np.multiply((1 - 2 * knn_labels), knn_distances.T).T

            confidence = weighted_knn_labels.mean()
            if (confidence > 0.7 or confidence < 0.3) and selected_gene_id not in samples_to_add:
                if round(confidence) == 1 and sum(pseudolabels) < 5:
                    samples_to_add.append(selected_gene_id)
                    pseudolabels.append(round(confidence))
                elif round(confidence) == 0 and len(pseudolabels) - sum(pseudolabels) < 5:
                    samples_to_add.append(selected_gene_id)
                    pseudolabels.append(round(confidence))
        
        # Transfer selected samples from unlabeled to labeled samples
        indices_of_samples_to_add = [unlabeled_idx[x] for x in samples_to_add]
        self.unlabeled_sampler.remove_indices(indices_of_samples_to_add)
        self.training_sampler.add_indices(indices_of_samples_to_add)
        # Update the newly added samples with the pseudolabels as their labels
        for x, y in zip(pseudolabels, indices_of_samples_to_add):
            self.training_data_loader.dataset.labels[y] = x

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
            prev_indices = list(self.indices)
            prev_indices.extend(indices)
            self.indices = prev_indices
        
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
            idx = np.argsort(pairwise.values, 1)[:, :5]
            # Get the 5 nearest distances and labels
            knn_distances = np.array([[pairwise.iloc[i, j] for j in row] for i, row in enumerate(idx)])
            knn_labels = np.array([[train_labels[element] for element in row] for row in idx])
            # Calculate the weighted knn where the distance between the sample and nearest neighbour is subtracted from the neighbour label according to:
            # Label_k * (1 - |distance to k|) + (1 - label_k) * |distance to k|, where k represents the nearest neighbour k
            weighted_knn_labels = knn_labels + ((1 - 2 * knn_labels) * knn_distances)
            
            accuracy = np.equal(weighted_knn_labels.mean(1).round(), test_labels).astype(float).mean()

            y_pred = weighted_knn_labels.mean(1).round()
            f1 = f1_score(test_labels, y_pred)
            y_prob = weighted_knn_labels.mean(1)
            ap = average_precision_score(test_labels, y_prob)
            auroc = roc_auc_score(test_labels, y_prob)
            return [accuracy, f1, ap, auroc]

        def get_accuracies_torch(self, train_embeddings:torch.Tensor, train_labels:torch.Tensor, test_embeddings:torch.Tensor, test_labels):
            # Pairwise distances between test/validation embeddings as the rows and training embeddings as the columns
            pairwise = torch.cdist(test_embeddings, train_embeddings, p=2)
            # Get the indices of the columns with the largest distance to each row
            idx = torch.argsort(pairwise, dim=1)
            # Get the 5 nearest distances and labels
            knn_distances = np.array([[pairwise[i, j] for j in row] for i, row in enumerate(idx)])
            knn_labels = np.array([[train_labels[element] for element in row] for row in idx])
            # Calculate the weighted knn where the distance between the sample and nearest neighbour is subtracted from the neighbour label according to:
            # Label_k * (1 - |distance to k|) + (1 - label_k) * |distance to k|, where k represents the nearest neighbour k
            weighted_knn_labels = knn_labels + ((1 - 2 * knn_labels) * knn_distances)
            
            accuracy = np.equal(weighted_knn_labels.mean(1).round(), test_labels).astype(float).mean()

            y_pred = weighted_knn_labels.mean(1).round()
            f1 = f1_score(test_labels, y_pred)
            y_prob = weighted_knn_labels.mean(1)
            ap = average_precision_score(test_labels, y_prob)
            auroc = roc_auc_score(test_labels, y_prob)
            return [accuracy, f1, ap, auroc]