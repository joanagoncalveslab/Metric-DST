from collections import OrderedDict
from copy import deepcopy
import torch
import torch.utils.data
from typing import Iterator, Sequence
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from customDataset import CustomDataset
from network import Net

def create_customDataset(data:pd.DataFrame) -> CustomDataset:
        features = data.drop(columns=['gene1', 'gene2', 'class'])#data.iloc[:, 4:]
        labels = data["class"].astype('long')
        genes = data[['gene1', 'gene2']]
        return CustomDataset(
            torch.Tensor(features.values.astype(np.float64)), 
            torch.Tensor(labels.values.astype(np.float64)), 
            genes.values.tolist())

class MySubsetRandomSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
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
        new_indices = list(self.indices)
        new_indices.extend(indices)
        self.indices = new_indices
    
    def remove_index(self, index: int) -> None:
        indices = list(self.indices)
        indices.remove(index)
        self.indices = indices

    def remove_indices(self, indices: Sequence[int]) -> None:
        for index in indices:
            self.remove_index(index)

    def set_indices(self, indices: Sequence[int]) -> None:
        self.indices = indices


class AccuracyCalculator():
    def __init__(self, knn:int) -> None:
        self.knn = knn

    def get_accuracies(self, train_embeddings:np.ndarray, train_labels:np.ndarray, test_embeddings:np.ndarray, test_labels:np.ndarray) -> Sequence[float]:
        # Pairwise distances between test/validation embeddings as the rows and training embeddings as the columns
        pairwise = pd.DataFrame(
            cdist(test_embeddings, train_embeddings, metric='euclidean')
        )
        # Get the indices of the columns with the largest distance to each row
        sorted_pairwise = np.argsort(pairwise.values, 1)
        idx = sorted_pairwise[:, :self.knn]
        # Get the 5 nearest distances and labels
        knn_distances = np.array([[min(pairwise.iloc[i, j], 0.5) for j in row] for i, row in enumerate(idx)])

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

class EarlyStop():
    def __init__(self, patience:int=5, delta:int=0, val_loss:float=np.Inf, model:Net=None, epoch:int=0):
        self.patience = patience
        self.delta = delta
        self.min_val_loss = np.Inf
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        if model is None:
            self.model_state = None
        else:
            self.save_checkpoint(model)
            self.min_val_loss = val_loss
            self.best_epoch = epoch

    def __call__(self, val_loss: float, model: Net, epoch: int) -> None:
        if val_loss < self.min_val_loss - self.delta:
            self.min_val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            self.best_epoch = epoch
        else:
            self.counter+=1
            if self.counter > self.patience:
                self.early_stop = True

    def check(self, val_loss: float, model: Net, epoch: int) -> None:
        if val_loss < self.min_val_loss - self.delta:
            print(f'updated model becuase {val_loss} is lower than {self.min_val_loss}')
            res = {
                'model_is_updated': True,
                'old_val_loss': self.min_val_loss,
                'new_val_loss': val_loss,
                'epoch': epoch
            }
            self.min_val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            self.best_epoch = epoch
            return res 
        else:
            self.counter+=1
            if self.counter >= self.patience:
                self.early_stop = True
            return {
                'model_is_updated': False,
                'old_val_loss': self.min_val_loss,
                'new_val_loss': val_loss,
                'epoch': epoch
            }

    def save_checkpoint(self, model:Net) -> None:
        self.model_state = deepcopy(model.state_dict())

    def load_checkpoint(self, model:Net) -> 'OrderedDict[str, torch.Tensor]':
        model.load_state_dict(self.model_state)
        return self.model_state

    def get_best_epoch(self) -> int:
        return self.best_epoch

    def get_patience(self) -> int:
        return self.patience

    def reset(self) -> None:
        self.min_val_loss = np.Inf
        self.counter = 0
        self.early_stop = False
        self.model_state = None
        self.best_epoch = 0
