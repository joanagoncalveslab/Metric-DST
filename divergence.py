from math import ceil, floor
import numpy as np
import torch
import time

from scipy.spatial.distance import cdist
from customDataset import CustomDataset
import pandas as pd

import torch.utils.data

from network import Network
from typing import Sequence

from visualize import visualize, visualize_show
from metric_learning_utils import EarlyStop, AccuracyCalculator, MySubsetRandomSampler, create_customDataset

class SelfTraining():
    def __init__(self, network:Network, test_data:pd.DataFrame, train_data:pd.DataFrame, unlabeled_data:pd.DataFrame, validation_data:pd.DataFrame, outputFolder: str, knn: int):
        self.network = network
        self.accuracyCalculator = AccuracyCalculator(knn=knn)
        self.outputFolder = outputFolder
        self.number_of_unlabeled_samples_added = 0
        self.knn = knn
        self.transform_data(test_data, train_data, unlabeled_data, validation_data)

    def transform_data(self, test_data:pd.DataFrame, train_data:pd.DataFrame, unlabeled_data:pd.DataFrame, validation_data:pd.DataFrame) -> None:
        self.train_data = train_data
        self.unlabeled_data = unlabeled_data

        combined_data = pd.concat([train_data, unlabeled_data], ignore_index=True)
        train_idx = range(len(train_data))
        unlabeled_idx = range(len(train_data), len(combined_data))
        assert(train_data.equals(combined_data.iloc[train_idx]))

        g_train = torch.Generator().manual_seed(42)
        g_unlabeled = torch.Generator().manual_seed(42)
        g_pseudolabel = torch.Generator().manual_seed(42)

        test_data = create_customDataset(test_data)
        combined_data = create_customDataset(combined_data)
        validation_data = create_customDataset(validation_data)

        test_sampler = torch.utils.data.SequentialSampler(test_data)
        train_sampler = MySubsetRandomSampler(train_idx, generator=g_train)
        unlabeled_sampler = MySubsetRandomSampler(unlabeled_idx, generator=g_unlabeled)
        validation_sampler = torch.utils.data.SequentialSampler(validation_data)
        pseudolabel_sampler = MySubsetRandomSampler([], generator=g_pseudolabel)
        self.train_sampler = train_sampler
        self.unlabeled_sampler = unlabeled_sampler
        self.pseudolabel_sampler = pseudolabel_sampler
        self.test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), sampler=test_sampler)
        self.train_data_loader = torch.utils.data.DataLoader(combined_data, batch_size=64, sampler=train_sampler)
        self.unlabeled_data_loader = torch.utils.data.DataLoader(combined_data, batch_size=64, sampler=unlabeled_sampler)
        self.validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=len(validation_data), sampler=validation_sampler)
        self.pseudolabel_data_loader = torch.utils.data.DataLoader(combined_data, batch_size=64, sampler=pseudolabel_sampler)
    
    def train(self, fold) -> None:
        training_rounds = 100
        early_stop = EarlyStop()

        metrics = []
        num_training_rounds = 0
        participating_training_rounds = 0
        temp_metrics = []
        all_metrics = []

        # Train initial model
        for epoch in range(training_rounds):
            training_loss = self.network.train(self.train_data_loader)
            num_training_rounds += 1
            participating_training_rounds += 1
            validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
            train_embeddings, train_labels, train_genes = self.network.run(self.train_data_loader)
            accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy())
            temp_metrics.append([participating_training_rounds, fold, 0, training_loss, validation_loss, *accuracies])
            all_metrics.append([num_training_rounds, fold, 0, training_loss, validation_loss, *accuracies])
            early_stop(validation_loss, self.network.model, participating_training_rounds)
            if early_stop.early_stop:
                print(f"Early stop after epoch: {epoch}")
                early_stop.load_checkpoint(self.network.model)
                metrics.extend(temp_metrics[:-(early_stop.get_patience()+1)])
                participating_training_rounds -= (early_stop.get_patience()+1)
                break

        # Loop and add new samples
        validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
        print(validation_loss)
        stop_self_training = EarlyStop(patience=10, val_loss=validation_loss, model=self.network.model, epoch=participating_training_rounds)

        for loop in range(1, 11):
            print(f'loop {loop}')
            self.find_sample_to_add(fold, loop)
            validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
            stop_training_new_samples = EarlyStop(patience=10, val_loss=validation_loss, model=self.network.model, epoch=participating_training_rounds)
            temp_metrics = []
            for epoch in range(10):
                training_loss = self.network.train(self.pseudolabel_data_loader)
                num_training_rounds += 1
                participating_training_rounds += 1
                validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
                training_loss, train_embeddings, train_labels, train_genes = self.network.evaluate(self.train_data_loader)
                # train_embeddings, train_labels, train_genes = self.network.run(self.train_data_loader)
                accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy())
                temp_metrics.append([participating_training_rounds, fold, loop, training_loss, validation_loss, *accuracies])
                all_metrics.append([num_training_rounds, fold, loop, training_loss, validation_loss, *accuracies])
                stop_training_new_samples(validation_loss, self.network.model, participating_training_rounds)
            metrics.extend(temp_metrics)
                # if stop_training_new_samples.early_stop:
                #     print(f"Early stop after epoch: {epoch}")
                #     stop_training_new_samples.load_checkpoint(self.network.model)
                #     metrics.extend(temp_metrics[:-(stop_training_new_samples.get_patience()+1)])
                #     participating_training_rounds -= (stop_training_new_samples.get_patience()+1)
                #     break

            validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
            stop_self_training.check(validation_loss, self.network.model, participating_training_rounds)
            # if stop_self_training.early_stop:
            #     print(f"Early stop after loop: {loop}")
            #     stop_self_training.load_checkpoint(self.network.model)
            #     print(f"Best model is: {stop_self_training.best_epoch}")
            #     break

        train_embeddings, train_labels, train_genes = self.network.run(self.train_data_loader)
        test_loss, test_embeddings, test_labels, test_genes = self.network.evaluate(self.test_data_loader)
        visualize(train_embeddings, train_labels, test_embeddings, test_labels, self.outputFolder, fold, 1000)
        accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), test_embeddings.numpy(), test_labels.numpy())
        
        pd.DataFrame(
            data=metrics, 
            columns=['epoch', 'fold', 'loop', 'train_loss', 'validation_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"performance.csv", mode='a', header=fold==0)
        pd.DataFrame(
            data=all_metrics, 
            columns=['epoch', 'fold', 'loop', 'train_loss', 'validation_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"complete_performance.csv", mode='a', header=fold==0)
        pd.DataFrame(
            data=[[fold, test_loss, *accuracies]],
            columns=['fold', 'test_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"test_performance.csv", mode='a', header=fold==0)

    def find_sample_to_add(self, fold, loop) -> None:
        training_embeddings, training_labels, training_genes = self.network.run(self.train_data_loader)
        unlabeled_embeddings, unlabeled_true_labels, unlabeled_genes, unlabeled_idx = self.network.run_with_idx(self.unlabeled_data_loader)
        unlabeled_idx  = np.array(unlabeled_idx)

        # Find samples to add
        number_of_samples_to_add = 2
        pseudolabels = []
        samples_to_add = []
        while len(pseudolabels) < number_of_samples_to_add:
            # Random point in space
            random_coordinate = []
            for _ in range(2):
                random_coordinate.append(np.random.random())

            # Find closest unlabeled sample
            selected_gene_id = pd.DataFrame(
                cdist(unlabeled_embeddings, [random_coordinate], metric='euclidean')
            ).idxmin().values[0]

            # Find confidence of selected gene
            closest_samples = pd.DataFrame(
                cdist(training_embeddings, [np.array(unlabeled_embeddings[selected_gene_id])], metric='euclidean')
            ).sort_values(by=0)
            knn_labels = np.array([training_labels[sample] for sample in closest_samples.index.values[:self.knn]])
            knn_distances = np.array(closest_samples.values[:self.knn, :].T[0])

            weighted_knn_labels = knn_labels + np.multiply((1 - 2 * knn_labels), knn_distances.T).T

            confidence = weighted_knn_labels.mean()
            if (confidence > 0.8 or confidence < 0.2) and selected_gene_id not in samples_to_add:
                if round(confidence) == 1 and sum(pseudolabels) < ceil(number_of_samples_to_add/2.0):
                    samples_to_add.append(selected_gene_id)
                    pseudolabels.append(round(confidence))
                elif round(confidence) == 0 and len(pseudolabels) - sum(pseudolabels) < floor(number_of_samples_to_add/2.0):
                    samples_to_add.append(selected_gene_id)
                    pseudolabels.append(round(confidence))
        
        # visualize_show(training_embeddings, training_labels, [unlabeled_embeddings[x] for x in samples_to_add], pseudolabels, self.outputFolder, fold, loop)

        # Transfer selected samples from unlabeled to labeled samples

        indices_of_samples_to_add = [unlabeled_idx[x] for x in samples_to_add]
        
        self.unlabeled_sampler.remove_indices(indices_of_samples_to_add)
        self.train_sampler.add_indices(indices_of_samples_to_add)
        self.pseudolabel_sampler.set_indices(indices_of_samples_to_add)
        # Update the newly added samples with the pseudolabels as their labels
        for x, y in zip(pseudolabels, indices_of_samples_to_add):
            self.train_data_loader.dataset.labels[y] = x
            
        self.number_of_unlabeled_samples_added += number_of_samples_to_add
