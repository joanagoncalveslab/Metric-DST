from math import ceil, floor
import numpy as np
import torch

from scipy.spatial.distance import cdist
import pandas as pd

import torch.utils.data
from pytorch_metric_learning import distances, losses, reducers

from network import Network

from visualize import visualize, visualize_show
from metric_learning_utils import EarlyStop, AccuracyCalculator, MySubsetRandomSampler, create_customDataset

class SelfTraining():
    def __init__(self, network:Network, test_data:pd.DataFrame, train_data:pd.DataFrame, unlabeled_data:pd.DataFrame, validation_data:pd.DataFrame, outputFolder: str, knn: int, confidence: float, num_pseudolabels):
        self.network = network
        self.accuracyCalculator = AccuracyCalculator(knn=knn)
        self.outputFolder = outputFolder
        self.number_of_unlabeled_samples_added = 0
        self.knn = knn
        self.confidence = confidence
        self.num_pseudolabels = num_pseudolabels
        self.max_training_rounds = 100
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
    
    def train(self, fold, retrain:bool=False, add_samples_to_convergence:bool=False) -> None:
        self.num_training_rounds = 0
        self.participating_training_rounds = 0
        self.metrics = []
        self.all_metrics = []

        # Train initial model
        self.train_until_convergence(self.train_data_loader, fold, 0, 5, 0)

        train_embeddings, train_labels, train_genes = self.network.run(self.train_data_loader)
        validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
        visualize(train_embeddings, train_labels, validation_embeddings, validation_labels, self.outputFolder, fold, self.participating_training_rounds)

        # Loop and add new samples
        # validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
        stop_self_training = EarlyStop(patience=10, val_loss=validation_loss, model=self.network.model, epoch=self.participating_training_rounds)

        for loop in (range(1, 101) if add_samples_to_convergence else range(1,10)):
            print(f'loop {loop}')
            self.find_sample_to_add(fold, loop)

            if retrain:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                device = torch.device(device)
                distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
                reducer = reducers.MeanReducer()
                loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)
                ntwrk = Network([128,8,2], loss_func, 0.01, device)

                self.network = ntwrk
                self.train_until_convergence(self.train_data_loader, fold, loop, 5, 0)
                
                validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
                stop_self_training.check(validation_loss, self.network.model, self.participating_training_rounds)
                if stop_self_training.early_stop:
                    print(f"Early stop after loop: {loop}")
                    stop_self_training.load_checkpoint(self.network.model)
                    print(f"Best model is: {stop_self_training.best_epoch}")
                    break

            elif add_samples_to_convergence:
                self.train_until_convergence(self.pseudolabel_data_loader, fold, loop, 10, 0, self.network.model, epoch=self.participating_training_rounds)
                validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
                stop_self_training.check(validation_loss, self.network.model, self.participating_training_rounds)
                if stop_self_training.early_stop:
                    print(f"Early stop after loop: {loop}")
                    stop_self_training.load_checkpoint(self.network.model)
                    print(f"Best model is: {stop_self_training.best_epoch}")
                    break
            else:
                self.train_num_epochs(self.pseudolabel_data_loader, fold, loop, 10)

        train_embeddings, train_labels, train_genes = self.network.run(self.train_data_loader)
        test_loss, test_embeddings, test_labels, test_genes = self.network.evaluate(self.test_data_loader)
        visualize(train_embeddings, train_labels, test_embeddings, test_labels, self.outputFolder, fold, self.num_training_rounds)
        accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), test_embeddings.numpy(), test_labels.numpy())
        
        pd.DataFrame(
            data=self.metrics, 
            columns=['epoch', 'fold', 'loop', 'train_loss', 'validation_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"performance.csv", mode='a', header=fold==0)
        pd.DataFrame(
            data=self.all_metrics, 
            columns=['epoch', 'fold', 'loop', 'train_loss', 'validation_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"complete_performance.csv", mode='a', header=fold==0)
        pd.DataFrame(
            data=[[fold, test_loss, *accuracies]],
            columns=['fold', 'test_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+"test_performance.csv", mode='a', header=fold==0)

    def train_until_convergence(self, data:torch.utils.data.DataLoader, fold:int, loop:int, patience:int, delta:float, model=None, epoch:int=0):
        early_stop = EarlyStop(patience=patience, delta=delta, model=model, epoch=epoch)
        temp_metrics = []

        for epoch in range(self.max_training_rounds):
            training_loss = self.network.train(data)
            self.num_training_rounds += 1
            self.participating_training_rounds += 1
            validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
            training_loss, train_embeddings, train_labels, train_genes = self.network.evaluate(self.train_data_loader)
            accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy())
            temp_metrics.append([self.participating_training_rounds, fold, loop, training_loss, validation_loss, *accuracies])
            self.all_metrics.append([self.num_training_rounds, fold, loop, training_loss, validation_loss, *accuracies])
            early_stop(validation_loss, self.network.model, self.participating_training_rounds)
            if early_stop.early_stop:
                print(f"Early stop after epoch: {epoch}")
                early_stop.load_checkpoint(self.network.model)
                self.metrics.extend(temp_metrics[:-(early_stop.get_patience()+1)])
                self.participating_training_rounds -= (early_stop.get_patience()+1)
                break
    
    def train_num_epochs(self, data:torch.utils.data.DataLoader, fold:int, loop:int, num_epochs:int=10):
        temp_metrics = []

        for epoch in range(num_epochs):
            training_loss = self.network.train(data)
            self.num_training_rounds += 1
            self.participating_training_rounds += 1
            validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
            training_loss, train_embeddings, train_labels, train_genes = self.network.evaluate(self.train_data_loader)
            accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), validation_embeddings.numpy(), validation_labels.numpy())
            temp_metrics.append([self.participating_training_rounds, fold, loop, training_loss, validation_loss, *accuracies])
            self.all_metrics.append([self.num_training_rounds, fold, loop, training_loss, validation_loss, *accuracies])
        self.metrics.extend(temp_metrics)

    def find_sample_to_add(self, fold, loop) -> None:
        training_embeddings, training_labels, training_genes = self.network.run(self.train_data_loader)
        unlabeled_embeddings, unlabeled_true_labels, unlabeled_genes, unlabeled_idx = self.network.run_with_idx(self.unlabeled_data_loader)
        unlabeled_idx  = np.array(unlabeled_idx)

        # Find samples to add
        number_of_samples_to_add = self.num_pseudolabels
        pseudolabels = []
        samples_to_add = []
        i = 0
        while len(pseudolabels) < number_of_samples_to_add and i < number_of_samples_to_add*20:
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
            # Cap the distances at .5 so positives can't become negatives and vice versa if the distances are too large
            knn_distances[knn_distances > 0.5] = 0.5
            weighted_knn_labels = knn_labels + np.multiply((1 - 2 * knn_labels), knn_distances.T).T

            confidence = weighted_knn_labels.mean()
            if (confidence > self.confidence or confidence < 1-self.confidence) and selected_gene_id not in samples_to_add:
                if round(confidence) == 1 and sum(pseudolabels) < ceil(number_of_samples_to_add/2.0):
                    samples_to_add.append(selected_gene_id)
                    pseudolabels.append(round(confidence))
                elif round(confidence) == 0 and len(pseudolabels) - sum(pseudolabels) < floor(number_of_samples_to_add/2.0):
                    samples_to_add.append(selected_gene_id)
                    pseudolabels.append(round(confidence))

        # If we ran out of time with finding samples we don't have a balanced set so we still need to balance
        if i == number_of_samples_to_add:
            print(f"Ran out of time with finding samples to add in loop {loop}.\nAdded {len(pseudolabels)} samples in total of which {sum(pseudolabels)} were positives and {len(pseudolabels) - sum(pseudolabels)} were negatives.\nWas looking for {number_of_samples_to_add} samples.")
        
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
