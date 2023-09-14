from math import ceil, floor
import numpy as np
import torch
import pickle

from scipy.spatial.distance import cdist
import pandas as pd

import torch.utils.data
from pytorch_metric_learning import distances, losses, reducers

from network import Network

from visualize import visualize, visualize_show
from metric_learning_utils import EarlyStop, AccuracyCalculator, MySubsetRandomSampler, create_customDataset

class SelfTraining():
    def __init__(self, network:Network, test_data:pd.DataFrame, train_data:pd.DataFrame, unlabeled_data:pd.DataFrame, validation_data:pd.DataFrame, outputFolder: str, name:str, knn: int, confidence: float, num_pseudolabels:int):
        self.network = network
        self.accuracyCalculator = AccuracyCalculator(knn=knn)
        self.outputFolder = outputFolder
        self.name = name
        self.number_of_unlabeled_samples_added = 0
        self.knn = knn
        self.confidence = confidence
        self.num_pseudolabels = num_pseudolabels
        self.max_training_rounds = 100
        self.network_layers = [train_data.shape[1]-3,8,2]
        self.transform_data(test_data, train_data, unlabeled_data, validation_data)
        self.iteration_stats = {0: {}}

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
    
    def train(self, fold, retrain:bool=False, add_samples_to_convergence:bool=False, supervised:bool=False, pseudolabel_method:str=None) -> None:
        self.num_training_rounds = 0
        self.participating_training_rounds = 0
        self.metrics = []
        self.all_metrics = []

        # Train initial model
        self.train_until_convergence(self.train_data_loader, fold, 0, 5, 0)

        train_embeddings, train_labels, train_genes = self.network.run(self.train_data_loader)
        validation_loss, validation_embeddings, validation_labels, validation_genes = self.network.evaluate(self.validation_data_loader)
        # visualize(train_embeddings, train_labels, validation_embeddings, validation_labels, self.outputFolder, fold, self.participating_training_rounds)

        if not supervised:
            # Loop and add new samples
            # validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
            stop_self_training = EarlyStop(patience=7, val_loss=validation_loss, model=self.network.model, epoch=self.participating_training_rounds)

            for loop in (range(1, 10000) if add_samples_to_convergence else range(1,11)):
                print(f'loop {loop}')
                self.iteration_stats[loop] = {}

                if pseudolabel_method == 'balanced_semi_supervised':
                    self.find_samples_balanced_semi_supervised(fold, loop)
                elif pseudolabel_method == 'divergence':
                    self.find_samples_divergence(fold, loop)
                elif pseudolabel_method == 'semi_supervised':
                    self.find_samples_semi_supervised(fold, loop)
                else:
                    print(f'Not a valid psuedolabeling method entered: {pseudolabel_method}')
                    return
                
                if retrain and add_samples_to_convergence:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    device = torch.device(device)
                    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
                    reducer = reducers.MeanReducer()
                    loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)
                    ntwrk = Network(self.network_layers, loss_func, 0.01, device)

                    self.network = ntwrk
                    self.train_until_convergence(self.train_data_loader, fold, loop, 7, 0)

                    validation_loss, _, _, _ = self.network.evaluate(self.validation_data_loader)
                    self.iteration_stats[loop]['early_stop'] = stop_self_training.check(validation_loss, self.network.model, self.participating_training_rounds)
                    if stop_self_training.early_stop:
                        print(f"Early stop after loop: {loop}")
                        stop_self_training.load_checkpoint(self.network.model)
                        print(f"Best model is: {stop_self_training.best_epoch}")
                        self.iteration_stats['Best model'] = stop_self_training.best_epoch
                        break
                elif retrain:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    device = torch.device(device)
                    distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
                    reducer = reducers.MeanReducer()
                    loss_func = losses.ContrastiveLoss(pos_margin=0.3, neg_margin=0.5, distance=distance, reducer=reducer)
                    ntwrk = Network(self.network_layers, loss_func, 0.01, device)

                    self.network = ntwrk
                    self.train_until_convergence(self.train_data_loader, fold, loop, 5, 0)

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
        # visualize(train_embeddings, train_labels, test_embeddings, test_labels, self.outputFolder, fold, self.num_training_rounds)
        accuracies = self.accuracyCalculator.get_accuracies(train_embeddings.numpy(), train_labels.numpy(), test_embeddings.numpy(), test_labels.numpy())
        
        pd.DataFrame(
            data=self.metrics, 
            columns=['epoch', 'fold', 'loop', 'train_loss', 'validation_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+self.name+"_performance.csv", mode='a', header=fold==0)
        pd.DataFrame(
            data=self.all_metrics, 
            columns=['epoch', 'fold', 'loop', 'train_loss', 'validation_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+self.name+"_complete_performance.csv", mode='a', header=fold==0)
        pd.DataFrame(
            data=[[fold, test_loss, *accuracies]],
            columns=['fold', 'test_loss', 'accuracy', 'f1_score', 'average_precision', 'auroc']
        ).to_csv(self.outputFolder+self.name+"_test_performance.csv", mode='a', header=fold==0)
        with open(f"{self.outputFolder}{self.name}_iteration_stats_fold_{fold}.pickle", 'wb') as output_file:
            pickle.dump(self.iteration_stats, output_file)
        import joblib
        filename1 = f'{self.outputFolder}{self.name}_model_weigths_fold_{fold}.sav'
        joblib.dump(self.network.model.state_dict(), filename1)
        filename2 = f'{self.outputFolder}{self.name}_final_embedding_fold_{fold}.sav'
        joblib.dump(self.network, filename2)

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
                self.iteration_stats[loop]['early_stop_after_epoch'] = epoch
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

    def find_samples_divergence(self, fold, loop) -> None:
        training_embeddings, training_labels, training_genes = self.network.run(self.train_data_loader)
        unlabeled_embeddings, unlabeled_true_labels, unlabeled_genes, unlabeled_idx = self.network.run_with_idx(self.unlabeled_data_loader)
        unlabeled_idx  = np.array(unlabeled_idx)

        # Find samples to add
        number_of_samples_to_add = self.num_pseudolabels
        pseudolabels_pos = []
        selected_confidences_pos = []
        samples_to_add_pos = []
        selected_genes_pos = []

        pseudolabels_neg = []
        selected_confidences_neg = []
        samples_to_add_neg = []
        selected_genes_neg = []

        i = 0
        while (len(pseudolabels_pos) + len(pseudolabels_neg)) < number_of_samples_to_add and i < number_of_samples_to_add*50:
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
            if (confidence > self.confidence or confidence < 1-self.confidence) and selected_gene_id not in samples_to_add_pos and selected_gene_id not in samples_to_add_neg:
                if round(confidence) == 1 and len(pseudolabels_pos) < ceil(number_of_samples_to_add/2.0):
                    samples_to_add_pos.append(selected_gene_id)
                    pseudolabels_pos.append(round(confidence))
                    selected_confidences_pos.append(confidence)
                    selected_genes_pos.append(unlabeled_genes[selected_gene_id])
                elif round(confidence) == 0 and len(pseudolabels_neg) < floor(number_of_samples_to_add/2.0):
                    samples_to_add_neg.append(selected_gene_id)
                    pseudolabels_neg.append(round(confidence))
                    selected_confidences_neg.append(confidence)
                    selected_genes_neg.append(unlabeled_genes[selected_gene_id])

            i = i + 1

        # If we ran out of time with finding samples we don't have a balanced set so we still need to balance
        if i == number_of_samples_to_add*200:
            print(f"Ran out of time with finding samples to add in loop {loop}.\nFound {len(pseudolabels_pos) + len(pseudolabels_neg)} samples in total of which {len(pseudolabels_pos)} were positives and {len(pseudolabels_neg)} were negatives.\nWas looking for {number_of_samples_to_add} samples.")

        min_size = min(len(pseudolabels_pos), len(pseudolabels_neg))
        pseudolabels = pseudolabels_pos[:min_size] + pseudolabels_neg[:min_size]
        selected_confidences = selected_confidences_pos[:min_size] + selected_confidences_neg[:min_size]
        samples_to_add = samples_to_add_pos[:min_size] + samples_to_add_neg[:min_size]
        selected_genes = selected_genes_pos[:min_size] + selected_genes_neg[:min_size]
        
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
        self.iteration_stats[loop].update({
            'samples': samples_to_add,
            'pseudolabels': pseudolabels,
            'confidences': selected_confidences,
            'genes': selected_genes
        })

    def find_samples_balanced_semi_supervised(self, fold, loop) -> None:
        training_embeddings, training_labels, training_genes = self.network.run(self.train_data_loader)
        unlabeled_embeddings, unlabeled_true_labels, unlabeled_genes, unlabeled_idx = self.network.run_with_idx(self.unlabeled_data_loader)
        unlabeled_idx  = np.array(unlabeled_idx)

        # Find samples to add
        number_of_samples_to_add = self.num_pseudolabels

        pairwise = pd.DataFrame(
            cdist(unlabeled_embeddings, training_embeddings, metric='euclidean')
        )

        # Get the indices of the columns with the largest distance to each row
        sorted_pairwise = np.argsort(pairwise.values, 1)
        idx = sorted_pairwise[:, :self.knn]
        # Get the 5 nearest distances and labels
        knn_distances = np.array([[min(pairwise.iloc[i, j], 0.5) for j in row] for i, row in enumerate(idx)])

        knn_labels = np.array([[training_labels[element] for element in row] for row in idx])
        # Calculate the weighted knn where the distance between the sample and nearest neighbour is subtracted from the neighbour label according to:
        # Label_k * (1 - |distance to k|) + (1 - label_k) * |distance to k|, where k represents the nearest neighbour k
        weighted_knn_labels = knn_labels + ((1 - 2 * knn_labels) * knn_distances)

        confidences  = weighted_knn_labels.mean(axis=1)
        
        argsorted_confidences = np.argsort(confidences)
        top = argsorted_confidences[-ceil(number_of_samples_to_add/2.0):]
        bottom = argsorted_confidences[:floor(number_of_samples_to_add/2.0)]

        samples_to_add = np.concatenate([top, bottom])
        selected_confidences = np.take(confidences, samples_to_add)
        pseudolabels = np.take(confidences, samples_to_add).round()
        selected_genes = np.take(unlabeled_genes, samples_to_add, axis=0)

        indices_of_samples_to_add = [unlabeled_idx[x] for x in samples_to_add]
        
        self.unlabeled_sampler.remove_indices(indices_of_samples_to_add)
        self.train_sampler.add_indices(indices_of_samples_to_add)
        self.pseudolabel_sampler.set_indices(indices_of_samples_to_add)
        # Update the newly added samples with the pseudolabels as their labels
        for x, y in zip(pseudolabels, indices_of_samples_to_add):
            self.train_data_loader.dataset.labels[y] = x
            
        self.number_of_unlabeled_samples_added += number_of_samples_to_add
        self.iteration_stats[loop].update({
            'samples': samples_to_add,
            'pseudolabels': pseudolabels,
            'confidences': selected_confidences,
            'genes': selected_genes
        })

    def find_samples_semi_supervised(self, fold, loop) -> None:
        training_embeddings, training_labels, training_genes = self.network.run(self.train_data_loader)
        unlabeled_embeddings, unlabeled_true_labels, unlabeled_genes, unlabeled_idx = self.network.run_with_idx(self.unlabeled_data_loader)
        unlabeled_idx  = np.array(unlabeled_idx)

        # Find samples to add
        number_of_samples_to_add = self.num_pseudolabels

        pairwise = pd.DataFrame(
            cdist(unlabeled_embeddings, training_embeddings, metric='euclidean')
        )

        # Get the indices of the columns with the largest distance to each row
        sorted_pairwise = np.argsort(pairwise.values, 1)
        idx = sorted_pairwise[:, :self.knn]
        # Get the 5 nearest distances and labels
        knn_distances = np.array([[min(pairwise.iloc[i, j], 0.5) for j in row] for i, row in enumerate(idx)])

        knn_labels = np.array([[training_labels[element] for element in row] for row in idx])
        # Calculate the weighted knn where the distance between the sample and nearest neighbour is subtracted from the neighbour label according to:
        # Label_k * (1 - |distance to k|) + (1 - label_k) * |distance to k|, where k represents the nearest neighbour k
        weighted_knn_labels = knn_labels + ((1 - 2 * knn_labels) * knn_distances)

        confidences  = weighted_knn_labels.mean(axis=1)
        
        class_independant_confidences = [max(confidence, 1 - confidence) for confidence in confidences]
        argsorted_confidences = np.argsort(class_independant_confidences)
        samples_to_add = argsorted_confidences[-number_of_samples_to_add:]

        selected_confidences = np.take(confidences, samples_to_add)
        pseudolabels = np.take(confidences, samples_to_add).round()
        selected_genes = np.take(unlabeled_genes, samples_to_add, axis=0)

        indices_of_samples_to_add = [unlabeled_idx[x] for x in samples_to_add]
        
        self.unlabeled_sampler.remove_indices(indices_of_samples_to_add)
        self.train_sampler.add_indices(indices_of_samples_to_add)
        self.pseudolabel_sampler.set_indices(indices_of_samples_to_add)
        # Update the newly added samples with the pseudolabels as their labels
        for x, y in zip(pseudolabels, indices_of_samples_to_add):
            self.train_data_loader.dataset.labels[y] = x
            
        self.number_of_unlabeled_samples_added += number_of_samples_to_add
        self.iteration_stats[loop].update({
            'samples': samples_to_add,
            'pseudolabels': pseudolabels,
            'confidences': selected_confidences,
            'genes': selected_genes
        })

    def find_all_confidences(self, fold, loop):
        training_embeddings, training_labels, training_genes = self.network.run(self.train_data_loader)
        unlabeled_embeddings, unlabeled_true_labels, unlabeled_genes, unlabeled_idx = self.network.run_with_idx(self.unlabeled_data_loader)
        unlabeled_idx  = np.array(unlabeled_idx)

        pairwise = pd.DataFrame(
            cdist(unlabeled_embeddings, training_embeddings, metric='euclidean')
        )

        # Get the indices of the columns with the largest distance to each row
        sorted_pairwise = np.argsort(pairwise.values, 1)
        idx = sorted_pairwise[:, :self.knn]
        # Get the 5 nearest distances and labels
        knn_distances = np.array([[min(pairwise.iloc[i, j], 0.5) for j in row] for i, row in enumerate(idx)])

        knn_labels = np.array([[training_labels[element] for element in row] for row in idx])
        # Calculate the weighted knn where the distance between the sample and nearest neighbour is subtracted from the neighbour label according to:
        # Label_k * (1 - |distance to k|) + (1 - label_k) * |distance to k|, where k represents the nearest neighbour k
        weighted_knn_labels = knn_labels + ((1 - 2 * knn_labels) * knn_distances)

        confidences  = weighted_knn_labels.mean(axis=1)

        res = pd.DataFrame(
            data=confidences,
            columns=['confidences']
        )
        res['fold'] = fold
        res['loop'] = loop

        res.to_csv(self.outputFolder+self.name+"_confidences.csv", mode='a', header=(fold==0 and loop==1))
