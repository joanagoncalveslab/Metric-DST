import torch
import torch.utils.data

from scipy.spatial.distance import cdist
import customDataset
import pandas as pd

from network import Network
from typing import Sequence

from visualize import visualize, visualize_gene
from metric_learning_utils import AccuracyCalculator, MySubsetRandomSampler

class MetricLearning():

    network:Network = None

    def __init__(self, network:Network, test_data:customDataset.CustomDataset, training_data:customDataset.CustomDataset, training_idx: Sequence[int], validation_idx: Sequence[int], outputFolder: str):
        self.network = network
        self.accuracyCalculator = AccuracyCalculator()
        self.outputFolder = outputFolder

        g_test = torch.Generator().manual_seed(43)
        g_train = torch.Generator().manual_seed(42)
        g_validation = torch.Generator().manual_seed(42)
        test_sampler = torch.utils.data.RandomSampler(test_data, generator=g_test)
        self.training_sampler = MySubsetRandomSampler(training_idx, g_train)
        self.validation_sampler = MySubsetRandomSampler(validation_idx, g_validation)

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
