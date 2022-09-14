from pytorch_metric_learning.utils import accuracy_calculator
from torch import Tensor
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score, average_precision_score

class CustomCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        # knn_labels: list with arrays of size k that contain the labels of the k closest pairs
        # query_labels: list of original labels
        return accuracy_calculator.precision_at_k(
            knn_labels, 
            query_labels[:, None], 
            5, 
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,)

    def calculate_accuracy(self, knn_labels:Tensor, query_labels:Tensor, **kwargs):
        return knn_labels.mean(1).round().eq(query_labels).float().mean().item()

    def calculate_f1_score(self, knn_labels:Tensor, query_labels:Tensor, **kwargs):
        y_pred = knn_labels.mean(1).round()
        return f1_score(query_labels.cpu(), y_pred.cpu())

    def calculate_average_precision(self, knn_labels:Tensor, query_labels:Tensor, **kwargs):
        y_prob = knn_labels.mean(1)
        return average_precision_score(query_labels.cpu(), y_prob.cpu())

    def calculate_auroc(self, knn_labels:Tensor, query_labels:Tensor, **kwargs):
        return roc_auc_score(query_labels.cpu(), knn_labels.mean(1).cpu())

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_5", "accuracy", "f1_score", "average_precision", "auroc"]
