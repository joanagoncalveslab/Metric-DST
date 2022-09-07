from pytorch_metric_learning.utils import accuracy_calculator

class CustomCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 5)

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_5"]
