import helper_methods as helper
from sklearn.linear_model import LogisticRegression
from adapt.instance_based import KMM, KLIEP
import dataset_generator as gen
import classifier_evaluators as evaluator
import biases


# Hyper-parameters for the experiments
n_trials = 30       # random sub-sampling trials
n_exp_steps = 25    # steps at which the independent variable (e.g. class imbalance ratio) is changed

# Classifier models (weighted and unweighted)
log_reg = LogisticRegression()
kmm = KMM(LogisticRegression(), kernel="rbf", gamma=0.1, B=1000, verbose=0, random_state=0)
kliep = KLIEP(LogisticRegression(), kernel="rbf", gamma=0.1, max_centers=100, algo="original", verbose=0, random_state=0)

# Dataset
x_set, y_set = gen.three_unbalanced_clusters([1100, 400], [1500])

# Experiment
# a) choose type of bias used for the training set
bias = biases.bias_labels

# b) compute scores
res_log_reg = evaluator.evaluate_biased_classifier(x_set, y_set, log_reg, bias, n_trials, n_exp_steps)
res_kmm = evaluator.evaluate_biased_classifier(x_set, y_set, kmm, bias, n_trials, n_exp_steps)
res_kliep = evaluator.evaluate_biased_classifier(x_set, y_set, kliep, bias, n_trials, n_exp_steps)
res_optimal = evaluator.evaluate_optimal_domain_classifier(x_set, y_set, log_reg, n_trials, n_exp_steps)

# Plot results
helper.plot_accuracy_curve(list(range(50, 100, int(50 / n_exp_steps))), "class 2 proportion (%)",
                           res_log_reg, res_kmm, res_kliep, res_optimal)
