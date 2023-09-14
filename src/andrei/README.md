# Evaluating the Effectiveness of Importance Weighting in Mitigating Sample Selection Bias

## About the Project
This is the implementation of an evaluation framework described in my Computer Science and Engineering
Bachelor Thesis at TU Delft, which aims to provide insight into the adaptation performance of
importance weighting techniques in dealing with sample selection bias.
So far the project provides example applications of the framework to KMM and KLIEP, but it
can, in principle, support any type of importance weighting technique. Lastly, the framework
is designed with classification tasks in mind, but the transition to regression experiments
should be possible with minimal effort.

## Contents
This repository contains support for the following operations:
1. creation of synthetic classification datasets
2. evaluating classifiers (with possibility of choosing the desired type of bias)
3. plotting the results of the experiments
4. performing statistical significance tests on the results

### 1. Creating Datasets
All methods for generating datasets are contained in the class ``dataset_generator.py``:
* for 2-dimensional 2-class datasets (i.e. test cases 1 & 2): ``three_unbalanced_clusters``, ``overlapped_linear_clusters``, ``rotated_moons``.
* for multi-dimensional 2-class datasets (i.e. test case 3): ``multidimensional_set``.

### 2. Evaluating Classifiers (and Selecting Bias)
All methods for evaluating the classifiers are contained in the class ``classifier_evaluator.py``:
* for 2-dimensional datasets (i.e. test cases 1 & 2): ``evaluate_biased_classifier``, ``evaluate_optimal_domain_classifier``.
* for multi-dimensional datasets (i.e. test case 3): ``evaluate_high_dimensional_classifier``, ``evaluate_optimal_high_dimensional_classifier``.

All methods for inducing sampling bias in the training set are contained in the class ``biases.py``:
1. for test case 1
    * biasing labels only in the form of class imbalance ratios: ``bias_labels``.
2. for test case 2
    * biasing features only (2-dimensional data) and subsequently reducing the training set size: ``bias_features_and_reduce_samples``.
    * reducing the training set size without biasing: ``reduce_samples_only``.
3. for test case 3
    * biasing the most important feature only (multi-dimensional dataset): ``bias_most_important_feature``.
    * biasing all features simultaneously (multi-dimensional dataset) with possible class imbalance:
``bias_all_features_and_target_PCA``.
    * biasing all features simultaneously (multi-dimensional dataset) with maintained class balance:
``bias_all_features_excl_target_PCA``.

### 3. Plotting Results
All methods for visualising data (either experimental results or related to datasets)
are contained in the class ``helper_methods.py``:
* for plotting the 2-dimensional datasets: ``plot_dataset``, ``plot_probability_density_function``.
* for plotting results (curves): ``plot_accuracy_curve``, ``plot_da_acc_percentages``.
* for plotting results (miscellaneous): ``plot_bias_sample_size_heatmap``, ``compute_marginal_std_from_heatmap``.

### 4. Performing Statistical Tests
All methods for performing the statistical significance analysis of the results are contained
in the class ``statistical_analysis.py``:
* for paired test on the superiority of the weighted classifier (i.e. _Shapiro-Wilk_ normality test followed by either
_corrected resampled t-test_ or _Wilcoxon signed-rank test_): ``make_table_1_outperforming``.
* for independent test on the closeness to the optimum line (i.e. _Mann-Whitney test_): ``make_table_2_close_optimum``.

## Setting Up the Experiments

### Main Steps
1. __Set experimental hyper-parameters__ - choose the number of train-test splits ``n_trials`` used in the
experiments and the number of steps ``n_exp_steps`` at which the independent variable (e.g. class imbalance ratio
for test case 1, training set size ratio for test case 2) should be varied during the experiment.
2. __Choose classifier__ - we use logistic regression from ``sklearn.linear_model.LogisticRegression``, but other
classifiers can be tried too, as long as they support the two methods ``fit(x_train, y_train)`` and
``predict(x_test)``.
3. __Choose importance weighting technique__ - choose the technique to use (e.g. KMM, KLIEP) and define its
specific hyper-parameters. We use the [ADAPT](https://adapt-python.github.io/adapt/) library for Python, but any library
can be used as long as it supports the two methods ``fit(x_train, y_train, x_unlabeled)`` and ``predict(x_test)``.
4. __Generate dataset__
5. __Evaluate classifier__ - choose the evaluation method and the bias that should be applied on the training
data (for evaluating the optimal classifier no bias is necessary)
6. __Plot results__

An example of how the steps above can be implemented for running one of the experiments (i.e. test case 1) is
provided in the class ``ex_experiment.py``. For the exact parameter and argument settings used in the other experiments,
please consult the thesis.

## Contribution and External Usage
Contributions are welcome! Feel free to make a pull request to this repository and enrich it!

If you are using either source code from this project or knowledge from my research paper
describing the framework, please reference it as follows:

```
@mastersthesis{actociu2023,
  author       = {Andrei Camil Tociu}, 
  title        = {Evaluating the Effectiveness of Importance Weighting Techniques In
                  Mitigating Sample Selection Bias},
  school       = {Delft University of Technology (TU Delft)},
  year         = 2023,
  month        = 6
}
```