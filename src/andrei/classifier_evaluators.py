from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import biases
from dataset_generator import multidimensional_set
import numpy as np


def evaluate_optimal_domain_classifier(x_set, y_set, model, n_trials, n_exp_steps):
    res_optimal = 0.0
    # Put aside 50% data for the unlabeled domain
    x_domain, x_rest, y_domain, y_rest = train_test_split(x_set, y_set, train_size=0.5, random_state=42)

    # Perform n_trials trials of training-test splits (random sub-sampling)
    for j in range(n_trials):
        x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=0.2, random_state=j)
        model.fit(x_domain, y_domain)
        y_predicted = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        res_optimal += accuracy

    # Average performance score over n_trials
    res_optimal = res_optimal / n_trials
    return [res_optimal] * n_exp_steps


def evaluate_biased_classifier(x_set, y_set, model, bias: biases, n_trials, n_exp_steps):
    # Put aside 50% data for the unlabeled domain
    x_domain, x_rest, y_domain, y_rest = train_test_split(x_set, y_set, train_size=0.5, random_state=42)
    results = [0.0] * n_exp_steps

    # Iterate over the number of experimental steps n_exp_steps
    for i in range(n_exp_steps):
        # Perform n_trials trials of training-test splits (random sub-sampling)
        for j in range(n_trials):
            # Split data into 40% training and 10% testing
            x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=0.2, random_state=j)

            # Bias the training set (depends on the used bias)
            x_train_biased, y_train_biased = bias(x_train, y_train, i, n_exp_steps)

            # Train on the biased training set
            if isinstance(model, LogisticRegression):
                model.fit(x_train_biased, y_train_biased)
            else:
                model.fit(x_train_biased, y_train_biased, x_domain)

            # Test on the unbiased test set
            y_predicted = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_predicted)
            results[i] += accuracy

        # Average performance score over n_trials
        results[i] = round(results[i] / n_trials, 5)
    return results


def evaluate_high_dimensional_classifier(model, bias: biases, n_trials, n_datasets, n_feature_steps, step_size):
    # Iterate over all feature sizes
    results = [0.0] * n_feature_steps

    for k in range(n_feature_steps):
        n_features = (1 + k) * step_size

        # Average results over multiple high-dimensional classification datasets
        score_datasets = [0.0] * n_datasets
        for i in range(n_datasets):
            # Generate dataset with specified number of features
            x_set, y_set = multidimensional_set(n_features, random_state=i)

            # Put aside 50% data for the unlabeled domain
            x_domain, x_rest, y_domain, y_rest = train_test_split(x_set, y_set, train_size=0.5, stratify=y_set,
                                                                  random_state=42)

            # Perform multiple trials of training-test splits (random sub-sampling)
            score_subsampling = 0.0
            for j in range(n_trials):
                # Split data into 40% training and 10% testing
                x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=0.2, stratify=y_rest,
                                                                    random_state=j)

                # Bias the training set (depends on the used bias)
                x_train_biased, y_train_biased = bias(x_train=x_train, y_train=y_train, a=3, b=4)

                # Train on the biased training set
                if isinstance(model, LogisticRegression):
                    model.fit(x_train_biased, y_train_biased)
                else:
                    model.fit(x_train_biased, y_train_biased, x_domain)

                # Test on the unbiased test set
                y_predicted = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_predicted)
                score_subsampling += accuracy

            # Average performance score over n_trials of sub-sampling
            score_subsampling_avg = score_subsampling / n_trials
            score_datasets[i] = score_subsampling_avg

        # Average performance score over n_datasets of datasets
        results[k] = np.mean(score_datasets)

    return results


def evaluate_optimal_high_dimensional_classifier(model, n_trials, n_datasets, n_feature_steps, step_size):
    # Iterate over all feature sizes
    results = [0.0] * n_feature_steps

    for k in range(n_feature_steps):
        n_features = (1 + k) * step_size

        # Average results over multiple high-dimensional classification datasets
        score_datasets = [0.0] * n_datasets
        for i in range(n_datasets):
            # Generate dataset with specified number of features
            x_set, y_set = multidimensional_set(n_features, random_state=i)

            # Put aside 50% data for the unlabeled domain
            x_domain, x_rest, y_domain, y_rest = train_test_split(x_set, y_set, train_size=0.5, stratify=y_set,
                                                                  random_state=42)

            # Perform multiple trials of training-test splits (random sub-sampling)
            score_subsampling = 0.0
            for j in range(n_trials):
                # Split data into 40% training and 10% testing
                x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=0.2, stratify=y_rest,
                                                                    random_state=j)

                # Train on the domain and test on the test set
                model.fit(x_domain, y_domain)
                y_predicted = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_predicted)
                score_subsampling += accuracy

            # Average performance score over n_trials of sub-sampling
            score_subsampling_avg = score_subsampling / n_trials
            score_datasets[i] = score_subsampling_avg

        # Average performance score over n_datasets of datasets
        results[k] = np.mean(score_datasets)

    return results
