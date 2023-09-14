"""Contains utility functions that can be used for evaluating a dataset and DA model's performance"""

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np


def acc_slope(acc: np.ndarray, last_portion: float = 0.05, preceding_portion: float = 0.05) -> float:
    """Compute an indication of convergence, by the slope of the accuracy.
    Averaged over two trail proportions of the acc curve.
    Values near 0 indicate convergence, positive values indicate that progress is still being made."""
    array_length = len(acc)
    last_num = max(1, int(last_portion * array_length))
    preceding_num = max(1, int(preceding_portion * array_length))

    last_samples = acc[-last_num:]
    preceding_samples = acc[-(last_num + preceding_num):-last_num]

    return (np.mean(last_samples) - np.mean(preceding_samples)) / (0.5*(last_num + preceding_num))


def analyze_data(dataset) -> dict:
    """Analyze properties of a dataset using some directly computable statistical measures, without ML.
    :param dataset: tuple (xg, yg, xs, ys, xt, yt) where s=source, g=global, t=target.
    :returns dict with metrics
    """
    res = dict()
    xg, yg, xs, ys, xt, yt = dataset
    for x, y, name in [
            (xg, yg, 'global'),
            (xs, ys, 'source'),
            (xt, yt, 'target')]:
        res[f'num-{name}'] = x.shape[0]
        res[f'uniqueness-{name}'] = np.unique(x, axis=0).shape[0] / x.shape[0]
        res[f'class-marginal-{name}'] = np.mean(y)
    return res


def evaluate_deep(dataset, model_builder, fit_params: dict, train_split: float, distance: bool = False, verbose: bool = False) -> dict:
    """Evaluate a domain adaptation model on a dataset.
    Trains and evaluates the model multiple times with various combinations of domains.
    Can be used to both evaluate the efficiency of the DA with respect to baselines,
    and give insight into the dataset's characteristics and separability.

    :param dataset: (tuple of xg, yg, xs, ys, xt, yt) data and labels for source, global and target sets
    :param model_builder: Function that returns a new model every call.
    Model should have fit and predict methods, like BaseAdaptDeep.
    Can also be a dictionary, to use a different model for every configuration. Keys formatted like "s-only" or "s->t".
    :param fit_params: parameters like epoch, batch size etc. for `model.fit`
    :param train_split: proportion to use for training data, use rest for test.
    :param distance: compute distance between domains by training a classifier.
    Costly, accounts for about 40% of total evaluation time, disable to speed up.
    :param verbose: show progress bars
    :returns dictionary with all computed results
    """

    xg, yg, xs, ys, xt, yt = dataset
    domains = {
        's': {'x': xs, 'y': ys},
        'g': {'x': xg, 'y': yg},
        't': {'x': xt, 'y': yt},
    }

    # compute index of train/test split, in case each domain has different sample count
    split_indexes = {
        'g': int(train_split*len(xg)),
        's': int(train_split*len(xs)),
        't': int(train_split*len(xt))
    }

    # first train models
    metrics = dict()
    models = dict()
    pbar = tqdm([
        ('s', 's'),
        ('t', 't'),
        ('s', 't'),
        ('s', 'g')], disable=not verbose)
    for source, target in pbar:
        name = f'{source}-only' if source == target else f'{source}->{target}'
        pbar.set_description(f"Training model '{name}'")

        # select model based on
        if type(model_builder) is dict:
            builder = model_builder[name]
        else:
            builder = model_builder

        # create new model and fit to the training split of the data
        models[name] = builder().fit(
            domains[source]['x'][split_indexes[source]:],
            domains[source]['y'][split_indexes[source]:],
            domains[target]['x'][split_indexes[target]:], **fit_params)

        # compute a convergence indication from the history
        if hasattr(models[name], 'history_') and 'acc' in models[name].history_:
            indication = acc_slope(models[name].history_['acc'])
            metrics[f'{name}-convergence-acc-slope'] = indication
        pbar.set_description("Finished training")

    # then evaluate accuracy on different test sets (not every combination is used)
    pbar = tqdm([
        ('t-only', 't'),
        ('s-only', 't'),
        ('s->t', 't'),
        ('s->g', 't')], disable=not verbose)
    for model, test in pbar:
        name = f"{model}-acc-on-{test}"
        pbar.set_description(f"Evaluating model '{model}' on '{test}'")

        x = domains[test]['x'][:split_indexes[test]]
        y = domains[test]['y'][:split_indexes[test]]

        y_pred = models[model].predict(x)
        acc = accuracy_score(y, y_pred > 0.5)
        metrics[name] = acc
        pbar.set_description("Finished evaluating")

    if distance:
        dists = _calculate_distance(domains, fit_params, model_builder, verbose)
        for key in dists:
            metrics[key] = dists[key]

    return metrics


def evaluate_single(dataset, model_builder, fit_params: dict, source: str, target: str,  train_split: float = 0.7) -> float:
    """Evaluate a domain adaptation model on a dataset and return accuracy on the target domain
    :param train_split: proportion to use for training data, use rest for validation.
    :param dataset: (tuple of xg, yg, xs, ys, xt, yt) data and labels for source, global and target sets
    :param model_builder: (() -> BaseAdaptDeep model) function that returns a new model every call
    :param fit_params: parameters like epoch, batch size etc. for `model.fit`
    :param source: labeled training data, either 's' 't' or 'g'
    :param target: unlabeled training data, either 's' 't' or 'g'
    :returns dictionary with all computed results
    """

    xg, yg, xs, ys, xt, yt = dataset
    domains = {
        's': {'x': xs, 'y': ys},
        'g': {'x': xg, 'y': yg},
        't': {'x': xt, 'y': yt},
    }

    split_indexes = {
        'g': int(train_split*len(xg)),
        's': int(train_split*len(xs)),
        't': int(train_split*len(xt))
    }

    # train
    model = model_builder().fit(
        domains[source]['x'][split_indexes[source]:],
        domains[source]['y'][split_indexes[source]:],
        domains[target]['x'][split_indexes[target]:], **fit_params)

    # evaluate
    x = domains[target]['x'][:split_indexes[target]]
    y = domains[target]['y'][:split_indexes[target]]
    y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred > 0.5)

    return acc


def _calculate_distance(domains: dict, fit_params, model_builder, verbose) -> dict:
    """
    Compute a classifier-dependent distance between domains.
    Trains the given model to classify domains labels, distance is based on classification accuracy.
    Called proxy A-distance (Ben-David et al., A theory of learning from different domains, 2010)
    Halved so that the range will be in [0,1], assuming that classification accuracy is never below 50%.
    """
    metrics = dict()
    pbar = tqdm([('s', 'g'), ('s', 't'), ('g', 't')], disable=not verbose)
    for domain_a, domain_b in pbar:
        name = f"half-A-dist-{domain_a}-{domain_b}"
        pbar.set_description(f"Estimate proxy A-distance between '{domain_a}' and '{domain_b}'")

        # combine datasets, and use domain as label, permutes randomly
        x = np.concatenate(
            [domains[domain_a]['x'],
             domains[domain_b]['x']])
        y = np.concatenate([np.full(domains[domain_a]['x'].shape[0], 0),
                            np.full(domains[domain_b]['x'].shape[0], 1)])
        p = np.random.permutation(x.shape[0])
        x, y = x[p], y[p]

        model = model_builder().fit(x, y, x, **fit_params)
        y_pred = model.predict(x)
        acc = accuracy_score(y, y_pred > 0.5)
        metrics[name] = 2 * acc - 1  # equiv to 0.5 * a_dist = 0.5 * 2*(1-2*err)
        pbar.set_description("Estimated distance")
    return metrics
