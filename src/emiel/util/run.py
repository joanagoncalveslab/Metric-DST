from evaluate.evaluate import evaluate_deep
from evaluate.evaluate import analyze_data
from storage.storage import Store
from typing import Tuple


def run_generate(builder, name: str = None, store_path: str = None) -> Tuple[Store, dict]:
    """Generate a dataset, and store it, along with basic analysis and configuration.
    :param builder: Dataset builder, from datagen.covshift or datagen.conceptshift
    :param name: name of the folder with the results of the run, timestamp if None
    :param store_path: path to the directory containing runs, <cwd>/results if None
    :returns created store, and dictionary with basic stats about the created set
    """
    data = builder.generate()
    data_stats = analyze_data(data)

    store = Store.new(name, store_path, overwrite=True)
    store.save_data(*data)
    store.save_config(builder)
    store.save_stats(data_stats)
    return store, data_stats


def model_param_combiner(model, params):
    """"Functions similar to `lambda: model(**params), but a method is required for lazy evaluation complications"""
    def _func():
        return model(**params)
    return _func


def run_eval(name: str, model, model_params: dict, fit_params: dict,
             train_split: float, multi_param=False, identifier: str = None, store_path: str = None) -> dict:
    """Load a stored dataset and evaluate a model's performance on it, then store results.
    :param name: name of the folder with the results of the run
    :param model: class of the adaptation model, such as adapt DANN, ADDA, MDD etc.
    :param model_params: parameters for the model class's __init__.
    If multi_param is True, instead a dictionary with a nested dictionaries of values for each configuration.
    Keys formatted like 's-only' and 's->t'.
    :param fit_params: parameters for the model class's fit
    :param train_split: proportion to use for training data, use rest for test.
    :param multi_param: use different parameters for different source/target configurations. See 'model_params'.
    :param identifier: appended to the results file name.
    Use to prevent overwriting when evaluating multiple models on the same dataset.
    :param store_path: path to the directory containing runs, <cwd>/results if None
    :returns resulting dictionary from evaluation
    """
    store = Store(name, store_path)
    data = store.load_data()

    if multi_param:
        builder = dict()
        for key in model_params:
            builder[key] = model_param_combiner(model, model_params[key])
    else:
        builder = lambda: model(**model_params)

    deep_metrics = evaluate_deep(data, builder, fit_params, train_split)
    store.save_eval(deep_metrics, model.__name__, model_params, fit_params, identifier)
    return deep_metrics
