import os

import numpy as np
import optuna
from adapt.feature_based import DANN

from experiment.presets.bias import bias_names
from experiment.presets.param import auto_param_gen, dann_param_gen
from models.autoencoder import Autoencoder
from util.batch import batch_eval_single

PREFIX = "v6"
STORAGE = "sqlite:///../db.sqlite3"
FIT_PARAMS = dict(epochs=20,
                  batch_size=16,
                  verbose=0)


def objective(path, model_cls, param_opt, source_domain, target_domain):
    """Generate an objective function for the current study.
    Requires a param_opt function that returns a model_params dict, using trial.suggest..."""
    def _objective(trial):
        model_params = param_opt(trial)
        acc = batch_eval_single(path, model_cls, model_params, FIT_PARAMS, source_domain, target_domain)
        return np.mean(acc)

    return _objective


def opt(model, param_generator, bias: str, source: str, target: str, n_trials: int = 15):
    """Run hyperparameter optimization for a given configuration.
       Stores result in Optuna database, with PREFIX and identifier in study name."""
    store_path = os.path.join(os.getcwd(), '../results', PREFIX, bias)
    adapt_name = f'{source}-only' if source == target else f'{source}->{target}'
    study_name = f"{PREFIX}-{bias}-{model.__name__}-{adapt_name}"

    # check if it exists already
    summaries = optuna.study.get_all_study_summaries(storage=STORAGE)
    if study_name in [s.study_name for s in summaries]:
        print(f"Skipping {study_name} because it already exists.")
        return

    obj = objective(store_path, model, param_generator, source, target)
    study = optuna.create_study(
        storage=STORAGE,
        study_name=study_name,
        direction='maximize',
        load_if_exists=False)
    study.optimize(obj, n_trials=n_trials)


if __name__ == "__main__":

    for bias_name in bias_names:

        # adaptation
        opt(DANN, dann_param_gen, bias_name, 's', 't')
        opt(DANN, dann_param_gen, bias_name, 's', 'g')
        opt(Autoencoder, auto_param_gen, bias_name, 's', 't')
        opt(Autoencoder, auto_param_gen, bias_name, 's', 'g')

        # supervised, zero adaptation parameter, only Autoencoder has params left to optimize
        opt(Autoencoder, lambda trial: auto_param_gen(trial, mmd_weight=0), bias_name, 't', 't')
