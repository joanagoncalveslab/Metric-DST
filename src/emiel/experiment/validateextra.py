import os

import optuna
from adapt.feature_based import DANN

from experiment.presets.bias import bias_names
from experiment.presets.param import auto_param_gen, dann_param_gen
from models.autoencoder import Autoencoder
from util.batch import batch_eval

PREFIX = "v6"

FIT_PARAMS = dict(epochs=30,
                  batch_size=16,
                  verbose=0)


def load_best_trial(bias, model, adapt_name):
    study_name = f"{PREFIX}-{bias}-{model.__name__}-{adapt_name}"
    return optuna.load_study(study_name=study_name, storage="sqlite:///../db.sqlite3").best_trial


if __name__ == "__main__":
    """Run the evaluation framework for covariate shift, 
    fixing the parameters to those that worked for concept strong."""

    for bias_name in bias_names:
        if 'cov' not in bias_name:
            continue
        store_path = os.path.join(os.getcwd(), '../results', PREFIX, f"{bias_name}_val")

        dann_params = {
            's-only': dann_param_gen(None, lambda_=0.0),
            's->t': dann_param_gen(load_best_trial("concept_strong", DANN, 's->t')),
            's->g': dann_param_gen(load_best_trial("concept_strong", DANN, 's->g')),
            't-only': dann_param_gen(None, lambda_=0.0)
        }

        auto_params = {
            's-only': auto_param_gen(load_best_trial("concept_strong", Autoencoder, 't-only'), mmd_weight=0.0),
            's->t': auto_param_gen(load_best_trial("concept_strong", Autoencoder, 's->t')),
            's->g': auto_param_gen(load_best_trial("concept_strong", Autoencoder, 's->g')),
            't-only': auto_param_gen(load_best_trial("concept_strong", Autoencoder, 't-only'), mmd_weight=0.0),
        }

        print(f"Evaluating with {bias_name}")
        batch_eval(store_path, DANN, dann_params, FIT_PARAMS,
                   train_split=.7, multi_param=True, identifier=DANN.__name__+'$')
        batch_eval(store_path, Autoencoder, auto_params, FIT_PARAMS,
                   train_split=.7, multi_param=True, identifier=Autoencoder.__name__+'$')
