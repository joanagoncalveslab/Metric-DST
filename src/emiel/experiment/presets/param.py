from tensorflow.keras.optimizers.legacy import Adam
from models.autoencoder import default_classifier


def auto_param_gen(trial, mmd_weight=None):
    """Generates parameters
    :param trial: Optuna trial for hyperparameter optimization, or Optuna study.best_trial to reuse best params
    :param mmd_weight: fixed mmd_weight to use. if None, included in HPO"""

    if mmd_weight is None:
        mmd_weight = trial.suggest_float('mmd_weight', 0.0, 10.0)
    return dict(input_dim=5, encoder_dim=3,
                aux_classifier_weight=trial.suggest_float('aux_class_weight', 0.0, 10.0),
                mmd_weight=mmd_weight)


def dann_param_gen(trial, lambda_=None):
    """Generates parameters
    :param trial: Required if lambda_ is None. Optuna trial for hyperparameter optimization,
    or Optuna study.best_trial to reuse best params.
    :param lambda_: fixed mmd_weight to use. if None, included in HPO."""

    if lambda_ is None:
        lambda_ = trial.suggest_float('lambda_', 0.0, 10.0)
    return dict(loss="bce",
                optimizer=Adam(0.001, beta_1=0.5),
                lambda_=lambda_,
                metrics=["acc"],
                task=default_classifier(),
                random_state=0)
