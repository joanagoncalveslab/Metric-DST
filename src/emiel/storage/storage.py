"""Module for handling folders with data, metadata, and evaluation results."""

import os
import shutil
import time
import numpy as np
import json
from types import FunctionType
from typing import Tuple

# names of files inside the store folder
DATA_FILE = 'data'
CFG_FILE = 'config'
STATS_FILE = 'stats'
EVAL_FILE = 'eval'


def serialize_soft(obj):
    """Try to serialize, otherwise return a placeholder using the classname"""
    if isinstance(obj, FunctionType):
        # Convert functions to their string representation
        return obj.__name__
    else:
        return f"<unserializable object of type '{obj.__class__.__name__}'>"


class Store:
    """Reference to a directory on disk, for storing a dataset for later retrieval, along with metadata."""

    def __init__(self, name: str, store_path: str = None):
        """
        Create an object referencing the storage folder on disk of a dataset, including metadata and other files.
        Does not create directory on disk. Use `Store.new` instead.

        :param name: Name of this dataset folder.
        :param store_path: Path to folder containing all stores. If None, uses '<root>/results'.
        """

        if store_path is None:
            store_path = os.path.join(os.getcwd(), 'results')

        self.name = name
        self.path_full = os.path.join(store_path, name)

        if not os.path.exists(self.path_full) or not os.path.isdir(self.path_full):
            raise FileNotFoundError(
                f"Store object references non-existent path. Expected directory in '{self.path_full}'")

    @classmethod
    def new(cls, name: str = None, store_path: str = None, overwrite: bool = False):
        """
        Create a storage folder on disk for a dataset, including metadata and other files.

        :param name: Name of this dataset folder. If None, uses timestamp, adding a postfix for duplicate timestamps
        :param store_path: Path to folder containing all stores. If None, uses '<cwd>/results'.
        :param overwrite: if False, raises exception if directory already exists, if True, deletes existing
        :returns: Store object, after creating directory.
        """

        if store_path is None:
            store_path = os.path.join(os.getcwd(), 'results')

        if name is None:
            name = time.strftime("%Y-%m-%d_%H-%M-%S")
            postfix = 1
            folder_name = name
            while os.path.exists(os.path.join(store_path, folder_name)):
                folder_name = f"{name}_{postfix}"
                postfix += 1
        else:
            folder_name = name

        path_full = os.path.join(store_path, folder_name)
        if os.path.exists(path_full):
            if overwrite:
                shutil.rmtree(path_full)
            else:
                raise FileExistsError(
                    f"Attempted to create new Store, but given path already exists in '{path_full}'.\n"
                    + "Use overwrite=True if intended.")
        os.makedirs(path_full)

        return Store(name, store_path)

    def save_data(self, xg, yg, xs, ys, xt, yt) -> None:
        """Store three sets of features and labels in this store"""
        np.savez(os.path.join(self.path_full, f'{DATA_FILE}.npz'),
                 xg=xg, yg=yg, xs=xs, ys=ys, xt=xt, yt=yt)

    def save_config(self, builder) -> None:
        """Save a JSON file describing the dataset builder configuration."""
        json_data = json.dumps(builder.to_json(), indent=4)
        path = os.path.join(self.path_full, f'{CFG_FILE}.json')
        with open(path, 'w') as f:
            f.write(json_data)

    def save_stats(self, stats: dict) -> None:
        """Store statistical measures of the dataset, that are independent of the machine learning model."""
        json_data = json.dumps(stats, sort_keys=True, indent=4)
        path = os.path.join(self.path_full, f'{STATS_FILE}.json')
        with open(path, 'w') as f:
            f.write(json_data)

    def save_eval(self, metrics: dict, model_type: str, model_params: dict, fit_params: dict, identifier: str = None):
        """Store metrics from evaluating a DA model on the dataset. Stores model configuration for future reference.
        :param metrics: results of the evaluation
        :param model_type: name of model, e.g. 'DANN' or 'MDD'
        :param model_params: parameters of the model, does not include the model type itself
        :param fit_params: parameters of fitting, like epochs, batch size etc.
        :param identifier: appended to filename, change the identifier to prevent
         overwriting if intending to save multiple configuration's results. leave None to ignore.
        """
        data = dict(model_type=model_type,
                    metrics=metrics,
                    model_params=model_params,
                    fit_params=fit_params)
        json_data = json.dumps(data, sort_keys=True, indent=4, default=serialize_soft)
        filename = f'{EVAL_FILE}_{identifier}.json' if identifier else f'{EVAL_FILE}.json'
        path = os.path.join(self.path_full, filename)
        with open(path, 'w') as f:
            f.write(json_data)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a dataset from this store, as the tuple (xg, yg, xs, ys, xt, yt),
        where g=global, s=source, t=target, x=features, y=label."""
        path = os.path.join(self.path_full, f'{DATA_FILE}.npz')
        loaded = np.load(path)
        return (loaded['xg'], loaded['yg'],
                loaded['xs'], loaded['ys'],
                loaded['xt'], loaded['yt'])
