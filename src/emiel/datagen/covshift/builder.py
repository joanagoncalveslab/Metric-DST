"""Contains builder class for generating datasets with some configuration"""
import numpy as np

from .selector import FeatureSelector
from sklearn.datasets import make_classification
from typing import Tuple


class CovShiftBuilder:
    def __init__(self, init_classify: dict, selector: FeatureSelector):
        """
        Create a dataset builder class with some configuration, that can be reused to create similar datasets.
        To generate from the same initial distribution, but using a different selection each time,
        pass an integer for "random_state" in init_classify.

        :param init_classify: parameters for sklearn.dataset.make_classification for the initial dataset
        :param selector: selector object that splits the initial set into source, global and target
        """
        self.init_classify = init_classify
        self.selector = selector

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a new source, global and test dataset using the configuration from initialization.

        :return:  tuple (xg, yg, xs, ys, xt, yt), where s=source, g=global, t=target.
        """
        x, y = make_classification(**self.init_classify)
        xg, yg, xs, ys, xt, yt = self.selector.select(x, y)
        return xg, yg, xs, ys, xt, yt

    def to_json(self):
        selector_json = self.selector.to_json()

        return {
            'class_name': self.__class__.__name__,
            'init_classify': self.init_classify,
            'selector': selector_json
        }