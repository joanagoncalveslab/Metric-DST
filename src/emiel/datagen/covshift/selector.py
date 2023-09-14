"""This module contains the Selector class that is used to simulate sample selection bias.
Intended use is passing it to the builder, instead of calling the methods directly."""

import numpy as np
from scipy.stats import norm
from typing import Tuple


def sample_biased(x_: np.ndarray, y_: np.ndarray, n: int,
                  mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use a (zero-covariance) gaussian distribution to sample from candidate data with replacement."""
    distr = norm(mean, std)
    probs = distr.pdf(x_).prod(1)
    probs /= probs.sum()
    sample = np.random.choice(np.arange(len(x_)), n, p=probs)
    return x_[sample], y_[sample]


class FeatureSelector:
    """Selector class to simulate sample selection bias on features, creating a shifted source and target set.
Intended use is passing it to the builder after creation, instead of calling the methods directly."""

    def __init__(self, n_global: int, n_source: int, n_target: int,
                 source_scale: float, target_scale: float, bias_dist: float):
        """
        Create a selector for simulating sample biased selection of source and target, on a given dataset.
        If the input data points are unique, it returns disjoint global, source and target.
        The sample bias is determined by a multivariate gaussian distribution on the features without covariance.
        The two gaussian's are located bias_dist/2 away from the data mean, on opposing sides.
        The type of selection bias specifically is MAR[1], which subsumes covariate shift.

        [1] Moreno-Torres et al. A unifying view on dataset shift in classification, 2012

        :param n_global: number of unbiased samples
        :param n_source: number of biased source samples
        :param n_target: number of biased target samples
        :param source_scale: scale of gaussian defining source bias (as factor of feature std)
        :param target_scale: scale of gaussian defining source bias (as factor of feature std)
        :param bias_dist: distance between the means of the source and target gaussian (as factor of feature std)
        """
        self.n_global = n_global
        self.n_source = n_source
        self.n_target = n_target
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.bias_dist = bias_dist

    def select(self, x: np.ndarray, y: np.ndarray) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform the selection on a dataset, as configured in initialization.
        :param x: Features as floats in shape (N,D)
        :param y: Labels in shape (N,)
        :return: tuple (xg, yg, xs, ys, xt, yt) where s=source, g=global, t=target.
        """

        # split into source candidates, global, target candidates
        xg, yg, xs_, ys_, xt_, yt_ = self._make_split(x, y)

        stds = x.std(0)
        center = x.mean(0)
        bias_dir = 2*np.random.rand(x.shape[1]) - 0.5
        bias = self.bias_dist * stds * bias_dir / np.linalg.norm(bias_dir)

        source_center = center + 0.5 * bias
        target_center = center - 0.5 * bias
        source_std = stds * self.source_scale
        target_std = stds * self.target_scale

        xs, ys = sample_biased(xs_, ys_, self.n_source, source_center, source_std)
        xt, yt = sample_biased(xt_, yt_, self.n_target, target_center, target_std)
        return xg, yg, xs, ys, xt, yt

    def _make_split(self, x: np.ndarray, y: np.ndarray) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partition the given dataset into unbiased and disjoint global, source and target.
        Permutes the data randomly first. The global partition has size n_global,
        the remainder is divided based on the ratio of n_source, n_target.
        :param x: Features as floats in shape (N,D)
        :param y: Labels in shape (N,)
        :return: tuple (xg, yg, xs, ys, xt, yt), where s=source, g=global, t=target."""
        n_original = len(x)
        assert len(x) == len(y)
        assert n_original > self.n_global
        p = np.random.permutation(n_original)
        x, y = x[p], y[p]

        # select global without bias
        xg, yg = x[:self.n_global], y[:self.n_global]

        # divide the remainder over source and target proportionally, initially without bias
        st_ratio = self.n_source / (self.n_source + self.n_target)
        m_source = int(st_ratio * (n_original - self.n_global))

        xs, ys = x[self.n_global:self.n_global + m_source], y[self.n_global:self.n_global + m_source]
        xt, yt = x[self.n_global + m_source:], y[self.n_global + m_source:]

        return xg, yg, xs, ys, xt, yt

    def to_json(self):
        return {
            'class_name': self.__class__.__name__,
            'n_global': self.n_global,
            'n_source': self.n_source,
            'n_target': self.n_target,
            'source_scale': self.source_scale,
            'target_scale': self.target_scale,
            'bias_dist': self.bias_dist
        }