import numpy as np
from typing import Tuple

from scipy.stats import special_ortho_group


class Shifter:
    def __init__(self, n_domains: int = 2, rot: float = 0, trans: float = 0, scale: float = 0):
        """
        Initialize shifter object, to be used with ConceptShiftDataBuilder.
        :param n_domains: number of partitions of the data
        :param rot: range of random rotation in [0,1]
        :param trans: scale factor of random translation, which is random uniform in [-stdev, stdev]
        :param scale: range of random scale in [0,inf). Scaling will be random uniform in [1,1+scale]
        """
        self.n_domains = n_domains
        self.rot = rot
        self.trans = trans
        self.scale = scale

    def shift(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Partition the given x and y uniformly into several domains,
        Each domain receives a randomized transformation, according to initialisation parameters.
        The data is permuted randomly, but the domain labels are included for distinguishing them
        :param x: features (N,M)
        :param y: labels (N,)
        :returns: tuple (x,y,domain), where domain is an integer label in [0,self.n_domains)
        """

        std = np.std(x)
        dim = x.shape[1]

        # randomly permute, so domains arent dependent on initial data order
        n = x.shape[0]
        p = np.random.permutation(n)
        splits = np.array_split(p, self.n_domains)

        res_x = np.empty_like(x)
        res_y = np.empty_like(y)
        res_domain = np.empty_like(y)
        progress = 0
        for domain, split in enumerate(splits):
            x_ = x[split]
            y_ = y[split]

            # scale
            scaling_vector = np.random.random(dim) * self.scale + 1
            x_ *= scaling_vector

            # random rotation (a bit hacky with interpolation)
            rot_init = special_ortho_group.rvs(dim)
            rot_init = (1-self.rot) * np.identity(dim) + self.rot * rot_init
            rot, _ = np.linalg.qr(rot_init)  # make it orthogonal again
            x_ = x_.dot(rot)

            # translate
            trans = 2*(np.random.random(dim) - 0.5) * std * self.trans
            x += trans

            # insert result into pre-allocated array
            res_x[progress:progress + len(split)] = x_
            res_y[progress:progress + len(split)] = y_
            res_domain[progress:progress + len(split)] = domain
            progress += len(split)

        # randomly permute again, so domains aren't grouped
        p = np.random.permutation(n)
        res_x = res_x[p]
        res_y = res_y[p]
        res_domain = res_domain[p]
        return res_x, res_y, res_domain

    def to_json(self):
        return {
            'class_name': self.__class__.__name__,
            'n_domains': self.n_domains,
            'rot': self.rot,
            'trans': self.trans,
            'scale': self.scale
        }
