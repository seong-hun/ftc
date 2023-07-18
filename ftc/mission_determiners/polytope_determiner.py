from itertools import product

import numpy as np
from scipy.spatial import Delaunay


class Vertices:
    def __init__(self, u_min, u_max):
        self.u_min = np.asarray(u_min)
        self.u_max = np.asarray(u_max)

    @property
    def points(self):
        return np.asarray(list(product(*zip(self.u_min, self.u_max))))

    def map(self, func):
        return Vertices(*func(self.u_min, self.u_max))

    def transform(self, transform):
        return [transform(p) for p in self.points]


class Polytope:
    """
        PolytopeDeterminer

    A mission-success determiner using polytope.
    """

    def __init__(self, points):
        """
        u_min: (m,) array; minimum input (element-wise)
        u_max: (m,) array; maximum input (element-wise)
        """
        self.points = points

    def contains(self, nu):
        return not Delaunay(self.points).find_simplex(nu) < 0
