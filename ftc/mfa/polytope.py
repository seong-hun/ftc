from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


class Vertices:
    def __init__(self, points):
        self.points = np.asarray(points)

    def map(self, func):
        return Vertices([func(p) for p in self.points])


class Hypercube:
    def __init__(self, u_min, u_max):
        self.u_min = np.asarray(u_min)
        self.u_max = np.asarray(u_max)

    @property
    def vertices(self):
        return Vertices(list(product(*zip(self.u_min, self.u_max))))

    def map(self, func):
        return Hypercube(*func(self.u_min, self.u_max))


class Polytope:
    """
        PolytopeDeterminer

    A mission-success determiner using polytope.
    """

    def __init__(self, vertices: Vertices):
        """
        u_min: (m,) array; minimum input (element-wise)
        u_max: (m,) array; maximum input (element-wise)
        """
        self.vertices = vertices

    @property
    def delaunay(self):
        return Delaunay(self.vertices.points)

    @property
    def convexhull(self):
        return ConvexHull(self.vertices.points)

    def contains(self, nu):
        return np.logical_not(self.delaunay.find_simplex(nu) < 0)
