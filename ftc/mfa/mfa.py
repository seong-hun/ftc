from functools import reduce

from ftc.mfa.polytope import Hypercube, Polytope


class MFA:
    def __init__(self, umin, umax, predictor, distribute, is_success):
        self.ubox = Hypercube(umin, umax)
        self.predictor = predictor
        self.distribute = distribute
        self.is_success = is_success

    def get_polynus(self, t, ubox: Hypercube):
        state, nu = self.predictor.get(t)
        vertices = ubox.vertices.map(self.create_distribute(t, state))
        return Polytope(vertices), nu[2:].ravel()

    def predict(self, tspan, fns):
        ubox = reduce(lambda ubox, fn: ubox.map(fn), (self.ubox, *fns))
        return self.is_success(map(lambda t: self.get_polynus(t, ubox), tspan))

    def create_distribute(self, t, state):
        def distribute(u):
            nu = self.distribute(t, state, u)
            return nu

        return distribute
