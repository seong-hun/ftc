from functools import reduce

from ftc.mfa.polytope import Hypercube, Polytope


class Assessment:
    def __init__(self, polynus, is_success):
        self.polynus = polynus
        self.is_success = is_success

    @property
    def result(self):
        return self.is_success(self.polynus)
    
class MFA:
    def __init__(self, umin, umax, predictor, distribute, is_success):
        self.ubox = Hypercube(umin, umax)
        self.predictor = predictor
        self.distribute = distribute
        self.is_success = is_success

    def get_polynu(self, t, ubox):
        state, nu = self.predictor.get(t)
        vertices = ubox.vertices.map(self.create_distribute(t, state))
        return Polytope(vertices), nu[2:].ravel()

    def get_ubox(self, fns):
        return reduce(lambda ubox, fn: ubox.map(fn), (self.ubox, *fns))

    def assess(self, tspan, fns):
        return Assessment(map(lambda t: self.get_polynu(t, self.get_ubox(fns)), tspan), self.is_success)

    def create_distribute(self, t, state):
        def distribute(u):
            nu = self.distribute(t, state, u)
            return nu

        return distribute

    def visualize(self, assessment:Assessment):
        assessment.polynus

