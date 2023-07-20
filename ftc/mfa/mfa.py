from ftc.mfa.polytope import Hypercube, Polytope


class MFA:
    def __init__(self, umin, umax, predictor, distribute, is_success):
        self.ubox = Hypercube(umin, umax)
        self.predictor = predictor
        self.distribute = distribute
        self.is_success = is_success

    def get_polynus(self, t, ubox):
        state, nu = self.predictor.get(t)
        vertices = ubox.vertices.map(self.create_distribute(t, state))
        return Polytope(vertices), nu[2:].ravel()

    def predict(self, tspan, lmbd, scaling_factor=1.0):
        ubox = self.ubox.map(lambda u_min, u_max: (lmbd * u_min, lmbd * u_max)).map(
            lambda u_min, u_max: shrink(u_min, u_max, scaling_factor)
        )

        return self.is_success(map(lambda t: self.get_polynus(t, ubox), tspan))

    def create_distribute(self, t, state):
        def distribute(u):
            nu = self.distribute(t, state, u)
            return nu

        return distribute


def shrink(u_min, u_max, scaling_factor=1.0):
    mean = (u_min + u_max) / 2
    width = (u_max - u_min) / 2
    u_min = mean - scaling_factor * width
    u_max = mean + scaling_factor * width
    return u_min, u_max
