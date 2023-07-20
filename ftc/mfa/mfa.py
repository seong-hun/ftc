import numpy as np

import ftc
from ftc.mfa.polytope import Hypercube, Polytope


class MFA:
    def __init__(self, env, distribute):
        pwm_min, pwm_max = env.plant.control_limits["pwm"]
        self.ubox = Hypercube(pwm_min * np.ones(6), pwm_max * np.ones(6))
        self.controller = ftc.make("Flat", env)
        self.distribute = distribute

    def predict(self, tspan, lmbd, scaling_factor=1.0):
        ubox = self.ubox.map(lambda u_min, u_max: (lmbd * u_min, lmbd * u_max)).map(
            lambda u_min, u_max: shrink(u_min, u_max, scaling_factor)
        )

        def is_success(t):
            state, nu = self.controller.get(t)
            distribute = self.create_distribute(t, state)
            vertices = ubox.vertices.map(distribute)
            polytope = Polytope(vertices)

            return polytope.contains(nu[2:].ravel())

        return next(filter(is_success, tspan), True)

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
