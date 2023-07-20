import numpy as np

from ftc.controllers.Flat.flat import FlatController
from ftc.mfa.polytope import Hypercube, Polytope


class MFA:
    def __init__(self, env, distribute):
        pwm_min, pwm_max = env.plant.control_limits["pwm"]
        self.ubox = Hypercube(pwm_min * np.ones(6), pwm_max * np.ones(6))
        self.controller = FlatController(env.plant.m, env.plant.g, env.plant.J)

        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        c, self.c_th = 0.0338, 128  # tq / th, th / rcmds
        self.B_r2f = np.array(
            (
                [-1, -1, -1, -1, -1, -1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-c, c, -c, c, c, -c],
            )
        )

    def predict(self, tspan, lmbd, scaling_factor=1.0):
        ubox = self.ubox.map(lambda u_min, u_max: (lmbd * u_min, lmbd * u_max)).map(
            lambda u_min, u_max: shrink(u_min, u_max, scaling_factor)
        )

        def is_success(t):
            state, nu = self.controller.get(t)
            distribute = self.create_distribute(t, state)
            vertices = (ubox.vertices).map(distribute)
            polytope = Polytope(vertices)

            return polytope.contains(nu)

        return next(filter(is_success, tspan), True)

    def distribute(self, t, state, pwms_rotor):
        nu = self.B_r2f @ (pwms_rotor - 1000) / 1000 * self.c_th
        return nu

    def create_distribute(self, t, state):
        def distribute(self, u):
            nu = self.distribute(t, state, u)
            return nu

        return distribute


def shrink(u_min, u_max, scaling_factor=1.0):
    mean = (u_min + u_max) / 2
    width = (u_max - u_min) / 2
    u_min = mean - scaling_factor * width
    u_max = mean + scaling_factor * width
    return u_min, u_max
