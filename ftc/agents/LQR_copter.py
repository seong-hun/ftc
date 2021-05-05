import numpy as np

from fym.utils.rot import quat2angle
from fym.agents import LQR


class LQR_copter:
    def __init__(self, env):
        self.ic = env.plant.ic
        Jinv = env.plant.Jinv
        m, g = env.plant.m, env.plant.g
        n_rotors = env.plant.mixer.n_rotor
        trim_rotors = np.vstack([m * g / n_rotors] * n_rotors)
        self.trim_forces = env.plant.mixer.inverse(trim_rotors)

        A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [-1/m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, Jinv[0, 0], 0, 0],
                      [0, 0, Jinv[1, 1], 0],
                      [0, 0, 0, Jinv[2, 2]]])
        Q = np.diag([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        R = np.diag([1, 1, 1, 1])

        self.K, *_ = LQR.clqr(A, B, Q, R)

    def observation(self, obs):
        return np.vstack((obs[0:6], np.vstack(quat2angle(obs[6:10])[::-1]), obs[10::]))

    def get_action(self, obs):
        x = self.observation(obs)
        forces = -self.K.dot(x - self.ic) + self.trim_forces
        return forces
