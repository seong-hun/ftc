import numpy as np
import matplotlib.pyplot as plt

from ftc.models.fixedWing import F16, F16lon

from fym.core import BaseEnv, BaseSystem
import fym.logging


class Env(BaseEnv):
    def __init__(self, lon, u):
        super().__init__(dt=0.01, max_t=10)
        self.x0 = lon
        self.plant = F16lon(lon)
        self.u = u

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        self.plant.set_dot(t, self.u)
        return dict(t=t, x=self.plant.state)


def run(lon, u):
    env = Env(lon, u)
    env.logger = fym.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def exp1_plot():
    data = fym.logging.load("data.h5")

    # state variables
    plt.figure()

    ax = plt.subplot(511)
    plt.plot(data["t"], data["x"][:, 0, 0], label="VT [m/s]")
    plt.legend()
    plt.subplot(512, sharex=ax)
    plt.plot(data["t"], np.rad2deg(data["x"][:, 1, 0]), label="gamma [deg]")
    plt.legend()
    plt.subplot(513, sharex=ax)
    plt.plot(data["t"], data["x"][:, 2, 0], label="h [m]")
    plt.legend()
    plt.subplot(514, sharex=ax)
    plt.plot(data["t"], np.rad2deg(data["x"][:, 3, 0]), label="alp [deg]")
    plt.legend()
    plt.subplot(515, sharex=ax)
    plt.plot(data["t"], np.rad2deg(data["x"][:, 4, 0]), label="q [deg/s]")
    plt.legend()

    plt.tight_layout()

    # input
    # plt.figure()

    # plt.plot(data["t"], data["u"][:, 0, 0], label="delt")
    # plt.plot(data["t"], data["u"][:, 1, 0], label="dele")
    # plt.legend()

    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    lon = np.vstack((1.530096e+2, 0., 0., 5.05748236e-2, 0.))
    u = np.vstack((0.18500661, -0.0121247))
    run(lon, u)
    exp1_plot()
