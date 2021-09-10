import numpy as np
import matplotlib.pyplot as plt

from ftc.models.fixedWing import F16

from fym.core import BaseEnv, BaseSystem
import fym.logging


class Env(BaseEnv):
    def __init__(self, long, euler, omega, pos, POW, u):
        super().__init__(dt=0.01, max_t=10)
        self.x0 = np.vstack((long, euler, omega, pos, POW))
        self.plant = F16(long, euler, omega, pos, POW)
        self.u = u

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        self.plant.set_dot(t, self.u)
        return dict(t=t, x=self.plant.observe_dict())


def run(long, euler, omega, pos, POW, u):
    env = Env(long, euler, omega, pos, POW, u)
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
    plt.plot(data["t"], data["x"]["long"][:, 0, 0], label="VT [m]")
    plt.plot(data["t"], np.rad2deg(data["x"]["long"][:, 1, 0]), label="alp [deg]")
    plt.plot(data["t"], np.rad2deg(data["x"]["long"][:, 2, 0]), label="bet [deg]")
    plt.legend()

    plt.subplot(512, sharex=ax)
    plt.plot(data["t"], np.rad2deg(data["x"]["euler"][:, 0, 0]), label="phi [deg]")
    plt.plot(data["t"], np.rad2deg(data["x"]["euler"][:, 1, 0]), label="theta [deg]")
    plt.plot(data["t"], np.rad2deg(data["x"]["euler"][:, 2, 0]), label="psi [deg]")
    plt.legend()

    plt.subplot(513, sharex=ax)
    plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, 0, 0]), label="p [deg/s]")
    plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, 1, 0]), label="q [deg/s]")
    plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, 2, 0]), label="r [deg/s]")
    plt.legend()

    plt.subplot(514, sharex=ax)
    plt.plot(data["t"], data["x"]["pos"][:, 0, 0], label="pn [m]")
    plt.plot(data["t"], data["x"]["pos"][:, 1, 0], label="pe [m]")
    plt.plot(data["t"], data["x"]["pos"][:, 2, 0], label="pd [m]")
    plt.legend()

    plt.subplot(515, sharex=ax)
    plt.plot(data["t"], data["x"]["POW"][:, 0], label="POW")
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
    long = np.vstack((1.530096e+2, 5.0581196e-02, 4.32599079e-07))
    euler = np.vstack((0., 5.0581196e-02, 0.))
    omega = np.vstack((0., 0., 0.))
    pos = np.vstack((0., 0., 0.))
    POW = 1.00031551e+1
    u = np.vstack((0.154036882, -1.21241555e-2, -1.40709399e-6, 1.09618012e-5))
    run(long, euler, omega, pos, POW, u)
    exp1_plot()
