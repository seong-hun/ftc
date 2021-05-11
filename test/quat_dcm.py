import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2quat
import fym.logging

from ftc.models.multicopter import Multicopter
from fym.utils.rot import dcm2quat, quat2dcm, angle2quat, quat2angle


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        self.plant = Multicopter(rtype="quad")

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.plant.state

        f = self.get_forces(x)
        u = self.plant.mixer.Binv.dot(f)

        self.plant.set_dot(t, u)

    def get_forces(self, x):
        return np.vstack((self.plant.m * self.plant.g, 0., 0., 0.))

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        dcm1 = states["plant"]["dcm"]
        quat2 = states["plant"]["quat"]
        quat1 = dcm2quat(dcm1)
        dcm2 = quat2dcm(quat2)
        ang1 = np.flip(quat2angle(quat1))
        ang2 = np.flip(quat2angle(quat2))

        return dict(t=t, quat1=quat1, quat2=quat2, ang1=ang1, ang2=ang2,
                    dcm1=dcm1, dcm2=dcm2)


def run():
    env = Env()
    env.logger = fym.logging.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def exp():
    run()


def exp_plot():
    data = fym.logging.load("data.h5")

    plt.figure()

    ax = plt.subplot(311)
    plt.plot(data["t"], data["ang1"][:, 0], "r--", label="dcm")
    plt.plot(data["t"], data["ang2"][:, 0], "k-", label="quat")
    plt.ylabel("phi")
    plt.legend()

    plt.subplot(312, sharex=ax)
    plt.plot(data["t"], data["ang1"][:, 1], "r--", label="dcm")
    plt.plot(data["t"], data["ang2"][:, 1], "k-", label="quat")
    plt.ylabel("theta")
    plt.legend()

    plt.subplot(313, sharex=ax)
    plt.plot(data["t"], data["ang1"][:, 2], "r--", label="dcm")
    plt.plot(data["t"], data["ang2"][:, 2], "k-", label="quat")
    plt.ylabel("psi")
    plt.legend()

    plt.figure()

    ax = plt.subplot(411)
    plt.plot(data["t"], data["quat1"][:, 0], "r--", label="dcm")
    plt.plot(data["t"], data["quat2"][:, 0], "k-", label="quat")
    plt.ylabel("q0")
    plt.legend()

    plt.subplot(412, sharex=ax)
    plt.plot(data["t"], data["quat1"][:, 1], "r--", label="dcm")
    plt.plot(data["t"], data["quat2"][:, 1], "k-", label="quat")
    plt.ylabel("q1")
    plt.legend()

    plt.subplot(413, sharex=ax)
    plt.plot(data["t"], data["quat1"][:, 2], "r--", label="dcm")
    plt.plot(data["t"], data["quat2"][:, 2], "k-", label="quat")
    plt.ylabel("q2")
    plt.legend()

    plt.subplot(414, sharex=ax)
    plt.plot(data["t"], data["quat1"][:, 3], "r--", label="dcm")
    plt.plot(data["t"], data["quat2"][:, 3], "k-", label="quat")
    plt.ylabel("q3")
    plt.legend()

    plt.figure()

    ax = plt.subplot(331)
    plt.plot(data["t"], data["dcm1"][:, 0, 0], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 0, 0], "k-", label="quat")
    plt.ylabel("dcm(1,1)")
    plt.legend()

    plt.subplot(332, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 0, 1], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 0, 1], "k-", label="quat")
    plt.ylabel("dcm(1,2)")
    plt.legend()

    plt.subplot(333, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 0, 2], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 0, 2], "k-", label="quat")
    plt.ylabel("dcm(1,3)")
    plt.legend()

    plt.subplot(334, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 1, 0], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 1, 0], "k-", label="quat")
    plt.ylabel("dcm(2,1)")
    plt.legend()

    plt.subplot(335, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 1, 1], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 1, 1], "k-", label="quat")
    plt.ylabel("dcm(2,2)")
    plt.legend()

    plt.subplot(336, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 1, 2], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 1, 2], "k-", label="quat")
    plt.ylabel("dcm(2,3)")
    plt.legend()

    plt.subplot(337, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 2, 0], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 2, 0], "k-", label="quat")
    plt.ylabel("dcm(3,1)")
    plt.legend()

    plt.subplot(338, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 2, 1], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 2, 1], "k-", label="quat")
    plt.ylabel("dcm(3,2)")
    plt.legend()

    plt.subplot(339, sharex=ax)
    plt.plot(data["t"], data["dcm1"][:, 2, 2], "r--", label="dcm")
    plt.plot(data["t"], data["dcm2"][:, 2, 2], "k-", label="quat")
    plt.ylabel("dcm(3,3)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    exp()
    exp_plot()
