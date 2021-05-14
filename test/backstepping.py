import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2quat
import fym.logging

from ftc.models.multicopter import Multicopter
from ftc.agents.backstepping import BacksteppingController
from ftc.agents.grouping import Grouping
from ftc.faults.actuator import LoE, LiP, Float


class FDI(BaseSystem):
    def __init__(self, no_act, tau):
        super().__init__(np.eye(no_act))
        self.tau = tau

    def get_true(self, u, uc):
        w = np.hstack([
            ui / uci
            if (ui != 0 and uci != 0) else 1
            if (ui == 0 and uci == 0) else 0
            for ui, uci in zip(u, uc)])
        return np.diag(w)

    def get_index(self, W):
        fault_index = np.where(np.diag(W) != 1)[0]
        return fault_index

    def set_dot(self, W):
        What = self.state
        self.dot = - 1 / self.tau * (What - W)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        pos0 = np.ones((3, 1))
        self.plant = Multicopter(pos=pos0)
        self.controller = BacksteppingController(
            self.plant.pos.state,
            self.plant.m,
            self.plant.g)

        # Define faults
        self.sensor_faults = []
        self.actuator_faults = [
            LoE(time=3, index=0, level=0.5),
            LoE(time=5, index=1, level=0.2),
            LoE(time=7, index=2, level=0.5),
            # Float(time=10, index=0),
        ]

        # Define FDI
        self.fdi = FDI(no_act=self.plant.mixer.B.shape[1], tau=0.1)
        self.grouping = Grouping(self.plant.mixer.B)

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.plant.state
        What = self.fdi.state

        u, W, _, Td_dot, xc, *_ = self._get_derivs(t, x, What)

        self.plant.set_dot(t, u)
        self.fdi.set_dot(W)
        self.controller.set_dot(Td_dot, xc)

    def control_allocation(self, f, What):
        fault_index = self.fdi.get_index(What)
        if len(fault_index) == 0:
            return np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(f)
        else:
            G = self.grouping.get(fault_index[0])
            return np.linalg.pinv(G.dot(What)).dot(f)

    def _get_derivs(self, t, x, What):
        # Set sensor faults
        for sen_fault in self.sensor_faults:
            x = sen_fault(t, x)

        FM, Td_dot = self.controller.command(
            *self.plant.observe_list(), *self.controller.observe_list(),
            self.plant.m, self.plant.J, np.vstack((0, 0, self.plant.g)),
        )
        pos_c = np.zeros((3, 1))  # TODO: position commander
        u = u_command = self.control_allocation(FM, What)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            u = act_fault(t, u)

        W = self.fdi.get_true(u, u_command)

        return u, W, u_command, Td_dot, pos_c

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x = states["plant"]
        What = states["fdi"]
        x_controller = states["controller"]
        u, W, uc, Td_dot, pos_c, *_ = self._get_derivs(t, x, What)
        return dict(
            t=t,
            x=x,
            What=What,
            u=u,
            uc=uc,
            W=W,
            x_controller=x_controller,
            pos_c=pos_c
        )


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


def exp2():
    run()


def exp2_plot():
    data = fym.logging.load("data.h5")

    plt.figure()
    plt.plot(data["t"], data["x"]["pos"][:, :, 0], "r--", label="pos")  # position
    plt.plot(data["t"], data["pos_c"][:, :, 0], "k--", label="position command")  # position command
    # plt.plot(data["t"], data["x_controller"]["xd"][:, :, 0], "b--", label="desired pos")  # desired position

    plt.legend()
    plt.show()


if __name__ == "__main__":
    exp2()
    exp2_plot()
