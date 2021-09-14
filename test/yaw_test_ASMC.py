import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.agents.CA import CA, ConstrainedCA
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager
from ftc.agents.AdaptiveSMC import AdaptiveSMController
from ftc.plotting import exp_plot

cfg = ftc.config.load()


class ActuatorDynamcs(BaseSystem):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def set_dot(self, rotors, rotors_cmd):
        self.dot = - 1 / self.tau * (rotors - rotors_cmd)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(solver="odeint", max_t=100, dt=10, ode_step_len=100)
        init = cfg.models.multicopter.init
        self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega)
        n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            LoE(time=3, index=0, level=0.),  # scenario a
            # LoE(time=6, index=2, level=0.),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        # self.CCA = ConstrainedCA(self.plant.mixer.B)
        ic = np.vstack((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        ref0 = np.vstack((-1, 1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        self.controller = AdaptiveSMController(self.plant.J,
                                               self.plant.m,
                                               self.plant.g,
                                               self.plant.d,
                                               ic,
                                               ref0)
        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, forces, What, t):
        rotors = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)

        # if len(fault_index) == 0:
        #     rotors = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)
        # else:
        #     rotors = self.CCA.solve_lp(fault_index, forces,
        #                                self.plant.rotor_min,
        #                                self.plant.rotor_max)
        return rotors

    def get_ref(self, t):
        pos_des = np.vstack([-1, 1, -2])
        vel_des = np.vstack([0, 0, 0])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        ref = np.vstack([pos_des, vel_des, quat_des, omega_des])

        return ref

    def set_dot(self, t):
        mult_states = self.plant.state
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        ref = self.get_ref(t)
        p, gamma = self.controller.observe_list()

        # Set sensor faults
        for sen_fault in self.sensor_faults:
            mult_states = sen_fault(t, mult_states)

        # Controller
        forces, sliding = self.controller.get_FM(mult_states, ref, p, gamma)
        rotors_cmd = self.control_allocation(forces, What, t)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(mult_states, ref, sliding)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref)


def run():
    env = Env()
    env.logger = fym.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            env_info = {
                "detection_time": env.detection_time,
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)
            break

    env.close()


def exp1():
    run()


if __name__ == "__main__":
    exp1()
    loggerpath = "data.h5"
    exp_plot(loggerpath)
