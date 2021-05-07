import numpy as np
import matplotlib.pyplot as plt

from fym.utils.rot import quat2angle, angle2quat
from fym.agents import LQR

from ftc.models.multicopter import Multicopter
import fym.logging
from fym.core import BaseEnv, BaseSystem


class Env(BaseEnv):
    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 quat=np.vstack((1, 0, 0, 0)),
                 omega=np.zeros((3, 1)),
                 ref=np.vstack([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                 rtype="hexa-x"):
        super().__init__(dt=0.001, max_t=10)
        self.plant = Multicopter(pos, vel, quat, omega, rtype)

    def reset(self):
        super().reset()
        return self.observe_flat()

    def get_forces(self, action):
        return action

    def step(self, action):
        self.action = action
        t = self.clock.get()
        states = self.observe_flat()
        info = self.observe_dict()

        forces = self.get_forces(action)
        rotors = self.plant.mixer(forces)
        *_, done = self.update()
        return t, states, info, rotors, done

    def set_dot(self, t):
        forces = self.get_forces(self.action)
        rotors = self.plant.mixer(forces)
        self.plant.set_dot(t, rotors)


class LQR_copter:
    def __init__(self, env, ref):
        self.ref = np.vstack((ref[:6], np.vstack(quat2angle(ref[6:10])[::-1]), ref[10:]))
        Jinv = env.plant.Jinv
        m, g = env.plant.m, env.plant.g
        n_rotors = env.plant.n_rotor
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
        obs = np.vstack((obs))
        return np.vstack((obs[:6], np.vstack(quat2angle(obs[6:10])[::-1]), obs[10:]))

    def get_action(self, obs):
        x = self.observation(obs)
        action = -self.K.dot(x - self.ref) + self.trim_forces
        return action


def run(env, pos, quat, agent=None):
    obs = env.reset()
    logger = fym.logging.Logger(path='data.h5')

    while True:
        env.render()

        if agent is None:
            action = 0
        else:
            action = agent.get_action(obs)

        t, next_obs, info, rotors, done = env.step(action)
        obs = next_obs
        logger.record(t=t, **info, rotors=rotors)

        if done:
            break

    env.close()
    logger.close()


def plot_var():
    data = fym.logging.load('data.h5')
    fig = plt.figure()

    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['plant']['pos'].squeeze())
    ax2.plot(data['t'], data['plant']['vel'].squeeze())
    ax3.plot(data['t'], data['plant']['quat'].squeeze())
    ax4.plot(data['t'], data['plant']['omega'].squeeze())

    ax1.set_ylabel('Position')
    ax1.legend([r'$x$', r'$y$', r'$z$'])
    ax1.grid(True)

    ax2.set_ylabel('Velocity')
    ax2.legend([r'$u$', r'$v$', r'$w$'])
    ax2.grid(True)

    ax3.set_ylabel('Quaternion')
    ax3.legend([r'$q_0$', r'$q_1$', r'$q_2$', r'$q_3$'])
    ax3.grid(True)

    ax4.set_ylabel('Angular Velocity')
    ax4.legend([r'$p$', r'$q$', r'$r$'])
    ax4.set_xlabel('Time [sec]')
    ax4.grid(True)

    plt.tight_layout()

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(6, 1, 1)
    ax2 = fig2.add_subplot(6, 1, 2, sharex=ax1)
    ax3 = fig2.add_subplot(6, 1, 3, sharex=ax1)
    ax4 = fig2.add_subplot(6, 1, 4, sharex=ax1)
    ax5 = fig2.add_subplot(6, 1, 5, sharex=ax1)
    ax6 = fig2.add_subplot(6, 1, 6, sharex=ax1)

    ax1.plot(data['t'], data['rotors'].squeeze()[:, 0])
    ax2.plot(data['t'], data['rotors'].squeeze()[:, 1])
    ax3.plot(data['t'], data['rotors'].squeeze()[:, 2])
    ax4.plot(data['t'], data['rotors'].squeeze()[:, 3])
    ax5.plot(data['t'], data['rotors'].squeeze()[:, 4])
    ax6.plot(data['t'], data['rotors'].squeeze()[:, 5])

    ax1.set_ylabel('rotor1')
    ax1.grid(True)
    ax2.set_ylabel('rotor2')
    ax2.grid(True)
    ax3.set_ylabel('rotor3')
    ax3.grid(True)
    ax4.set_ylabel('rotor4')
    ax4.grid(True)
    ax5.set_ylabel('rotor5')
    ax5.grid(True)
    ax6.set_ylabel('rotor6')
    ax6.grid(True)
    ax6.set_xlabel('Time [sec]')

    plt.tight_layout()


def test_LQR_copter(pos, quat, ref):
    env = Env(pos=pos, quat=quat)
    agent = LQR_copter(env, ref)
    run(env=env, pos=pos, quat=quat, agent=agent)
    plot_var()
    plt.show()


if __name__ == "__main__":
    # reference
    x = 0
    y = 0
    z = 50
    ref = np.vstack([x, y, -z, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    # perturbation
    pos_pertb = ref[:3] - np.vstack([0, 0, 3])
    yaw = 10
    pitch = 0
    roll = 0
    quat_pertb = angle2quat(*np.deg2rad([yaw, pitch, roll]))
    test_LQR_copter(pos=pos_pertb, quat=quat_pertb, ref=ref)
