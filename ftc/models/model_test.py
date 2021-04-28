import numpy as np
import matplotlib.pyplot as plt

from multicopter import Multicopter

# testing list
test1 = 0 * np.ones((6, 1))
test2 = 50 * np.ones((6, 1))
r1, r2, r3, r4, r5, r6 = 0, 0, 0, 50, 0, 100
test3 = np.vstack([r1, r2, r3, r4, r5, r6])

pos = np.zeros((3, 1))
vel = np.zeros((3, 1))
quat = np.vstack([1, 0, 0, 0])
omeg = np.zeros((3, 1))

system = Multicopter(pos=pos, vel=vel, quat=quat, omega=omeg)

t = np.array([0])
dt = 0.01

for i in range(20):
    dpos, dvel, dquat, domeg = system.deriv(
        pos[:, -1:], vel[:, -1:], quat[:, -1:], omeg[:, -1:], test3)

    t = np.append(t, dt + t[-1])
    pos = np.hstack((pos, dt * dpos + pos[:, -1:]))
    vel = np.hstack((vel, dt * dvel + vel[:, -1:]))
    quat = np.hstack((quat, dt * dquat + quat[:, -1:]))
    omeg = np.hstack((omeg, dt * domeg + omeg[:, -1:]))

fig = plt.figure()
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)

ax1.plot(t, pos[0, :], t, pos[1, :], t, pos[2, :])
ax2.plot(t, vel[0, :], t, vel[1, :], t, vel[2, :])
ax3.plot(t, quat[0, :], t, quat[1, :], t, quat[2, :], t, quat[3, :])
ax4.plot(t, omeg[0, :], t, omeg[1, :], t, omeg[2, :])

ax1.set_xlabel('t')
ax1.set_ylabel('pos')
ax1.legend(['posx', 'posy', 'posz'])
ax2.set_xlabel('t')
ax2.set_ylabel('vel')
ax2.legend(['u', 'v', 'w'])
ax3.set_xlabel('t')
ax3.set_ylabel('quat')
ax3.legend(['q0', 'q1', 'q2', 'q3'])
ax4.set_xlabel('t')
ax4.set_ylabel('omega')
ax4.legend(['p', 'q', 'r'])

plt.show()
