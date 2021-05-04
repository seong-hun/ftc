import numpy as np
import scipy
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2dcm


def skew(x):
    x = x.ravel()  # squeeze
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def T_omega(T):
    return np.array([[0, -T, 0],
                     [T, 0, 0],
                     [0, 0, 0]])

def T_u_inv(T):
    return np.array([[0, 1/T, 0], [-1/T, 0, 0], [0, 0, -1]])


def T_u_inv_dot(T, T_dot):
    return np.array([[0, -T_dot/(T**2), 0],
                     [T_dot/(T**2), 0, 0],
                     [0, 0, 0]])


class BacksteppingController(BaseEnv):
    """ References:
    - Controller
    [1] G. P. Falconi and F. Holzapfel,
    “Adaptive Fault Tolerant Control Allocation for a Hexacopter System,”
    Proc. Am. Control Conf., vol. 2016-July, pp. 6760–6766, 2016.
    - Reference model (e.g., xd, vd, ad, ad_dot, ad_ddot)
    [2] S. J. Su, Y. Y. Zhu, H. R. Wang, and C. Yun, “A Method to Construct a Reference Model for Model Reference Adaptive Control,” Adv. Mech. Eng., vol. 11, no. 11, pp. 1–9, 2019.
    """
    def __init__(self, pos0, m, grav, **kwargs):
        super().__init__(**kwargs)
        self.xd = BaseSystem(pos0)
        self.vd = BaseSystem(np.zeros((3, 1)))
        self.ad = BaseSystem(np.zeros((3, 1)))
        self.ad_dot = BaseSystem(np.zeros((3, 1)))
        self.ad_ddot = BaseSystem(np.zeros((3, 1)))
        self.Td = BaseSystem(m*grav)
        # position
        self.Kx = m*1*np.eye(3)
        self.Kv = m*1*1.82*np.eye(3)
        self.Kp = np.hstack([self.Kx, self.Kv])
        self.Q = np.diag(1*np.ones(6))
        # thrust
        self.Kt = np.diag(4*np.ones(3))
        # angular
        self.Komega = np.diag(20*np.ones(3))
        # reference model
        self.Kxd = np.diag(1*np.ones(3))
        self.Kvd = np.diag(3.4*np.ones(3))
        self.Kad = np.diag(5.4*np.ones(3))
        self.Kad_dot = np.diag(4.9*np.ones(3))
        self.Kad_ddot = np.diag(2.7*np.ones(3))
        # others
        self.Ap = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-(1/m)*self.Kx, -(1/m)*self.Kv],
        ])
        self.Bp = np.block([
            [np.zeros((3, 3))],
            [(1/m)*np.eye(3)],
        ])
        self.P = scipy.linalg.solve_lyapunov(self.Ap.T, -self.Q)

    def dynamics(self, xd, vd, ad, ad_dot, ad_ddot, Td_dot, xc):
        d_xd = vd
        d_vd = ad
        d_ad = ad_dot
        d_ad_dot = ad_ddot
        d_ad_ddot = (-self.Kxd @ xd -self.Kvd @ vd - self.Kad @ ad - self.Kad_dot @ ad_dot - self.Kad_ddot @ ad_ddot + self.Kxd @ xc)
        d_Td = Td_dot
        return d_xd, d_vd, d_ad, d_ad_dot, d_ad_ddot, d_Td

    def set_dot(self, Td_dot, xc):
        xd, vd, ad, ad_dot, ad_ddot, _ = self.observe_list()
        self.xd.dot, self.vd.dot, self.ad.dot, self.ad_dot.dot, self.ad_ddot.dot, self.Td.dot = self.dynamics(xd, vd, ad, ad_dot, ad_ddot, Td_dot, xc)

    def command(self, pos, vel, quat, omega,
                      xd, vd, ad, ad_dot, ad_ddot, Td,
                      m, J, g):
        """Notes:
            Be careful; `rot` denotes the rotation matrix from B- to I- frame,
            which is opposite to the conventional notation in aerospace engineering.
            Please see the reference works carefully.
        """
        rot = quat2dcm(quat).T  # be careful: `rot` denotes the rotation matrix from B-frame to I-frame
        ex = xd - pos
        ev = vd - vel
        ep = np.vstack((ex, ev))
        # u1
        u1 = m * (ad - g) + self.Kp @ ep
        zB = rot.T @ np.vstack((0, 0, 1))
        td = -Td * zB
        et = u1 - td
        Ap, Bp, P = self.Ap, self.Bp, self.P
        ep_dot = Ap @ ep + Bp @ et
        u1_dot = m * ad_dot + self.Kp @ ep_dot
        T = Td  # TODO: no lag
        # u2
        u2 = T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ ep + u1_dot + self.Kt @ et)
        Td_dot = u2[-1]  # third element
        T_dot = Td_dot  # TODO: no lag
        zB_dot = -rot.T @ T_omega(1.0) @ omega
        et_dot = u1_dot + u2[-1] * zB + Td * zB_dot
        ep_ddot = Ap @ ep_dot + Bp @ et_dot
        u1_ddot = m * ad_ddot + self.Kp @ ep_ddot
        rot_dot = -skew(omega) @ rot
        u2_dot = (T_u_inv_dot(T[0], T_dot[0]) @ rot + T_u_inv(T[0]) @ rot_dot) @ (2*Bp.T @ P @ ep + u1_dot + self.Kt@et) + (T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ ep_dot + u1_ddot + self.Kt @ et_dot))
        omegad_dot = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]) @ u2_dot
        omegad = np.vstack((u2[:2], 0))
        eomega = omegad - omega
        Md = np.cross(omega, J@omega, axis=0) + J @ (T_omega(T[0]).T @ rot @ et + omegad_dot + self.Komega @ eomega)
        nud = np.vstack((Td, Md))
        return nud, Td_dot

    def step(self):
        t = self.clock.get()
        info = dict(t=t, **self.observe_dict())
        *_, done = self.update()
        next_obs = self.observe_list()
        return next_obs, np.zeros(1), info, done


if __name__ == "__main__":
    pass
