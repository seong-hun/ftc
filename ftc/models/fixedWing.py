import numpy as np
import numpy.linalg as nla
from numpy import cos, sin
import scipy.linalg as sla
import scipy.interpolate
import scipy.optimize

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2dcm, quat2angle, angle2quat


def get_rho(altitude):
    pressure = 101325 * (1 - 2.25569e-5 * alltitude)**5.25616
    temperature = 288.14 - 0.00649 * altitude
    return pressure / (287*temperature)


class MorphingPlane(BaseEnv):
    g = 9.80665  # [m/s^2]
    m = 9298.6436  # [kg]
    S = 27.8709  # [m]
    cbar = 3.755136  # [m]
    b = 9.144  # [m]
    Tmax = 

    control_limits = {
        "delt": (0, 1),
        "dele": np.deg2rad((-25, 25)),
        "dela": np.deg2rad((-21.5, 21.5)),
        "delr": np.deg2rad((-30, 30)),
        # "eta1" : (0, 1),
        # "eta2" : (0, 1),
    }

    coords = {
        # "eta1" : np.linspace(0, 1, 3),
        # "eta2" : np.linspace(0, 1, 3),
        "dele": np.linspace(control_limits["dele"], 3),
        "alpha": np.deg2rad(np.linspace(-10, 20, 61))
    }

    polycoeffs = {
        "CD": 
        "CL": 
    }

    J = np.array([[12820.614648, 0, 1331.4132386],
                  [0, 75673.623725, 0],
                  [0, 0, 85552.113395]])
    # J_template =
    # J_yy_data =
    # J_yy = scipy.interpolate.interp2d(coords["eta1"], coords["eta2"], J_yy_data)

    def __init__(self,
                 pos=np.vstack((0, 0, 0)),
                 vel=np.vstack((0, 0, 0)),
                 quat=np.vstack((1, 0, 0, 0)),
                 omega=np.vstack((0, 0, 0))):
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.quat = BaseSystem(quat)
        self.omega = BaseSystem(omega)

    def _aero_base(self, name, *x):
        # x = [eta1, eta2, dele, alp]
        a0, a1, a2 = self.polycoeffs[name]
        return a0 + np.dot(a1, x) + np.sum(x * np.dot(a2, x), axis=0)

    def CD(self, eta1, eta2, dele, alp):
        return self._aero_base("CD", eta1, eta2, dele, alp)

    def CL(self, eta1, eta2, dele, alp):
        return self._aero_base("CL", eta1, eta2, dele, alp)

    def Cm(self, eta1, eta2, dele, alp):
        return self._aero_base("Cm", eta1, eta2, dele, alp)

    def get_derivs(self, x, u, eta):
        pos, vel, quat, omega = self.observe_list(x)
        F, M = self.aerodyn(pos, vel, quat, omega, eta)
        J = self.J(*eta)

        # navigation eq
        vpos = quat2dcm(quat).T.dot(vel)
        # force eq
        vvel = F / self.m - np.cross(omega, v)
        # kinematic eq
        vquat = 1 / 2 * np.append(-omega.dot(quat[1:]),
                                  omega*q[0] - np.cross(omega, q[1:]))
        # moment eq
        vomega = nla.inv(J).dot(M - np.cross(omega, J.dot(omega)))

        return vpos, vvel, vquat, vomega

    def aerocoeff(self, *args):
        # *args = [eta1, eta2, dele, alp]
        # output : CL, CD, Cm, CC, Cl, Cn
        return self.CL(*args), self.CD(*args), self.Cm(*args), 0, 0, 0

    def state_readable(self, pos=None, vel=None, quat=None, omega=None, preset="vel"):
        VT = sla.norm(vel)
        alp = np.arctan2(vel[2], vel[0])
        bet = np.arcsin(v[1] / VT)

        if preset == "vel":
            return VT, alp, bet
        else:
            _, theta, _ = quat2angle(quat)
            gamma = theta - alpha
            Q = omega[1]
            return {"VT": VT, "gamma": gamma, "alpha": alp, "Q": Q,
                    "theta": theta, "beta": bet}

    # in fym, aerodyn
    def get_FM(self, pos, vel, quat, omega, u, eta):
        delt, dele, dela, delr = u
        x_cg, z_cg = 0

        VT, alp, bet = self.state_readable(vel=vel, preset="vel")
        qbar = 1 / 2 * get_rho(-p[2]) * VT**2

        CL, CD, Cm, CC, Cl, Cn = self.aerocoeff(*eta, dele, alp)

        CX = cos(alp)*cos(bet)*(-CD) - cos(alp)*sin(bet)*(-CC) - sin(alp)*(-CL)
        CY = sin(bet)*(-CD) + cos(bet)*(-CC) + 0*(-CL)
        CZ = sin(alp)*cos(bet)*(-CD) - sin(alp)*sin(bet)*(-CC) + cos(alp)*(-CL)

        S, cbar, b, Tmax = self.S, self.cbar, self.b, self.Tmax

        # aerodynamic force
        F_A = qbar*S*np.array((CX, CY, CZ))  # [N]
        X_A, Y_A, Z_A = F_A

        # aerodynamic moment
        M_A = qbar*S*np.array((b*Cl + z_cg*Y_A,
                               cbar*Cm + x_cg*Z_A - z_cg*X_A,
                               b*Cn - x_cg*Y_A))  # [Nm]

        # Thruster force and moment
        T = Tmax*delt  # thrust [N]
        X_T, Y_T, Z_T = T, 0, 0  # force, body axes component [N]
        l_T, m_T, n_T = 0, 0, 0  # moment, body axes component [Nm]
        F_T = np.array((X_T, Y_T, Z_T))
        M_T = np.array((l_T, m_T, n_T))

        # gravity force
        F_G = quat2dcm(quat).dot(np.array((0, 0, self.m*self.g)))

        F = F_A + F_T + F_G
        M = M_A + M_T

        return F, M

    def get_trim(self, z0={"alp": 0.1, "delt": 0.13, "dele": 0},
                 fixed={"h"=300, "VT": 16, "eta"=(0, 1)},
                 method="SLSQP", options={"disp": True, "ftol": 1e-10}):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = {(self.cords["alp"].min(), self.coords["alpha"].max()),
                  self.control_limits["delt"],
                  self.control_limits["dele"]}

        result = scipy.optimize.minimize(self._trim_cost(z0, fixed), z0,
                                         args=(fixed,), bounds=bounds,
                                         method=method, options=options)

        return self._trim_convert(result.x, fixed)

    def _trim_cost(self, z, fixed):
        x, u, eta = self._trim_convert(z, fixed)

        self.set_dot(x, u, eta)
        weight = np.diag((1, 1, 100))

        dxs = np.append(self.vel.dot[(0, 2), ], self.omega.dot[1])
        return dxs.dot(weight).dot(dxs)

    def _trim_convert(self, z, fixed):
        h, VT, eta = fixed
        alp = z[0]
        pos = np.array((0, 0, -h))
        vel = np.array((VT*cos(alp), 0, VT*sin(alp)))
        quat = angle2quat(0, alp, 0).reshape(3,)
        omega = np.vstack((0, 0, 0))
        delt, dele, dela, delr = z[1], z[2], 0, 0

        x = np.hstack((pos, vel, quat, omega))
        u = np.array((delt, dele, dela, delr))
        return x, u, eta
