import numpy as np
import numpy.linalg as nla
from numpy import cos, sin
import scipy.linalg as sla
import scipy.interpolate
import scipy.optimize

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2dcm, quat2angle, angle2quat


def get_rho(altitude):
    pressure = 101325 * (1 - 2.25569e-5 * altitude)**5.25616
    temperature = 288.14 - 0.00649 * altitude
    return pressure / (287*temperature)


class FixedWing(BaseEnv):
    g = 9.80665  # [m/s^2]
    mass = 10  # [kg]
    S = 0.84  # reference area (norminal planform area) [m^2]
    # longitudinal reference length (nominal mean aerodynamic chord) [m]
    cbar = 0.288
    b = 3  # lateral reference length (nominal span) [m]
    Tmax = 50  # maximum thrust [N]

    control_limits = {
        "delt": (0, 1),
        "dele": np.deg2rad((-10, 10)),
        "dela": (-0.5, 0.5),
        "delr": (-0.5, 0.5),
    }

    coords = {
        "dele": np.deg2rad(np.linspace(-10, 10, 3)),  # dele
        "alpha": np.deg2rad(np.linspace(-10, 20, 61))  # alpha
    }

    polycoeffs = {
        "CD": [0.03802,
               [-0.0023543, 0.0113488, -0.00549877, 0.0437561],
               [[0.0012769, -0.00220993, 1166.938, 672.113],
                [0.00188837, 0.000115637, -203.85818, -149.4225],
                [-1166.928, 203.8535, 0.1956192, -115.13404],
                [-672.111624, 149.417, 115.76766, 0.994464]]],
        "CL": [0.12816,
               [0.13625538, 0.1110242, 1.148293, 6.0995634],
               [[-0.147822776, 1.064541, 243.35532, -330.0270179],
                [-1.13021511, -0.009309088, 166.28991, -146.8964467],
                [-243.282881, -166.2709286, 0.071258483, 4480.53564],
                [328.541707, 148.945785, -4480.67456545, -0.99765511]]],
        "Cm": [0.09406144,
               [-0.269902, 0.24346326, -7.46727, -2.7296],
               [[0.35794703, -7.433699, 647.83725, -141.0390569],
                [6.8532466, -0.0510021, 542.882121, -681.325],
                [-647.723162, -542.8638, 0.76322739, 2187.33517],
                [135.66547, 678.941, -2186.1196, 0.98880322]]]
    }

    J = np.array([[0.9010, -0.0003, 0.0054],
                  [-0.0003, 9.2949, 0.0],
                  [0.0054, 0.0, 10.1708]])

    def __init__(self, pos, vel, quat, omega):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.quat = BaseSystem(quat)
        self.omega = BaseSystem(omega)

    def _aero_base(self, name, *x):
        # x = [eta1, eta2, dele, alp]
        dele, alp = x
        _x = np.hstack((0, 0, dele, alp))
        a0, a1, a2 = self.polycoeffs[name]
        return a0 + np.dot(a1, _x) + np.sum(_x * np.dot(a2, _x), axis=0)

    def CD(self, dele, alp):
        return self._aero_base("CD", dele, alp)

    def CL(self, dele, alp):
        return self._aero_base("CL", dele, alp)

    def Cm(self, dele, alp):
        return self._aero_base("Cm", dele, alp)

    def deriv(self, pos, vel, quat, omega, u):
        F, M = self.aerodyn(pos, vel, quat, omega, u)
        J = self.J

        w = np.ravel(omega)
        Omega = np.array([[0., -w[2], w[1]],
                          [w[2], 0., -w[0]],
                          [-w[1], w[0], 0.]])
        # force equation
        dvel = F / self.mass - Omega.dot(vel)

        # moment equation
        domega = nla.inv(J).dot(M - np.cross(omega, J.dot(omega), axis=0))

        # kinematic equation
        dquat = 0.5 * np.vstack(np.append(-np.transpose(omega).dot(quat[1:]),
                                          omega*quat[0, 0] - Omega.dot(quat[1:])))

        # navigation equation
        dpos = quat2dcm(quat).T.dot(vel)

        return dpos, dvel, dquat, domega

    def set_dot(self, t, u):
        states = self.observe_list()
        dots = self.deriv(*states, u)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots

    def state_readable(self, pos=None, vel=None, quat=None, omega=None,
                       preset="vel"):
        VT = sla.norm(vel)
        alp = np.arctan2(vel[2], vel[0])
        bet = np.arcsin(vel[1] / VT)

        if preset == "vel":
            return VT, alp, bet
        else:
            _, theta, _ = quat2angle(quat)
            gamma = theta - alp
            Q = omega[1]
            return {'VT': VT, 'gamma': gamma, 'alpha': alp, 'Q': Q,
                    'theta': theta, 'beta': bet}

    def aerocoeff(self, *args):
        # *args: eta1(=0), eta2(=0), dele, alp
        # output: CL, CD, Cm, CC, Cl, Cn
        return self.CL(*args), self.CD(*args), self.Cm(*args), 0, 0, 0

    def aerodyn(self, pos, vel, quat, omega, u):
        delt, dele, dela, delr = u
        x_cg, z_cg = 0, 0

        VT, alp, bet = self.state_readable(vel=vel, preset="vel")
        qbar = 0.5 * get_rho(-pos[2]) * VT**2

        CL, CD, Cm, CC, Cl, Cn = self.aerocoeff(dele, alp)

        CX = cos(alp)*cos(bet)*(-CD) - cos(alp)*sin(bet)*(-CC) - sin(alp)*(-CL)
        CY = sin(bet)*(-CD) + cos(bet)*(-CC) + 0*(-CL)
        CZ = cos(bet)*sin(alp)*(-CD) - sin(alp)*sin(bet)*(-CC) + cos(alp)*(-CL)

        S, cbar, b, Tmax = self.S, self.cbar, self.b, self.Tmax

        X_A = qbar*CX*S  # aerodynamic force along body x-axis
        Y_A = qbar*CY*S  # aerodynamic force along body y-axis
        Z_A = qbar*CZ*S  # aerodynamic force along body z-axis

        # Aerodynamic moment
        l_A = qbar*S*b*Cl + z_cg*Y_A  # w.r.t. body x-axis
        m_A = qbar*S*cbar*Cm + x_cg*Z_A - z_cg*X_A  # w.r.t. body y-axis
        n_A = qbar*S*b*Cn - x_cg*Y_A  # w.r.t. body z-axis

        F_A = np.vstack([X_A, Y_A, Z_A])  # aerodynamic force [N]
        M_A = np.vstack([l_A, m_A, n_A])  # aerodynamic moment [N*m]

        # thruster force and moment are computed here
        T = Tmax*delt  # thrust [N]
        X_T, Y_T, Z_T = T, 0, 0  # thruster force body axes component [N]
        l_T, m_T, n_T = 0, 0, 0  # thruster moment body axes component [N*m]

        # Thruster force, momentum, and gravity force
        F_T = np.vstack([X_T, Y_T, Z_T])  # in body coordinate [N]
        M_T = np.vstack([l_T, m_T, n_T])  # in body coordinate [N*m]
        F_G = quat2dcm(quat).dot(np.array([0, 0, self.mass*self.g])).reshape(3, 1)

        F = F_A + F_T + F_G
        M = M_A + M_T

        return F, M

    def get_trim(self, z0={"alpha": 0.1, "delt": 0.13, "dele": 0},
                 fixed={"h": 300, "VT": 16}, method="SLSQP",
                 options={"disp": True, "ftol": 1e-10}):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            (self.coords["alpha"].min(), self.coords["alpha"].max()),
            self.control_limits["delt"],
            self.control_limits["dele"]
        )
        result = scipy.optimize.minimize(
            self._trim_cost, z0, args=(fixed,),
            bounds=bounds, method=method, options=options)

        return self._trim_convert(result.x, fixed)

    def _trim_cost(self, z, fixed):
        x, u = self._trim_convert(z, fixed)

        self.set_dot(x, u)
        weight = np.diag([1, 1, 1000])

        dxs = np.append(self.vel.dot[(0, 2), ], self.omega.dot[1])
        return dxs.dot(weight).dot(dxs)

    def _trim_convert(self, z, fixed):
        h, VT = fixed
        alp = z[0]
        vel = np.array([VT*cos(alp), 0, VT*sin(alp)])
        omega = np.array([0, 0, 0])
        quat = angle2quat(0, alp, 0)
        pos = np.array([0, 0, -h])
        delt, dele, dela, delr = z[1], z[2], 0, 0

        x = np.hstack((pos, vel, quat, omega))
        u = np.array([delt, dele, dela, delr])
        return x, u


class F16(BaseEnv):
    g = 9.80665  # [m/s^2]
    mass = 9298.6436  # [kg]
    S = 27.8709  # reference area (norminal planform area) [m^2]
    # longitudinal reference length (nominal mean aerodynamic chord) [m]
    cbar = 1.0516624  # [m]
    b = 2.78709  # lateral reference length (nominal span) [m]

    control_limits = {
        "delt": (0, 1),
        "dele": np.deg2rad((-25, 25)),
        "dela": np.deg2rad((-21.5, 21.5)),
        "delr": np.deg2rad((-30, 30))
    }

    coords = {
        "dele": ,  # dele
        "alpha": np.deg2rad(np.linspace(-10, 45, 12))  # alpha
    }

    polycoeffs = {
        "damp": [[-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76,
                  2.05, 1.50, 1.49, 1.83, 1.21],
                 [.882, .852, .876, .958, .962, .974, .819,
                  .483, .590, 1.21, -.493, -1.04],
                 [-.108, -.108, -.188, .110, .258, .226, .344,
                  .362, .611, .529, .298, -2.27],
                 [-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7,
                  -28.2, -29.0, -29.8, -38.3, -35.3],
                 [-.126, -.026, .063, .113, .208, .230, .319,
                  .437, .680, .100, .447, -.330],
                 [-.360, -.359, -.443, -.420, -.383, -.375, -.329,
                  -.294, -.230, -.210, -.120, -.100],
                 [-7.21, -.540, -5.23, -6.11, -6.64, -5.69,
                  -6.00, -6.20, -6.40, -6.60, -6.00],
                 [-.380, -.363, -.378, -.386, -.370, -.453, -.550,
                  -.582, -.595, -.637, -1.02, -.840],
                 [.061, .052, .052, -.012, -.013, -.024, .050,
                  .150, .130, .158, .240, .150]]
        "CD":
        "CL": [0.12816,
               [0.13625538, 0.1110242, 1.148293, 6.0995634],
               [[-0.147822776, 1.064541, 243.35532, -330.0270179],
                [-1.13021511, -0.009309088, 166.28991, -146.8964467],
                [-243.282881, -166.2709286, 0.071258483, 4480.53564],
                [328.541707, 148.945785, -4480.67456545, -0.99765511]]],
        "Cm": [0.09406144,
               [-0.269902, 0.24346326, -7.46727, -2.7296],
               [[0.35794703, -7.433699, 647.83725, -141.0390569],
                [6.8532466, -0.0510021, 542.882121, -681.325],
                [-647.723162, -542.8638, 0.76322739, 2187.33517],
                [135.66547, 678.941, -2186.1196, 0.98880322]]]
    }

    J = np.array([[0.9010, -0.0003, 0.0054],
                  [-0.0003, 9.2949, 0.0],
                  [0.0054, 0.0, 10.1708]])

    def __init__(self, long, euler, omega, pos, POW):
        # long = [VT, alp, bet]
        # euler = [phi, theta, psi]
        # omega = [p, q, r]
        # pos = [x, y, h]
        # POW = actual power level
        super().__init__()
        self.long = BaseSystem(long)
        self.euler = BaseSystem(euler)
        self.omega = BaseSystem(omega)
        self.pos = BaseSystem(pos)
        self.POW = BaseSystem(POW)

    def _aero_base(self, name, *x):
        # x = [eta1, eta2, dele, alp]
        dele, alp = x
        _x = np.hstack((0, 0, dele, alp))
        a0, a1, a2 = self.polycoeffs[name]
        return a0 + np.dot(a1, _x) + np.sum(_x * np.dot(a2, _x), axis=0)

    def CD(self, dele, alp):
        return self._aero_base("CD", dele, alp)

    def CL(self, dele, alp):
        return self._aero_base("CL", dele, alp)

    def Cm(self, dele, alp):
        return self._aero_base("Cm", dele, alp)

    def deriv(self, pos, vel, quat, omega, u):
        # x = [VT, alp, bet, phi, theta, psi, p, q, r, x, y, h, POW
        # u = [delt, dele, dela, delr]

        


    def set_dot(self, t, u):
        states = self.observe_list()
        dots = self.deriv(*states, u)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots

    def state_readable(self, pos=None, vel=None, quat=None, omega=None,
                       preset="vel"):
        VT = sla.norm(vel)
        alp = np.arctan2(vel[2], vel[0])
        bet = np.arcsin(vel[1] / VT)

        if preset == "vel":
            return VT, alp, bet
        else:
            _, theta, _ = quat2angle(quat)
            gamma = theta - alp
            Q = omega[1]
            return {'VT': VT, 'gamma': gamma, 'alpha': alp, 'Q': Q,
                    'theta': theta, 'beta': bet}

    def aerocoeff(self, *args):
        # *args: eta1(=0), eta2(=0), dele, alp
        # output: CL, CD, Cm, CC, Cl, Cn
        return self.CL(*args), self.CD(*args), self.Cm(*args), 0, 0, 0

    def aerodyn(self, pos, vel, quat, omega, u):
        delt, dele, dela, delr = u
        x_cg, z_cg = 0, 0

        VT, alp, bet = self.state_readable(vel=vel, preset="vel")
        qbar = 0.5 * get_rho(-pos[2]) * VT**2

        CL, CD, Cm, CC, Cl, Cn = self.aerocoeff(dele, alp)

        CX = cos(alp)*cos(bet)*(-CD) - cos(alp)*sin(bet)*(-CC) - sin(alp)*(-CL)
        CY = sin(bet)*(-CD) + cos(bet)*(-CC) + 0*(-CL)
        CZ = cos(bet)*sin(alp)*(-CD) - sin(alp)*sin(bet)*(-CC) + cos(alp)*(-CL)

        S, cbar, b, Tmax = self.S, self.cbar, self.b, self.Tmax

        X_A = qbar*CX*S  # aerodynamic force along body x-axis
        Y_A = qbar*CY*S  # aerodynamic force along body y-axis
        Z_A = qbar*CZ*S  # aerodynamic force along body z-axis

        # Aerodynamic moment
        l_A = qbar*S*b*Cl + z_cg*Y_A  # w.r.t. body x-axis
        m_A = qbar*S*cbar*Cm + x_cg*Z_A - z_cg*X_A  # w.r.t. body y-axis
        n_A = qbar*S*b*Cn - x_cg*Y_A  # w.r.t. body z-axis

        F_A = np.vstack([X_A, Y_A, Z_A])  # aerodynamic force [N]
        M_A = np.vstack([l_A, m_A, n_A])  # aerodynamic moment [N*m]

        # thruster force and moment are computed here
        T = Tmax*delt  # thrust [N]
        X_T, Y_T, Z_T = T, 0, 0  # thruster force body axes component [N]
        l_T, m_T, n_T = 0, 0, 0  # thruster moment body axes component [N*m]

        # Thruster force, momentum, and gravity force
        F_T = np.vstack([X_T, Y_T, Z_T])  # in body coordinate [N]
        M_T = np.vstack([l_T, m_T, n_T])  # in body coordinate [N*m]
        F_G = quat2dcm(quat).dot(np.array([0, 0, self.mass*self.g])).reshape(3, 1)

        F = F_A + F_T + F_G
        M = M_A + M_T

        return F, M

    def get_trim(self, z0={"alpha": 0.1, "delt": 0.13, "dele": 0},
                 fixed={"h": 300, "VT": 16}, method="SLSQP",
                 options={"disp": True, "ftol": 1e-10}):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            (self.coords["alpha"].min(), self.coords["alpha"].max()),
            self.control_limits["delt"],
            self.control_limits["dele"]
        )
        result = scipy.optimize.minimize(
            self._trim_cost, z0, args=(fixed,),
            bounds=bounds, method=method, options=options)

        return self._trim_convert(result.x, fixed)

    def _trim_cost(self, z, fixed):
        x, u = self._trim_convert(z, fixed)

        self.set_dot(x, u)
        weight = np.diag([1, 1, 1000])

        dxs = np.append(self.vel.dot[(0, 2), ], self.omega.dot[1])
        return dxs.dot(weight).dot(dxs)

    def _trim_convert(self, z, fixed):
        h, VT = fixed
        alp = z[0]
        vel = np.array([VT*cos(alp), 0, VT*sin(alp)])
        omega = np.array([0, 0, 0])
        quat = angle2quat(0, alp, 0)
        pos = np.array([0, 0, -h])
        delt, dele, dela, delr = z[1], z[2], 0, 0

        x = np.hstack((pos, vel, quat, omega))
        u = np.array([delt, dele, dela, delr])
        return x, u


if __name__ == "__main__":
    pass
