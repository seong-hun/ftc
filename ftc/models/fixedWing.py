import numpy as np
import numpy.linalg as nla
from numpy import cos, sin, arctan2
import scipy.linalg as sla
from scipy import interpolate
import scipy.optimize

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2dcm, angle2dcm, quat2angle, angle2quat


def get_rho(alt, VT):
    rho = 2.377e-3 * (1 - (.703e-5) * alt) ** (4.14)  # air density
    if alt < 35000:
        temp = 519 * (1 - .703e-5 * alt)  # temperature
    else:
        temp = 390
    Mach = VT / ((1.4 * 1716.3 * temp) ** (.5))
    return rho, Mach


def signum(val):
    if val > 0:
        out = 1
    elif val < 0:
        out = -1
    else:
        out = 0
    return out


class MorphingPlane(BaseEnv):
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
    '''
    F16 model based on
    Stevens, "Aircraft Control and Simulation", Appendix A, 2004
    state (x)   : 13 x 1 vector of [VT, alp, bet, phi, theta, psi, p, q, r, x, y, h, POW]
    input (u)   : 4 x 1 vector of [delt, dele, dela, delr]
    '''
    # unit convertion
    ft2m = 0.3048
    m2ft = 3.280839895013123

    # F-16 Nonlinear Aircraft
    weight = 25000.0  # [lbs]
    g = 32.2
    mass = weight/g
    S = 300.  # [ft^2]
    b = 30.  # [ft]
    cbar = 11.32  # [ft]

    x_cgr = 0.35
    x_cg = 0.35

    en_mx = 160.  # engine angular momentum [slug-ft^2/s]
    en_my = 0.
    en_mz = 0.

    Jxx = 9496.  # [slug-ft^2]
    Jyy = 55814.
    Jzz = 63100.
    Jxz = 982.

    control_limits = {
        "delt": (0, 1),
        "dele": np.deg2rad((-25, 25)),
        "dela": np.deg2rad((-21.5, 21.5)),
        "delr": np.deg2rad((-30, 30))
    }
    polycoeffs = {
        "POW_idle": [[1060., 1060., 670., 880., 1140., 1500., 1860., 1860.],
                     [1060., 1060., 670., 880., 1140., 1500., 1860., 1860.],
                     [635., 635., 425., 690., 1010., 1330., 1700., 1700.],
                     [60., 60., 25., 345., 755., 1130., 1525., 1525.],
                     [-1020., -1020., -710., -300., 350., 910., 1360., 1360.],
                     [-2700., -2700., -1900., -1300., -247., 600., 1100., 1100.],
                     [-3600., -3600., -1400., -595., -342., -200., 700., 700.],
                     [-3600., -3600., -1400., -595., -342., -200., 700., 700.]],
        "POW_mil": [[12680., 12680., 9150., 6200., 3950., 2450., 1400., 1400.],
                    [12680., 12680., 9150., 6200., 3950., 2450., 1400., 1400.],
                    [12680., 12680., 9150., 6313., 4040., 2470., 1400., 1400.],
                    [12610., 12610., 9312., 6610., 4290., 2600., 1560., 1560.],
                    [12640., 12640., 9839., 7090., 4660., 2840., 1660., 1660.],
                    [12390., 12390., 10176., 7750., 5320., 3250., 1930., 1930.],
                    [11680., 11680., 9848., 8050., 6100., 3800., 2310., 2310.],
                    [11680., 11680., 9848., 8050., 6100., 3800., 2310., 2310.]],
        "POW_max": [[20000., 20000., 15000., 10800., 7000., 4000., 2500., 2500.],
                    [20000., 20000., 15000., 10800., 7000., 4000., 2500., 2500.],
                    [21420., 21420., 15700., 11225., 7323., 4435., 2600., 2600.],
                    [22700., 22700., 16860., 12250., 8154., 5000., 2835., 2835.],
                    [24240., 24240., 18910., 13760., 9285., 5700., 3215., 3215.],
                    [26070., 26070., 21075., 15975., 11115., 6860., 3950., 3950.],
                    [28886., 28886., 23319., 18300., 13484., 8642., 5057., 5057.],
                    [28886., 28886., 23319., 18300., 13484., 8642., 5057., 5057.]],
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
                 [-7.21, -.540, -5.23, -5.26, -6.11, -6.64, -5.69,
                  -6.00, -6.20, -6.40, -6.60, -6.00],
                 [-.380, -.363, -.378, -.386, -.370, -.453, -.550,
                  -.582, -.595, -.637, -1.02, -.840],
                 [.061, .052, .052, -.012, -.013, -.024, .050,
                  .150, .130, .158, .240, .150]],
        "CX": [[-.099, -.099, -.081, -.081, -.063, -.025, .044, .097,
                .113, .145, .167, .174, .166, .166],
               [-.099, -.099, -.081, -.081, -.063, -.025, .044, .097,
                .113, .145, .167, .174, .166, .166],
               [-.048, -.048, -.038, -.040, -.021, .016, .083, .127,
                .137, .162, .177, .179, .167, .167],
               [-.022, -.022, -.020, -.021, -.004, .032, .094, .128,
                .130, .154, .161, .155, .138, .138],
               [-.040, -.040, -.038, -.039, -.025, .006, .062, .087,
                .085, .100, .110, .104, .091, .091],
               [-.083, -.083, -.073, -.076, -.072, -.046, .012, .024,
                .025, .043, .053, .047, .040, .040],
               [-.083, -.083, -.073, -.076, -.072, -.046, .012, .024,
                .025, .043, .053, .047, .040, .040]],
        "CZ": [.770, .770, .241, -.100, -.416, -.731, -1.053,
               -1.366, -1.646, -1.917, -2.120, -2.248, -2.229, -2.229],
        "CM": [[.205, .205, .168, .186, .196, .213, .251, .245,
                .248, .252, .231, .298, .192, .192],
               [.205, .205, .168, .186, .196, .213, .251, .245,
                .248, .252, .231, .298, .192, .192],
               [.081, .081, .077, .107, .110, .110, .141, .127,
                .119, .133, .108, .081, .093, .093],
               [-.046, -.046, -.020, -.009, -.005, -.006, .010, .006,
                -.001, .014, .000, -.013, .032, .032],
               [-.174, -.174, -.145, -.121, -.127, -.129, -.102, -.097,
                -.113, -.087, -.084, -.069, -.006, -.006],
               [-.259, -.259, -.202, -.184, -.193, -.199, -.150, -.160,
                -.167, -.104, -.076, -.041, -.005, -.005],
               [-.259, -.259, -.202, -.184, -.193, -.199, -.150, -.160,
                -.167, -.104, -.076, -.041, -.005, -.005]],
        "CL": [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [-.001, -.001, -.004, -.008, -.012, -.016, -.019, -.020,
                -.020, -.015, -.008, -.013, -.015, -.015],
               [-.003, -.003, -.009, -.017, -.024, -.030, -.034, -.040,
                -.037, -.016, -.002, -.010, -.019, -.019],
               [-.001, -.001, -.010, -.020, -.030, -.039, -.044, -.050,
                -.049, -.023, -.006, -.014, -.027, -.027],
               [.000, .000, -.010, -.022, -.034, -.047, -.046, -.059,
                -.061, -.033, -.036, -.035, -.035, -.035],
               [.007, .007, -.010, -.023, -.034, -.049, -.046, -.068,
                -.071, -.060, -.058, -.062, -.059, -.059],
               [.009, .009, -.011, -.023, -.037, -.050, -.047, -.074,
                -.079, -.091, -.076, -.077, -.076, -.076],
               [.009, .009, -.011, -.023, -.037, -.050, -.047, -.074,
                -.079, -.091, -.076, -.077, -.076, -.076]],
        "CN": [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [.018, .018, .019, .018, .019, .019, .018, .013,
                .007, .004, -.014, -.017, -.033, -.033],
               [.038, .038, .042, .042, .042, .043, .039, .030,
                .017, .004, -.035, -.047, -.057, -.057],
               [.056, .056, .057, .059, .058, .058, .053, .032,
                .012, .002, -.046, -.071, -.073, -.073],
               [.064, .064, .077, .076, .074, .073, .057, .029,
                .007, .012, -.034, -.065, -.041, -.041],
               [.074, .074, .086, .093, .089, .080, .062, .049,
                .022, .028, -.012, -.002, -.013, -.013],
               [.079, .079, .090, .106, .106, .096, .080, .068,
                .030, .064, .015, .011, -.001, -.001],
               [.079, .079, .090, .106, .106, .096, .080, .068,
                .030, .064, .015, .011, -.001, -.001]],
        "DLDA": [[-.041, -.041, -.052, -.053, -.056, -.050, -.056, -.082,
                  -.059, -.042, -.038, -.027, -.017, -.017],
                 [-.041, -.041, -.052, -.053, -.056, -.050, -.056, -.082,
                  -.059, -.042, -.038, -.027, -.017, -.017],
                 [-.041, -.041, -.053, -.053, -.053, -.050, -.051, -.066,
                  -.043, -.038, -.027, -.023, -.016, -.016],
                 [-.042, -.042, -.053, -.052, -.051, -.049, -.049, -.043,
                  -.035, -.026, -.016, -.018, -.014, -.014],
                 [-.040, -.040, -.052, -.051, -.052, -.048, -.048, -.042,
                  -.037, -.031, -.026, -.017, -.012, -.012],
                 [-.043, -.043, -.049, -.048, -.049, -.043, -.042, -.042,
                  -.036, -.025, -.021, -.016, -.011, -.011],
                 [-.044, -.044, -.048, -.048, -.047, -.042, -.041, -.020,
                  -.028, -.013, -.014, -.011, -.010, -.010],
                 [-.043, -.043, -.049, -.047, -.045, -.042, -.037, -.003,
                  -.013, -.010, -.003, -.007, -.008, -.008],
                 [-.043, -.043, -.049, -.047, -.045, -.042, -.037, -.003,
                  -.013, -.010, -.003, -.007, -.008, -.008]],
        "DLDR": [[.005, .005, .017, .014, .010, -.005, .009, .019,
                  .005, -.000, -.005, -.011, .008, .008],
                 [.005, .005, .017, .014, .010, -.005, .009, .019,
                  .005, -.000, -.005, -.011, .008, .008],
                 [.007, .007, .016, .014, .014, .013, .009, .012,
                  .005, .000, .004, .009, .007, .007],
                 [.013, .013, .013, .011, .012, .011, .009, .008,
                  .005, -.002, .005, .003, .005, .005],
                 [.018, .018, .015, .015, .014, .014, .014, .014,
                  .015, .013, .011, .006, .001, .001],
                 [.015, .015, .014, .013, .013, .012, .011, .011,
                  .010, .008, .008, .007, .003, .003],
                 [.021, .021, .011, .010, .011, .010, .009, .008,
                  .010, .006, .005, .000, .001, .001],
                 [.023, .023, .010, .011, .011, .011, .010, .008,
                  .010, .006, .014, .020, .000, .000],
                 [.023, .023, .010, .011, .011, .011, .010, .008,
                  .010, .006, .014, .020, .000, .000]],
        "DNDA": [[.001, .001, -.027, -.017, -.013, -.012, -.016, -.001,
                  .017, .011, .017, .008, .016, .016],
                 [.001, .001, -.027, -.017, -.013, -.012, -.016, -.001,
                  .017, .011, .017, .008, .016, .016],
                 [.002, .002, -.014, -.016, -.016, -.014, -.019, -.021,
                  .002, .012, .015, .015, .011, .011],
                 [-.006, -.006, -.008, -.006, -.006, -.005, -.008, -.005,
                  .007, .004, .007, .006, .006, .006],
                 [-.011, -.011, -.011, -.010, -.009, -.008, -.006, .000,
                  .004, .007, .010, .004, .010, .010],
                 [-.015, -.015, -.015, -.014, -.012, -.011, -.008, -.002,
                  .002, .006, .012, .011, .011, .011],
                 [-.024, -.024, -.010, -.004, -.002, -.001, .003, .014,
                  .006, -.001, .004, .004, .006, .006],
                 [-.022, -.022, .002, -.003, -.005, -.003, -.001, -.009,
                  -.009, -.001, .003, -.002, .011, .011],
                 [-.022, -.022, .002, -.003, -.005, -.003, -.001, -.009,
                  -.009, -.001, .003, -.002, .011, .011]],
        "DNDR": [[-.018, -.018, -.052, -.052, -.052, -.053, -.049, -.059,
                  -.051, -.030, -.037, -.026, -.013, -.013],
                 [-.018, -.018, -.052, -.052, -.052, -.053, -.049, -.059,
                  -.051, -.030, -.037, -.026, -.013, -.013],
                 [-.028, -.028, -.051, -.043, -.046, -.045, -.049, -.057,
                  -.052, -.030, -.033, -.030, -.008, -.008],
                 [-.037, -.037, -.041, -.038, -.040, -.040, -.038, -.037,
                  -.030, -.027, -.024, -.019, -.013, -.013],
                 [-.048, -.048, -.045, -.045, -.045, -.044, -.045, -.047,
                  -.048, -.049, -.045, -.033, -.016, -.016],
                 [-.043, -.043, -.044, -.041, -.041, -.040, -.038, -.034,
                  -.035, -.035, -.029, -.022, -.009, -.009],
                 [-.052, -.052, -.034, -.036, -.036, -.035, -.028, -.024,
                  -.023, -.020, -.016, -.010, -.014, -.014],
                 [-.062, -.062, -.034, -.027, -.028, -.027, -.027, -.023,
                  -.023, -.019, -.009, -.025, -.010, -.010],
                 [-.062, -.062, -.034, -.027, -.028, -.027, -.027, -.023,
                  -.023, -.019, -.009, -.025, -.010, -.010]]
    }
    coords = {
        "alp": np.array([-1000, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         1000]),
        "alp_": np.linspace(-10, 45, 12),
        "bet1": np.array([0, 5, 10, 15, 20, 25, 30, 1000]),
        "bet2": np.array([-1000, -30, -20, -10, 0, 10, 20, 30, 1000]),
        "dele": np.array([-10000, -24, -12, 0, 12, 24, 10000]),
        "d": np.linspace(1., 9., 9),
        "h": np.array([-10, 0, 10000, 20000, 30000, 40000, 50000, 1000000]),
        "M": np.array([-10, 0, -.2, 0.4, 0.6, 0.8, 1.0, 10])
    }

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

        # for coefficient interpolation
        self.damp = interpolate.interp2d(self.coords["alp_"], self.coords["d"],
                                         self.polycoeffs["damp"])
        self.CX = interpolate.interp2d(self.coords["alp"], self.coords["dele"],
                                       self.polycoeffs["CX"])
        self.CZ = interpolate.interp1d(self.coords["alp"], self.polycoeffs["CZ"])
        self.CL = interpolate.interp2d(self.coords["alp"], self.coords["bet1"],
                                       self.polycoeffs["CL"])
        self.CM = interpolate.interp2d(self.coords["alp"], self.coords["dele"],
                                       self.polycoeffs["CM"])
        self.CN = interpolate.interp2d(self.coords["alp"], self.coords["bet1"],
                                       self.polycoeffs["CN"])
        self.DLDA = interpolate.interp2d(self.coords["alp"], self.coords["bet2"],
                                         self.polycoeffs["DLDA"])
        self.DLDR = interpolate.interp2d(self.coords["alp"], self.coords["bet2"],
                                         self.polycoeffs["DLDR"])
        self.DNDA = interpolate.interp2d(self.coords["alp"], self.coords["bet2"],
                                         self.polycoeffs["DNDA"])
        self.DNDR = interpolate.interp2d(self.coords["alp"], self.coords["bet2"],
                                         self.polycoeffs["DNDR"])
        self.fidl = interpolate.interp2d(self.coords["h"], self.coords["M"],
                                         self.polycoeffs["POW_idle"])
        self.fmil = interpolate.interp2d(self.coords["h"], self.coords["M"],
                                         self.polycoeffs["POW_mil"])
        self.fmax = interpolate.interp2d(self.coords["h"], self.coords["M"],
                                         self.polycoeffs["POW_max"])

    def TGEAR(self, delt):
        if delt <= 0.77:
            tgear = 64.94 * delt
        else:
            tgear = 217.38 * delt - 117.38
        return tgear

    def PDOT(self, P3, P1):
        if P1 >= 50.:
            if P3 >= 50:
                T = 5.
                P2 = P1
            else:
                P2 = 60.
                T = self.RTAU(P2 - P3)
        else:
            if P3 >= 50:
                T = 5.
                P2 = 40.
            else:
                P2 = P1
                T = self.RTAU(P2 - P3)

        Pdot = T * (P2 - P3)
        return Pdot

    def RTAU(self, dp):
        if dp <= 25.:
            rtau = 1.  # recipropal time constant
        elif dp >= 50.:
            rtau = .1
        else:
            rtau = 1.9 * .036*dp

        return rtau

    def THRUST(self, POW, alt, Mach):
        Tidl = self.fidl(alt, Mach)
        Tmil = self.fmil(alt, Mach)
        Tmax = self.fmax(alt, Mach)
        if POW < 50.:
            thrust = Tidl + (Tmil - Tidl) * POW * .02
        else:
            thrust = Tmil + (Tmax - Tmil) * (POW - 50.) * .02

        return thrust

    def CY(self, bet, dela, delr):  # sideforce coeff
        CY = -.02 * bet + .021 * (dela / 20.) + .086 * (delr / 30.)
        return CY

    def deriv(self, long, euler, omega, pos, POW, u):
        # x = [VT, alp, bet, phi, theta, psi, p, q, r, x, y, h, POW]
        # input and output are SI, calculated in Imperial Unit
        # u = [delt, dele, dela, delr]
        g, mass, S, cbar, b = self.g, self.mass, self.S, self.cbar, self.b
        en_mx, en_my, en_mz = self.en_mx, self.en_my, self.en_mz
        Jxx, Jyy, Jzz, Jxz = self.Jxx, self.Jyy, self.Jzz, self.Jxz

        # Assign state, control variables
        VT = long[0] * self.m2ft
        alp, bet = long[1:]
        _alp, _bet = np.rad2deg(long[1:])
        phi, theta, psi = euler
        p, q, r = omega
        delt = u[0]
        _dele, _dela, _delr = np.rad2deg(u[1:])

        # Standard Atmosphere Model
        alt = pos[2] * self.m2ft
        rho, Mach = get_rho(alt, VT)
        qbar = 1 / 2 * rho * (VT**2)

        # Moment Coefficient
        c1 = ((Jyy-Jzz)*Jzz - Jxz**2) / (Jxx*Jzz-Jxz**2)
        c2 = (Jxx-Jyy+Jzz) * Jxz / (Jxx*Jzz-Jxz**2)
        c3 = Jzz / (Jxx*Jzz-Jxz**2)
        c4 = Jxz / (Jxx*Jzz-Jxz**2)
        c5 = (Jzz-Jxx) / Jyy
        c6 = Jxz / Jyy
        c7 = 1 / Jyy
        c8 = (Jxx*(Jxx-Jyy) + Jxz**2) / (Jxx*Jzz-Jxz**2)
        c9 = Jxx / (Jxx*Jzz-Jxz**2)

        # Aerodynamic data
        CXT = self.CX(_alp, _dele)
        CYT = self.CY(_bet, _dela, _delr)
        CZT = self.CZ(_alp) * (1 - (_bet/57.3)**2) - .19 * (_dele/25.)
        CLT = self.CL(_alp, _bet)*signum(_bet) + self.DLDA(_alp, _bet)*_dela/20.\
            + self.DLDR(_alp, _bet)*_delr/30.
        CMT = self.CM(_alp, _dele)
        CNT = self.CN(_alp, _bet)*signum(_bet) + self.DNDA(_alp, _bet)*_dela/20.\
            + self.DNDR(_alp, _bet)*_delr/30.

        # damping derivatives
        x_cgr = self.x_cgr
        x_cg = self.x_cg
        D = np.zeros((9,))
        for i in range(9):
            D[i] = self.damp(_alp, i+1)
        CQ = cbar * q * .5 / VT
        B2V = b * .5 / VT
        CXT = CXT + CQ * D[0]
        CYT = CYT + B2V * (D[1]*r + D[2]*p)
        CZT = CZT + CQ * D[3]
        CLT = CLT + B2V * (D[4]*r + D[5]*p)
        CMT = CMT + CQ * D[6] + CZT * (x_cgr - x_cg)
        CNT = CNT + B2V * (D[7]*r + D[8]*p) - CYT * (x_cgr - x_cg) * cbar / b

        # Aerodynamic Force & Moment
        Fax = CXT * qbar * S
        Fay = CYT * qbar * S
        Faz = CZT * qbar * S
        La = CLT * qbar * S * b
        Ma = CMT * qbar * S * cbar
        Na = CNT * qbar * S * b

        # Thrust Force & Engine Moment
        CPOW = self.TGEAR(delt)  # throttle gearing
        dPOW = self.PDOT(POW, CPOW)
        T = self.THRUST(POW, alt, Mach)
        Le = q * en_mz - r * en_my
        Me = r * en_mx - p * en_mz
        Ne = p * en_my - q * en_mx

        # Gravity Force
        Fgx = -mass * g * sin(theta)
        Fgy = mass * g * sin(phi) * cos(theta)
        Fgz = mass * g * cos(phi) * cos(theta)

        # External Force & Moment
        Fx = Fax + Fgx + T
        Fy = Fay + Fgy
        Fz = Faz + Fgz
        L = La + Le
        M = Ma + Me
        N = Na + Ne

        # Euler Velocity Definition
        U = VT * cos(alp) * cos(bet)
        V = VT * sin(bet)
        W = VT * sin(alp) * cos(bet)

        # Force equation
        dU = r*V - q*W + Fx / mass
        dV = p*W - r*U + Fy / mass
        dW = q*U - p*V + Fz / mass
        dVT = (U*dU + V*dV + W*dW) / VT
        dalp = (U*dW - W*dU) / (U**2 + W**2)
        dbet = (VT*dV - V*dVT) / cos(bet) / (U**2 + W**2)

        # kinematic equation
        dphi = p + sin(theta) / cos(theta) * (q*sin(phi) + r*cos(phi))
        dtheta = q*cos(phi) - r*sin(phi)
        dpsi = (q*sin(phi) + r*cos(phi)) / cos(theta)

        # moments equation
        dp = (c1*r + c2*p)*q + c3*L + c4*N
        dq = c5*p*r - c6*(p**2 - r**2) + c7*M
        dr = (c8*p - c2*r)*q + c4*L + c9*N

        # navigation equation
        dpn = U*cos(theta)*cos(psi)\
            + V*(sin(phi)*cos(psi)*sin(theta) - cos(phi)*sin(psi))\
            + W*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
        dpe = U*cos(theta)*sin(psi)\
            + V*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))\
            + W*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
        dpd = U*sin(theta) - V*sin(phi)*cos(theta) - W*cos(phi)*cos(theta)

        dlong = np.vstack((dVT*self.ft2m, dalp, dbet))
        deuler = np.vstack((dphi, dtheta, dpsi))
        domega = np.vstack((dp, dq, dr))
        dpos = np.vstack((dpn, dpe, dpd)) * self.ft2m

        return dlong, deuler, domega, dpos, dPOW

    def set_dot(self, t, u):
        states = self.observe_list()
        dots = self.deriv(*states, u)
        self.long.dot, self.euler.dot, self.omega.dot, self.pos.dot, self.POW.dot = dots

    def set_dot_trim(self, x, u):
        long = x[0:3]
        euler = x[3:6]
        omega = x[6:9]
        pos = x[9:12]
        POW = x[12]
        dots = self.deriv(long, euler, omega, pos, POW, u)
        self.long.dot, self.euler.dot, self.omega.dot, self.pos.dot, self.POW.dot = dots

    # def aerodyn(self, long, euler, omega, pos, POW, u):
    #     delt, dele, dela, delr = u
    #     S, cbar, b = self.S, self.cbar, self.b

    #     VT, alp, bet = long
    #     p, q, r = omega
    #     rho, *_ = get_rho(pos[2])
    #     qbar = 0.5 * rho * VT**2

    #     # look-up table and component buildup
    #     CXT = self.CX(alp, dele)
    #     CYT = self.CY(bet, dela, delr)
    #     CZT = self.CZ(alp, bet, dele)
    #     CLT = self.CL(alp, bet) + self.DLDA(alp, bet)*dela/20.\
    #         + self.DLDR(alp, bet)*delr/30.
    #     CMT = self.CM(alp, dele)
    #     CNT = self.CN(alp, bet) + self.DNDA(alp, bet)*dela/20.\
    #         + self.DNDR(alp, bet)*delr/30.

    #     # damping derivatives
    #     x_cgr = self.x_cgr
    #     x_cg = x_cgr
    #     D1, D2, D3, D4, D5, D6, D7, D8, D9 = self.damp(alp)
    #     CQ = .5 * cbar * q / VT
    #     B2V = .5 * b / VT
    #     CXT = CXT + CQ * D1
    #     CYT = CYT + B2V * (D2*r + D3*p)
    #     CZT = CZT + CQ * D4
    #     CLT = CLT + B2V * (D5*r + D6*p)
    #     CMT = CMT + CQ * D7 + CZT * (x_cgr - x_cg)

    #     X = qbar*CXT*S  # aerodynamic force along body x-axis
    #     Y = qbar*CYT*S  # aerodynamic force along body y-axis
    #     Z = qbar*CZT*S  # aerodynamic force along body z-axis

    #     l = qbar * S * b * CLT
    #     m = qbar * S * cbar * CMT
    #     n = qbar * S * b * CNT

    #     F_AT = np.vstack([X, Y, Z])  # aerodynamic & thrust force [N]
    #     M_AT = np.vstack([l, m, n])  # aerodynamic & thrust moment [N*m]

    #     # gravity force
    #     F_G = angle2dcm(euler).dot(np.array([0, 0, self.mass*self.g])).reshape(3, 1)

    #     F = F_AT + F_G
    #     M = M_AT

    #     return F, M

    def lin_mode(self, x_t, u_t):
        # numerical computation of state-space model
        long = x_t[0:3]
        euler = x_t[3:6]
        omega = x_t[6:9]
        pos = x_t[9:12]
        POW = x_t[12]
        dx0 = np.vstack((self.deriv(long, euler, omega, pos, POW, u_t)))

        ptrb = 1e-9

        N = 13  # state variables
        M = 4  # input variables

        dfdx = np.zeros((N, N))
        for i in range(0, N):
            ptrbvecx = np.zeros((N, 1))
            ptrbvecx[i] = ptrb
            long_p = long + ptrbvecx[0:3]
            euler_p = euler + ptrbvecx[3:6]
            omega_p = omega + ptrbvecx[6:9]
            pos_p = pos + ptrbvecx[9:12]
            POW_p = POW + ptrbvecx[12]
            dx = np.vstack((self.deriv(long_p, euler_p, omega_p, pos_p, POW_p, u_t)))
            dfdx[:, i] = (dx[:, 0] - dx0[:, 0]) / ptrb

        dfdu = np.zeros((N, M))
        for i in range(0, M):
            ptrbvecu = np.zeros((M, 1))
            ptrbvecu[i] = ptrb
            dx = np.vstack((self.deriv(long, euler, omega, pos, POW, (u_t + ptrbvecu))))
            dfdu[:, i] = (dx[:, 0] - dx0[:, 0]) / ptrb

        return dfdx, dfdu

    def get_trim(self, z0={"delt": 0.1385, "dele": -np.deg2rad(0.7588),
                           "alp": 0.036, "dela": -np.deg2rad(1.2e-7),
                           "delr": np.deg2rad(6.2e-7), "bet": -4e-9},
                 fixed={"VT": 153.0096, "psi": 0., "pn": 0., "pe": 0., "h": 0.},
                 method="SLSQP", options={"disp": True, "ftol": 1e-10}):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            self.control_limits["delt"],
            self.control_limits["dele"],
            (np.deg2rad(-10), np.deg2rad(45)),
            self.control_limits["dela"],
            self.control_limits["delr"],
            (np.deg2rad(-30), np.deg2rad(30))
        )
        result = scipy.optimize.minimize(
            self._trim_cost, z0, args=(fixed,),
            bounds=bounds, method=method, options=options)

        return self.constraint(result.x, fixed)

    def _trim_cost(self, z, fixed):
        x, u = self.constraint(z, fixed)

        self.set_dot_trim(x, u)
        dVT, dalp, dbet = self.long.dot
        dp, dq, dr = self.omega.dot

        cost = 2*dVT**2 + 100*(dalp**2 + dbet**2) + 10*(dp**2 + dq**2 + dr**2)
        return cost

    def constraint(self, z, fixed,
                   constr={"gamma": 0., "RR": 0., "PR": 0., "TR": 0.,
                           "phi": 0., "coord": 0, "stab": 0.}):
        # RR(roll rate), PR(pitch rate), TR(turnning rate)
        # coord(coordinated turn logic), stab(stability axis roll)
        delt, dele, alp, dela, delr, bet = z
        gamma, RR, PR, TR, phi, coord, stab = list(constr.values())
        X0, X5, X9, X10, X11 = fixed
        X1 = alp
        X2 = bet
        X12 = self.TGEAR(delt)

        if coord:
            # coordinated turn logic
            pass
        elif TR != 0:
            # skidding turn logic
            pass
        else:
            X3 = phi
            D = alp

            if phi != 0:  # inverted
                D = -alp
            elif sin(gamma) != 0:  # climbing
                X4 = D + arctan2(sin(gamma)/cos(bet), (1-(sin(gamma)/cos(bet))**2)**(0.5))
            else:
                X4 = D

            X6 = RR
            X7 = PR

            if stab:  # stability axis roll
                X8 = RR * sin(alp) / cos(alp)
            else:  # body axis roll
                X8 = 0

        long = np.array((X0, X1, X2))
        euler = np.array((X3, X4, X5))
        omega = np.array((X6, X7, X8))
        pos = np.array((X9, X10, X11))
        POW = X12

        x = np.hstack((long, euler, omega, pos, POW))
        u = np.array((delt, dele, dela, delr))
        return x, u


class F16lon(BaseSystem, F16):
    ''' Nonlinear Simulation of F16 only considering the longitudinal dynamics
    state (x)   : 5 x 1 vector of (VT, gamma, h, alp, q)
    control (u) : 2 x 1 vector of (delt, dele)
    '''
    Tmax = 12746.4398211  # calculated base on fym/models/aircraft/MorphingPlane/.py 검증 필요

    def __init__(self, lon=None):
        if lon is None:
            lon, *_ = self.get_trim()

        super().__init__(lon)
        self.damp = interpolate.interp2d(self.coords["alp_"], self.coords["d"],
                                         self.polycoeffs["damp"])
        self.CX = interpolate.interp2d(self.coords["alp"], self.coords["dele"],
                                       self.polycoeffs["CX"])
        self.CZ = interpolate.interp1d(self.coords["alp"], self.polycoeffs["CZ"])
        self.CM = interpolate.interp2d(self.coords["alp"], self.coords["dele"],
                                       self.polycoeffs["CM"])

    def deriv(self, lon, u):
        # input and output are SI unit, calculation is in Imperial Unit
        g, mass, S, cbar = self.g, self.mass, self.S, self.cbar
        Jyy, Tmax = self.Jyy, self.Tmax

        # Assign state, control variables
        VT = lon[0] * self.m2ft
        h = lon[2] * self.m2ft
        gamma, alp, q = lon[1], lon[3], lon[4]
        _alp = np.rad2deg(lon[3])
        _bet = 0.
        delt, _dele = u[0], np.rad2deg(u[1])

        # look-up table and component buildup
        CXT = self.CX(_alp, _dele)
        CZT = self.CZ(_alp) * (1 - (_bet/57.3)**2) - .19 * (_dele/25.)
        CMT = self.CM(_alp, _dele)

        # damping derivatives
        x_cgr = self.x_cgr
        x_cg = x_cgr
        Dx, Dz, Dm = self.damp(_alp, 1), self.damp(_alp, 4), self.damp(_alp, 7)
        CQ = cbar * q * .5 / VT
        CXT = CXT + CQ * Dx
        CZT = CZT + CQ * Dz
        CMT = CMT + CQ * Dm + CZT * (x_cgr - x_cg)

        # aerodynamic & thruster force & moment
        CL = CXT * sin(alp) - CZT * cos(alp)
        CD = -CXT * cos(alp) - CZT * sin(alp)
        rho, *_ = get_rho(h, VT)
        qbar = .5 * rho * VT**2
        L = qbar * S * CL
        D = qbar * S * CD
        M = qbar * S * cbar * CMT
        T = Tmax * delt

        # derivs
        dVT = (T*cos(alp) - D) / mass - g * sin(gamma)
        dgamma = (L + T*sin(alp)) / mass / VT - g * cos(gamma) / VT
        dh = VT * sin(gamma)
        dalp = q - dgamma
        dq = M / Jyy

        return np.vstack((dVT*self.ft2m, dgamma, dh*self.ft2m, dalp, dq))

    def set_dot(self, t, u):
        self.dot = self.deriv(self.state, u)

    def _trim_cost(self, z, fixed):
        x, u = self._trim_convert(z, fixed)
        dxs = self.deriv(x, u)
        cost = dxs[0]**2 + dxs[1]**2 + dxs[2]**2 + dxs[3]**2 + 100*dxs[4]**2
        return cost

    def _trim_convert(self, z, fixed):
        V, h = fixed
        alp, delt, dele = z
        q, gamma = 0, 0

        x = np.vstack([V, gamma, h, alp, q])
        u = np.vstack([delt, dele])
        return x, u

    def get_trim(self, z={"alp": 0.036, "delt": 0.1385, "dele": -np.deg2rad(0.7588)},
                 fixed={"VT": 153.0096, "h": 0}, method="SLSQP",
                 options={"disp": True, "ftol": 1e-10}, verbose=False):
        z = list(z.values())
        fixed = list(fixed.values())
        bounds = (
            (np.deg2rad(-10), np.deg2rad(45)),
            self.control_limits["delt"],
            self.control_limits["dele"]
        )
        result = scipy.optimize.minimize(
            self._trim_cost, z, args=(fixed,),
            bounds=bounds, method=method, options=options)
        x, u = self._trim_convert(result.x, fixed)
        return x, u

    def lin_mode_lon(self, A, B):
        # get A, B matrix from F16, and transform to longitudinal A, B
        N = 13
        M = 4

        Elon1 = np.zeros((5, N))
        Elon1[0, 0] = 1  # VT
        Elon1[1, 1] = 1  # alp
        Elon1[2, 4] = 1  # theta
        Elon1[3, 7] = 1  # q
        Elon1[4, 11] = 1  # h

        Elon2 = np.zeros((2, M))
        Elon2[0, 0] = 1  # delt
        Elon2[1, 1] = 1  # dele

        Alon = Elon1.dot(A).dot(Elon1.T)
        Blon = Elon1.dot(B).dot(Elon2.T)

        # state transition from theta to gamma
        Alon[2, :] = Alon[2, :] - Alon[1, :]
        Blon[2, :] = Blon[2, :] - Blon[1, :]

        # match VT, alp, gamma, q, h to VT, gamma, h, a, q
        _Alon = Alon.copy()
        _Blon = Blon.copy()

        Alon[1, :] = _Alon[2, :]
        Alon[2, :] = _Alon[4, :]
        Alon[3, :] = _Alon[1, :]
        Alon[4, :] = _Alon[3, :]

        Blon[1, :] = _Blon[2, :]
        Blon[2, :] = _Blon[4, :]
        Blon[3, :] = _Blon[1, :]
        Blon[4, :] = _Blon[3, :]

        # augmentated matrix for lqi
        # longitudinal state = [VT, gamma, h, alp, q]
        # performance output = [VT, gamma]
        Elon = np.array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0]])
        AlonAug1 = np.vstack([Alon, Elon])
        AlonAug2 = np.vstack([np.zeros((5, 2)), np.zeros((2, 2))])
        AlonAug = np.hstack([AlonAug1, AlonAug2])
        BlonAug = np.vstack([Blon, np.zeros((2, 2))])

        return Alon, Blon, AlonAug, BlonAug


if __name__ == "__main__":
    long = np.vstack((502, 5.05807418e-02, 7.85459681e-07))
    euler = np.vstack((0., 5.05807418e-02, 0.))
    omega = np.vstack((0., 0., 0.))
    pos = np.vstack((0., 0., 0.))
    POW = 1.00031243e+1
    u = np.vstack((0.154036408, -1.21242062e-2, -2.91493958e-7, 1.86731213e-6))
    system = F16(long, euler, omega, pos, POW)

    # f16 lon
    system1 = F16lon(np.zeros((5, 1)))

    # print(repr(system))
    print(system.get_trim())
    print(system1.get_trim())
