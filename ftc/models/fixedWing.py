import numpy as np
import numpy.linalg as nla
from numpy import cos, sin, arctan2
import scipy.linalg as sla
from scipy import interpolate
import scipy.optimize

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2dcm, angle2dcm, quat2angle, angle2quat


def get_rho(altitude):
    pressure = 101325 * (1 - 2.25569e-5 * altitude)**5.25616
    temperature = 288.14 - 0.00649 * altitude
    return pressure / (287*temperature)


def ADC(altitude, VT):  # air data computer
    R0 = 2.377e-3  # see level density
    Tfac = 1. - .703e-5 * altitude
    T = 519. * Tfac  # temperature
    if altitude >= 35000.:
        T = 390.
    rho = R0 * (Tfac**4.14)  # density
    Mach = VT / (1.4*1716.3*T)**(1/2)  # Mach number
    return rho, Mach


def signum(a, b):
    if b > 0:
        c = 1
    elif b < 0:
        c = -1
    else:
        c = 0
    return c


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
    weight = 25000.0  # [lbs]
    g = 32.17
    mass = weight/g
    S = 300.  # [ft^2]
    b = 30.  # [ft]
    cbar = 11.32  # [ft]
    x_cgr = 0.35
    hx = 160.  # engine angular momentum [slug-ft^2/s]
    Jxx = 9496.  # [slug-ft^2]
    Jyy = 55814.
    Jzz = 63100.
    Jxz = 982.
    Xpq = Jxz * (Jxx - Jyy + Jzz)
    Xqr = Jzz * (Jzz - Jyy) + Jxz**2
    Zpq = (Jxx - Jyy) * Jxx + Jxz**2
    Ypr = Jzz - Jxx
    gam = Jxx * Jzz - Jxz**2

    control_limits = {
        "delt": (0, 1),
        "dele": np.deg2rad((-25, 25)),
        "dela": np.deg2rad((-21.5, 21.5)),
        "delr": np.deg2rad((-30, 30))
    }
    polycoeffs = {
        "POW_idle": [[1060., 670., 880., 1140., 1500., 1860.],
                     [635., 425., 690., 1010., 1330., 1700.],
                     [60., 25., 345., 755., 1130., 1525.],
                     [-1020., -710., -300., 350., 910., 1360.],
                     [-2700., -1900., -1300., -247., 600., 1100.],
                     [-3600., -1400., -595., -342., -200., 700.]],
        "POW_mil": [[12680., 9150., 6200., 3950., 2450., 1400.],
                    [12680., 9150., 6313., 4040., 2470., 1400.],
                    [12610., 9312., 6610., 4290., 2600., 1560.],
                    [12640., 9839., 7090., 4660., 2840., 1660.],
                    [12390., 10176., 7750., 5320., 3250., 1930.],
                    [11680., 9848., 8050., 6100., 3800., 2310.]],
        "POW_max": [[20000., 15000., 10800., 7000., 4000., 2500.],
                    [21420., 15700., 11225., 7323., 4435., 2600.],
                    [22700., 16860., 12250., 8154., 5000., 2835.],
                    [24240., 18910., 13760., 9285., 5700., 3215.],
                    [26070., 21075., 15975., 11115., 6860., 3950.],
                    [28886., 23319., 18300., 13484., 8642., 5057.]],
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
        "CX": [[-.099, -.081, -.081, -.063, -.025, .044, .097,
                .113, .145, .167, .174, .166],
               [-.048, -.038, -.040, -.021, .016, .083, .127,
                .137, .162, .177, .179, .167],
               [-.022, -.020, -.021, -.004, .032, .094, .128,
                .130, .154, .161, .155, .138],
               [-.040, -.038, -.039, -.025, .006, .062, .087,
                .085, .100, .110, .104, .091],
               [-.083, -.073, -.076, -.072, -.046, .012, .024,
                .025, .043, .053, .047, .040]],
        "CZ": [.770, .241, -.100, -.416, -.731, -1.053,
               -1.366, -1.646, -1.917, -2.120, -2.248, -2.229],
        "CM": [[.205, .168, .186, .196, .213, .251, .245,
                .248, .252, .231, .298, .192],
               [.081, .077, .107, .110, .110, .141, .127,
                .119, .133, .108, .081, .093],
               [-.046, -.020, -.009, -.005, -.006, .010, .006,
                -.001, .014, .000, -.013, .032],
               [-.174, -.145, -.121, -.127, -.129, -.102, -.097,
                -.113, -.087, -.084, -.069, -.006],
               [-.259, -.202, -.184, -.193, -.199, -.150, -.160,
                -.167, -.104, -.076, -.041, -.005]],
        "CL": [[-.001, -.004, -.008, -.012, -.016, -.019, -.020,
                -.020, -.015, -.008, -.013, -.015],
               [-.003, -.009, -.017, -.024, -.030, -.034, -.040,
                -.037, -.016, -.002, -.010, -.019],
               [-.001, -.010, -.020, -.030, -.039, -.044, -.050,
                -.049, -.023, -.006, -.014, -.027],
               [.000, -.010, -.022, -.034, -.047, -.046, -.059,
                -.061, -.033, -.036, -.035, -.035],
               [.007, -.010, -.023, -.034, -.049, -.046, -.068,
                -.071, -.060, -.058, -.062, -.059],
               [.009, -.011, -.023, -.037, -.050, -.047, -.074,
                -.079, -.091, -.076, -.077, -.076]],
        "CN": [[.018, .019, .018, .019, .019, .018, .013,
                .007, .004, -.014, -.017, -.033],
               [.038, .042, .042, .042, .043, .039, .030,
                .017, .004, -.035, -.047, -.057],
               [.056, .057, .059, .058, .058, .053, .032,
                .012, .002, -.046, -.071, -.073],
               [.064, .077, .076, .074, .073, .057, .029,
                .007, .012, -.034, -.065, -.041],
               [.074, .086, .093, .089, .080, .062, .049,
                .022, .028, -.012, -.002, -.013],
               [.079, .090, .106, .106, .096, .080, .068,
                .030, .064, .015, .011, -.001]],
        "DLDA": [[-.041, -.052, -.053, -.056, -.050, -.056, -.082,
                  -.059, -.042, -.038, -.027, -.017],
                 [-.041, -.053, -.053, -.053, -.050, -.051, -.066,
                  -.043, -.038, -.027, -.023, -.016],
                 [-.042, -.053, -.052, -.051, -.049, -.049, -.043,
                  -.035, -.026, -.016, -.018, -.014],
                 [-.040, -.052, -.051, -.052, -.048, -.048, -.042,
                  -.037, -.031, -.026, -.017, -.012],
                 [-.043, -.049, -.048, -.049, -.043, -.042, -.042,
                  -.036, -.025, -.021, -.016, -.011],
                 [-.044, -.048, -.048, -.047, -.042, -.041, -.020,
                  -.028, -.013, -.014, -.011, -.010],
                 [-.043, -.049, -.047, -.045, -.042, -.037, -.003,
                  -.013, -.010, -.003, -.007, -.008]],
        "DLDR": [[.005, .017, .014, .010, -.005, .009, .019,
                  .005, -.000, -.005, -.011, .008],
                 [.007, .016, .014, .014, .013, .009, .012,
                  .005, .000, .004, .009, .007],
                 [.013, .013, .011, .012, .011, .009, .008,
                  .005, -.002, .005, .003, .005],
                 [.018, .015, .015, .014, .014, .014, .014,
                  .015, .013, .011, .006, .001],
                 [.015, .014, .013, .013, .012, .011, .011,
                  .010, .008, .008, .007, .003],
                 [.021, .011, .010, .011, .010, .009, .008,
                  .010, .006, .005, .000, .001],
                 [.023, .010, .011, .011, .011, .010, .008,
                  .010, .006, .014, .020, .000]],
        "DNDA": [[.001, -.027, -.017, -.013, -.012, -.016, -.001,
                  .017, .011, .017, .008, .016],
                 [.002, -.014, -.016, -.016, -.014, -.019, -.021,
                  .002, .012, .015, .015, .011],
                 [-.006, -.008, -.006, -.006, -.005, -.008, -.005,
                  .007, .004, .007, .006, .006],
                 [-.011, -.011, -.010, -.009, -.008, -.006, .000,
                  .004, .007, .010, .004, .010],
                 [-.015, -.015, -.014, -.012, -.011, -.008, -.002,
                  .002, .006, .012, .011, .011],
                 [-.024, -.010, -.004, -.002, -.001, .003, .014,
                  .006, -.001, .004, .004, .006],
                 [-.022, .002, -.003, -.005, -.003, -.001, -.009,
                  -.009, -.001, .003, -.002, .011]],
        "DNDR": [[-.018, -.052, -.052, -.052, -.053, -.049, -.059,
                  -.051, -.030, -.037, -.026, -.013],
                 [-.028, -.051, -.043, -.046, -.045, -.049, -.057,
                  -.052, -.030, -.033, -.030, -.008],
                 [-.037, -.041, -.038, -.040, -.040, -.038, -.037,
                  -.030, -.027, -.024, -.019, -.013],
                 [-.048, -.045, -.045, -.045, -.044, -.045, -.047,
                  -.048, -.049, -.045, -.033, -.016],
                 [-.043, -.044, -.041, -.041, -.040, -.038, -.034,
                  -.035, -.035, -.029, -.022, -.009],
                 [-.052, -.034, -.036, -.036, -.035, -.028, -.024,
                  -.023, -.020, -.016, -.010, -.014],
                 [-.062, -.034, -.027, -.028, -.027, -.027, -.023,
                  -.023, -.019, -.009, -.025, -.010]]
    }
    coords = {
        "alp": np.linspace(-10., 45., 12),
        "bet1": np.linspace(0., 30., 6),
        "bet2": np.linspace(-30., 30., 7),
        "dele": np.linspace(-25., 25., 5),
        "d": np.linspace(1., 9., 9),
        "h": np.linspace(0, 50000, 6),
        "M": np.linspace(0, 1, 6)
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
        A = self.polycoeffs["POW_idle"]
        B = self.polycoeffs["POW_mil"]
        C = self.polycoeffs["POW_max"]

        ch = self.coords["h"]
        cM = self.coords["M"]
        fidl = interpolate.interp2d(ch, cM, A)
        fmil = interpolate.interp2d(ch, cM, B)
        fmax = interpolate.interp2d(ch, cM, C)
        Tidl = fidl(alt, Mach)
        Tmil = fmil(alt, Mach)
        Tmax = fmax(alt, Mach)
        if POW < 50.:
            thrust = Tidl + (Tmil - Tidl) * POW * .02
        else:
            thrust = Tmil + (Tmax - Tmil) * (POW - 50.) * .02

        return thrust

    def damp(self, alp):
        A = self.polycoeffs["damp"]

        # s = .2*alp
        # k = int(s)
        # if k <= -2:
        #     k = -1
        # elif k >= 9:
        #     k = 8
        # da = s - float(k)
        # l = k + int(signum(1, da))
        # D = np.zeros((9,))
        # for i in range(9):
        #     D[i] = A[k+2][i] + abs(da) * (A[l+2][i] - A[k+2][i])
        calp = self.coords["alp"]
        cd = self.coords["d"]
        f = interpolate.interp2d(calp, cd, A)
        D = np.zeros((9,))
        for i in range(9):
            D[i] = f(alp, i+1)
        return D

    def CX(self, alp, dele):  # x-axis aerodynamic force coeff.
        A = self.polycoeffs["CX"]

        calp = self.coords["alp"]
        cdele = self.coords["dele"]
        f = interpolate.interp2d(calp, cdele, A)
        return f(alp, dele)

    def CY(self, bet, dela, delr):  # sideforce coeff
        CY = -.02 * bet + .021 * (dela / 20.) + .086 * (delr / 30.)
        return CY

    def CZ(self, alp, bet, dele):  # z-axis force coeff
        A = self.polycoeffs["CZ"]

        calp = self.coords["alp"]
        f = interpolate.interp1d(calp, A)
        CZ = f(alp) * (1 - (bet/57.3)**2) - .19 * (dele/25.)
        return CZ

    def CM(self, alp, dele):  # pitching moment coeff
        A = self.polycoeffs["CM"]

        calp = self.coords["alp"]
        cdele = self.coords["dele"]
        f = interpolate.interp2d(calp, cdele, A)
        return f(alp, dele)

    def CL(self, alp, bet):  # rolling moment coeff
        A = self.polycoeffs["CL"]

        calp = self.coords["alp"]
        cbet = self.coords["bet1"]
        f = interpolate.interp2d(calp, cbet, A)
        return f(alp, bet)

    def CN(self, alp, bet):  # yawing moment coeff
        A = self.polycoeffs["CN"]

        calp = self.coords["alp"]
        cbet = self.coords["bet1"]
        f = interpolate.interp2d(calp, cbet, A)
        return f(alp, bet)

    def DLDA(self, alp, bet):  # rolling moment due to ailerons
        A = self.polycoeffs["DLDA"]

        calp = self.coords["alp"]
        cbet = self.coords["bet2"]
        f = interpolate.interp2d(calp, cbet, A)
        return f(alp, bet)

    def DLDR(self, alp, bet):  # rolling moment due to rudder
        A = self.polycoeffs["DLDR"]

        calp = self.coords["alp"]
        cbet = self.coords["bet2"]
        f = interpolate.interp2d(calp, cbet, A)
        return f(alp, bet)

    def DNDA(self, alp, bet):  # yawing moment due to ailerons
        A = self.polycoeffs["DNDA"]

        calp = self.coords["alp"]
        cbet = self.coords["bet2"]
        f = interpolate.interp2d(calp, cbet, A)
        return f(alp, bet)

    def DNDR(self, alp, bet):  # yawing moment due to rudder
        A = self.polycoeffs["DNDR"]

        calp = self.coords["alp"]
        cbet = self.coords["bet2"]
        f = interpolate.interp2d(calp, cbet, A)
        return f(alp, bet)

    def deriv(self, long, euler, omega, pos, POW, u):
        # x = [VT, alp, bet, phi, theta, psi, p, q, r, x, y, h, POW
        # u = [delt, dele, dela, delr]
        g, mass, S, cbar, b = self.g, self.mass, self.S, self.cbar, self.b

        # Assign state, control variables
        VT, alp, bet = long
        _alp, _bet = np.rad2deg(long[1:])
        phi, theta, psi = euler
        p, q, r = omega
        alt = pos[2]
        delt, dele, dela, delr = u

        # air data and engine model
        rho, Mach = ADC(alt, VT)
        CPOW = self.TGEAR(delt)  # throttle gearing
        dPOW = self.PDOT(POW, CPOW)
        T = self.THRUST(POW, alt, Mach)

        # look-up table and component buildup
        CXT = self.CX(_alp, dele)
        CYT = self.CY(_bet, dela, delr)
        CZT = self.CZ(_alp, _bet, dele)
        CLT = self.CL(_alp, _bet) + self.DLDA(_alp, _bet)*dela/20.\
            + self.DLDR(_alp, _bet)*delr/30.
        CMT = self.CM(_alp, dele)
        CNT = self.CN(_alp, _bet) + self.DNDA(_alp, _bet)*dela/20.\
            + self.DNDR(_alp, _bet)*delr/30.

        # damping derivatives
        x_cgr = self.x_cgr
        x_cg = 0.4
        D1, D2, D3, D4, D5, D6, D7, D8, D9 = self.damp(_alp)
        CQ = cbar * q * .5 / VT
        B2V = b * .5 / VT
        CXT = CXT + CQ * D1
        CYT = CYT + B2V * (D2*r + D3*p)
        CZT = CZT + CQ * D4
        CLT = CLT + B2V * (D5*r + D6*p)
        CMT = CMT + CQ * D7 + CZT * (x_cgr - x_cg)
        CNT = CNT + B2V * (D8*r + D9*p) - CYT * (x_cgr - x_cg) * cbar / b

        # speed in earth frame
        U = VT * cos(alp) * cos(bet)
        V = VT * sin(bet)
        W = VT * sin(alp) * cos(bet)

        # Force equation
        qbar = .5 * rho * VT**2
        dU = r*V - q*W - g*sin(theta) + (qbar*S*CXT + T)/mass
        dV = p*W - r*U + g*cos(theta)*sin(phi) + qbar*S*CYT/mass
        dW = q*U - p*V + g*cos(theta)*cos(phi) + qbar*S*CZT/mass
        dVT = (U*dU + V*dV + W*dW) / VT
        dalp = (U*dW - W*dU) / (U**2 + W**2)
        dbet = (VT*dV - V*dVT) * cos(bet) / (U**2 + W**2)

        # kinematic equation
        dphi = p + sin(theta) / cos(theta) * (q*sin(phi) + r*cos(phi))
        dtheta = q*cos(phi) - r*sin(phi)
        dpsi = (q*sin(phi) + r*cos(phi)) / cos(theta)

        # moments equation
        Jxx, Jyy, Jzz, Jxz = self.Jxx, self.Jyy, self.Jzz, self.Jxz
        hx = self.hx
        Xpq, Xqr, Ypr, Zpq, gam = self.Xpq, self.Xqr, self.Ypr, self.Zpq, self.gam
        roll = qbar * S * b * CLT
        pitch = qbar * S * cbar * CMT
        yaw = qbar * S * b * CNT
        dp = (Xpq*p*q - Xqr*q*r + Jzz*roll + Jxz*(yaw + q*hx)) / gam
        dq = (Ypr*p*r - Jxz*(p**2 - r**2) + pitch - r*hx) / Jyy
        dr = (Zpq*p*q - Xpq*q*r + Jxz*roll + Jxx*(yaw + q*hx)) / gam

        # navigation equation
        dpn = U*cos(theta)*cos(psi)\
            + V*(sin(phi)*cos(psi)*sin(theta) - cos(phi)*sin(psi))\
            + W*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
        dpe = U*cos(theta)*sin(psi)\
            + V*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))\
            + W*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
        dpd = U*sin(theta) - V*sin(phi)*cos(theta) - W*cos(phi)*cos(theta)

        dlong = np.vstack((dVT, dalp, dbet))
        deuler = np.vstack((dphi, dtheta, dpsi))
        domega = np.vstack((dp, dq, dr))
        dpos = np.vstack((dpn, dpe, dpd))

        return dlong, deuler, domega, dpos, dPOW

    def set_dot(self, t, u):
        states = self.observe_list()
        dots = self.deriv(*states, u)
        self.long.dot, self.euler.dot, self.omega.dot, self.pos.dot, self.POW.dot = dots

    def aerodyn(self, long, euler, omega, pos, POW, u):
        delt, dele, dela, delr = u
        S, cbar, b = self.S, self.cbar, self.b

        VT, alp, bet = long
        _alp = np.rad2deg(long[1])
        p, q, r = omega
        qbar = 0.5 * get_rho(-pos[2]) * VT**2

        # look-up table and component buildup
        CXT = self.CX(alp, dele)
        CYT = self.CY(bet, dela, delr)
        CZT = self.CZ(alp, bet, dele)
        CLT = self.CL(alp, bet) + self.DLDA(alp, bet)*dela/20.\
            + self.DLDR(alp, bet)*delr/30.
        CMT = self.CM(alp, dele)
        CNT = self.CN(alp, bet) + self.DNDA(alp, bet)*dela/20.\
            + self.DNDR(alp, bet)*delr/30.

        # damping derivatives
        x_cgr = self.x_cgr
        x_cg = 0.4
        D1, D2, D3, D4, D5, D6, D7, D8, D9 = self.damp(_alp)
        CQ = .5 * cbar * q / VT
        B2V = .5 * b / VT
        CXT = CXT + CQ * D1
        CYT = CYT + B2V * (D2*r + D3*p)
        CZT = CZT + CQ * D4
        CLT = CLT + B2V * (D5*r + D6*p)
        CMT = CMT + CQ * D7 + CZT * (x_cgr - x_cg)

        X = qbar*CXT*S  # aerodynamic force along body x-axis
        Y = qbar*CYT*S  # aerodynamic force along body y-axis
        Z = qbar*CZT*S  # aerodynamic force along body z-axis

        l = qbar * S * b * CLT
        m = qbar * S * cbar * CMT
        n = qbar * S * b * CNT

        F_AT = np.vstack([X, Y, Z])  # aerodynamic & thrust force [N]
        M_AT = np.vstack([l, m, n])  # aerodynamic & thrust moment [N*m]

        # gravity force
        F_G = angle2dcm(euler).dot(np.array([0, 0, self.mass*self.g])).reshape(3, 1)

        F = F_AT + F_G
        M = M_AT

        return F, M

    '''
    너무 하기 싫자나ㅏ~~~
    '''
    def get_trim(self, z0={"delt": 8.35e-1, "dele": -1.48,
                           "alp": np.deg2rad(1.37e+1), "dela": 9.54e-2,
                           "delr": -4.11e-1, "bet": np.deg2rad(2.92e-2)},
                 fixed={"VT": 502, "psi": 0., "pn": 0., "pe": 0., "h": 0.},
                 method="SLSQP", options={"disp": True, "ftol": 1e-10}):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            self.control_limits["delt"],
            self.control_limits["dele"],
            (self.coords["alp"].min(), self.coords["alp"].max()),
            self.control_limits["dela"],
            self.control_limits["delr"],
            (self.coords["bet2"].min(), self.coords["bet2"].max())
        )
        result = scipy.optimize.minimize(
            self._trim_cost, z0, args=(fixed,),
            bounds=bounds, method=method, options=options)

        return self.constraint(result.x, fixed)

    def _trim_cost(self, z, fixed):
        x, u = self.constraint(z, fixed)

        self.set_dot(x, u)
        dVT, dalp, dbet = self.long.dot
        dp, dq, dr = self.omega.dot

        # dxs = np.vstack((dVT, dalp, dbet, dp, dq, dr))
        # weight = np.diag([2, 100, 100, 10, 10, 10])
        # return np.transpose(dxs).dot(weight).dot(dxs)
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


if __name__ == "__main__":
    # test
    # long = np.vstack((500., 0.5, -0.2))
    # euler = np.vstack((-1, 1, -1))
    # omega = np.vstack((0.7, -0.8, 0.9))
    # pos = np.vstack((1000, 900, 10000))
    # POW = 90
    # u = np.vstack((0.9, 20, -15, -20))
    # trim
    # long = np.vstack((502., 0.03691, -4.0e-9))
    # euler = np.vstack((0., 0.03691, 0))
    # omega = np.vstack((0, 0, 0))
    # pos = np.zeros((3, 1))
    # POW = 6.412363e+1
    # u = np.vstack((8.349601e-1, -1.481766, 9.553108e-2, -4.118124e-1))
    long = np.vstack((502., 2.39110108e-1, np.deg2rad(2.92e-2)))
    euler = np.vstack((0., 2.39110108e-1, 0.))
    omega = np.vstack((0., 0., 0.))
    pos = np.vstack((0., 0., 0.))
    # POW = 6.41323000e+1
    POW = 6.41323e+1
    u = np.vstack((0.835, 0.43633231, -0.37524579, 0.52359878))
    system = F16(long, euler, omega, pos, POW)
    system.set_dot(t=0, u=u)
    print(repr(system))
    print(system.get_trim())
