import numpy as np
from pathlib import Path
from src.envs import HexacopterEnv, BacksteppingController, HexacopterBacksteppingControllerEnv, ActuatorFDIFirstOrderLagEnv
from fym.utils.rot import angle2dcm
import fym.logging as logging
import matplotlib.pyplot as plt


def test_command():
    t_fault = 10.0
    hexa, ctrl = make_hexa_ctrl(t_fault)
    t = 0.0
    pos = np.ones(3)
    vel = np.ones(3)
    yaw0, pitch0, roll0 = np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)
    angle0 = np.array([yaw0, pitch0, roll0])
    rot = angle2dcm(*angle0)
    omega = np.ones(3)
    xd = np.ones(3)
    vd = np.ones(3)
    ad = np.ones(3)
    ad_dot = np.ones(3)
    ad_ddot = np.ones(3)
    m = hexa.m
    J = hexa.J
    g = hexa.g
    Td = np.array([m*np.linalg.norm(g)])
    nud, Td_dot = ctrl.command(pos, vel, rot, omega,
                       xd, vd, ad, ad_dot, ad_ddot, Td,
                       m, J, g)
    u = 10000*np.array([1, 2, 3, 4, 5, 6])
    print(nud, Td_dot)
    print(hexa.B)
    derivative_hexa = hexa.dynamics(t, vel, rot, omega, u)
    print(derivative_hexa)

def sim(env, logger):
    env.reset()
    while True:
        env.render()
        next_obs, reward, info, done = env.step()
        logger.record(**info)
        if done:
            break
    env.close()
    logger.close()

def make_hexa_ctrl(t_fault):
    # env
    # pos0 = np.zeros(3)
    pos0 = np.array([0.1, 0.2, -0.1])
    vel0 = np.zeros(3)
    yaw0, pitch0, roll0 = np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)
    # yaw0, pitch0, roll0 = np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)
    angle0 = np.array([yaw0, pitch0, roll0])
    rot0 = angle2dcm(*angle0)
    # omega0 = np.ones(3)
    omega0 = np.ones(3)
    def Lambda(t):
        coeffs = np.eye(6)
        if t > t_fault:
            coeffs = np.array([[0.1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0 ,0, 1, 0, 0, 0],
                               [0 ,0, 0, 1, 0, 0],
                               [0 ,0, 0, 0, 1, 0],
                               [0 ,0, 0, 0, 0, 1],
                              ])
        return coeffs

    hexa = HexacopterEnv(pos0, vel0, rot0, omega0, dt=0.01, max_t=1.0, Lambda=Lambda)
    ctrl = BacksteppingController(pos0, hexa.m, np.linalg.norm(hexa.g), dt=0.01, max_t=20.0)
    return hexa, ctrl

def make_hexa_ctrl_fdi(t_fault):
    hexa, ctrl = make_hexa_ctrl(t_fault)
    fdi = ActuatorFDIFirstOrderLagEnv()
    return hexa, ctrl, fdi

def run_and_plot(env, dir_name, names, t_fault=None):
    # save
    dir_log = Path(dir_name)
    logger = logging.Logger(dir_log / "sample.h5")
    sim(env, logger)
    # load
    data = logging.load(logger.path)
    for name in names:
        fig, ax = plt.subplots()
        ax.set_title(f"{name} (t_fault = {t_fault} [s])")
        if len(data[name].shape) == 2:
            for i in range(data[name].shape[1]):
                ax.plot(data["t"], data[name][:, i], color=f"C{i+1}")
        else:
            ax.plot(data["t"], data[name], color="C1")
        if name == "pos":
            if len(data["xc"].shape) == 2:
                for i in range(data["xc"].shape[1]):
                    ax.plot(data["t"], data["xc"][:, i], color=f"C{i+1}", linestyle="dashed")
            else:
                ax.plot(data["t"], data["xc"], color="C1", linestyle="dashed")
        fig.savefig(dir_log / f"{name}.png", dpi=300)

def test_hexa():
    t_fault = 10.0
    hexa, _ = make_hexa_ctrl(t_fault)
    dir_name = "data/hexa"
    names = ["pos", "vel", "angle", "omega"]
    run_and_plot(hexa, dir_name, names)

def test_ctrl():
    t_fault = 10.0
    hexa, ctrl = make_hexa_ctrl(t_fault)  # dummy
    dir_name = "data/ctrl"
    names = ["xd", "vd", "ad", "ad_dot", "ad_ddot", "Td"]
    run_and_plot(ctrl, dir_name, names)

def test_FDI():
    fdi = ActuatorFDIFirstOrderLagEnv(dt=0.01, max_t=20.0)
    dir_name = "data/FDI"
    names = ["Lambda_flatten", "Lambda_hat_flatten"]
    run_and_plot(fdi, dir_name, names)

def test_hexa_ctrl_fdi():
    t_fault = 20.0
    hexa, ctrl, fdi = make_hexa_ctrl_fdi(t_fault)
    env = HexacopterBacksteppingControllerEnv(hexa, ctrl, fdi, dt=0.001, max_t=50.0)
    dir_name = "data/hexa_ctrl_fdi"
    names = ["pos", "vel", "angle", "omega",
             "ex", 
             # "xd", "vd", "ad", "ad_dot", "ad_ddot", "Td",
             "T", "M", "thrust", "u",
             "Lambda_flatten", "Lambda_hat_flatten", "Lambda_diagonal", "Lambda_hat_diagonal",
            ]
    run_and_plot(env, dir_name, names, t_fault=t_fault)


if __name__ == "__main__":
    # test_command()
    # test_hexa()
    # test_ctrl()
    # test_FDI()
    test_hexa_ctrl_fdi()
