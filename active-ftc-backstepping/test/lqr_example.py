import numpy as np
import fym.core import BaseEnv, BaseSystem
import fym.logging
import matplotlib.pyplot as plt

from fym.agents import LQR

class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt='''time step''' , max_t ='''finishing time''' )
        # define system
        self.x = BaseSystem('''initial condition''')
        self.A = 
        self.B = 
        self.C = 
        self.D = 

        # define LQR
        Q = np.eye()
        R = np.eye()
        self.K , *_ = LQR_clqr(self.A, self.B, Q, R)

    # reset state's initial value
    def reset(self):
         super().reset()
    
    # define state equation
    def set_dot(self, t):
        x = self.x.state
        u = -self.K.dot(x)
        self.x.dot = self.A.dot(x) + self.B.dot(u)

    # define time step and run, updates system's state, and return them
    def step(self):
        t = self.clock.get() # define variables so that we can use it in iteration
        x = self.x.state
        u = -self.K.dot(x)
        y = self.C.dot(x) + self.D.dot(u)

        *_, done = self.update() # run and update
        return t, x, u, y, done

# system simulation
def run():
    env = Env()
    x = env.reset()
    env.logger = sym.logging.Logger(path='data.h5') # save Logger in env

    while True:
        env.render() # show process running
        t, x, u, y, done = env.step()
        env.logger.record(t=t, x=x, u=u, y=y)

        if done:
            break

    env.close()

def plot_var():
    data = fym.logging.load('data.h5')
    plt.plot(data['t'], data['x'].squeeze())
    plt.xlabel('t [sec]')
    plt.ylabel('')
    ply.title('LQR simulation')
    plt.grid(True)
    plt.show()

def plot_y():
    data = fym.logging.load('data.h5')
    plt.plot(data['t'], data['y'].squeeze())
    plt.xlabel('t [sec]')
    plt.ylabel('')
    plt.title('LQR simulation')
    plt.grid(True)
    plt.show()

run()
plot_var()
plot_y()

