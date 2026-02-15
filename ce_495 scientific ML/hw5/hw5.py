

import numpy as np
from copy import deepcopy
from collections import defaultdict
from scipy.integrate import RK45

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import figure

import torch as th
from torch import nn

# global parameters
DT = 1e-1
K = 100.0
B = 10.0
M = 5.0
COUNT = 1

T0 = 0.0
TF = 5.0

def time_plot(t, z, v):
    plt.plot(t, z)
    plt.plot(t, v)

def phase_plot(z, v):
    plt.plot(z, v)

class DampedHarmonicOscillator:
    def __init__(self, k=K, b=B, m=M):
        self.k = k
        self.b = b
        self.m = m

        # state
        self.h = np.zeros((2, 1))

        # SoE matrix
        self.A = np.array([[0, 1],
                           [-self.k / self.m, -self.b / self.m]])
        
        self.set_initial_conditions()
        
    def set_initial_conditions(self):
        """
        seems like we just need one, so might as well make it interesting
        """
        z = 0.5 # m
        v = 0.1 # m/s

        self.h0 = np.array([[z, v]]).T

        self.h[0:2, :] = self.h0


    def compute_h_dot(self, t, h):
        h = h.reshape((2, 1))
        assert(h.shape == (2, 1))
        assert(self.A.shape == (2, 2))

        h_dot = self.A @ h

        return h_dot.squeeze()

class ForwardEuler:
    def __init__(self,
                 xprimefcn,
                 ) -> None:
        self.xprimefcn = xprimefcn
        

    def step(self, xn, xprimen, t, dt):
        # x_n+1 = x_n + dt * x'_n

        xnplusone = xn + dt * xprimen

        return xnplusone
    
    def get_xprime(self, t, x):
        xprime = self.xprimefcn(t, x)
        return xprime
    
    def steps(self, t0, tf, x0, dt):
        # multiple steps
        t0 = t0
        tf = tf

        nb = int((tf - t0) / dt)

        ts = np.linspace(t0, tf, nb)

        x = x0
        history = defaultdict(list)

        for idx, t in enumerate(ts):
            xprime = self.get_xprime(t, x)
            xprime = xprime.reshape((2, 1))
            x = self.step(x, xprime, t, dt)

            # save histories for plotting
            history['z'].append(x[0])
            history['v'].append(x[1])
            history['t'].append(t)

        self.history = history
        return self.history

    def plot(self, fig: figure.Figure = None): #type:ignore
        z = self.history['z']
        v = self.history['v']
        t = self.history['t']

        time_plot(t, z, v)


class RK4(ForwardEuler):
    def __init__(self, xprimefcn) -> None:
        super().__init__(xprimefcn)

        self.rk45 = None
        self.xprimefcn

    def steps(self, t0, tf, x0, dt):
        self.rk45 = RK45(
            fun = self.xprimefcn,
            t0 = t0,
            y0 = x0.squeeze(),
            t_bound = tf,
            first_step = dt,
            max_step = dt,
        )

        return super().steps(t0, tf, x0, dt)

    def step(self, xn, xprimen, t, dt):
        # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html#scipy.integrate.RK45
        assert(isinstance(self.rk45, RK45))

        t = self.rk45.t

        tf = t + dt

        while t < tf:
            self.rk45.step()
            t = self.rk45.t

        xnplusone = self.rk45.y

        xnplusone = xnplusone.reshape((2, 1))

        return xnplusone


class Exercise2:
    """
    Solve your chosen system numerically using the classical Runge–Kutta method of order 4 (RK4) with a fixed time step. Define clearly the time interval, the step size, and the initial condition. Make sure your implementation is correct and that the trajectory behaves as expected for the chosen system.
    """
    def __init__(self) -> None:
        self.system = DampedHarmonicOscillator()

        
        self.solve()

        self.plot()

    def solve(self):
        """
        solve RK4
        """
        self.solver = RK4(self.system.compute_h_dot)
        self.history = self.solver.steps(T0, TF, self.system.h0, DT)

        # convert to numpy
        self.history = {
            key: np.array(self.history[key]) for key in self.history
        }

    def get_history(self):
        return self.history
    
    def plot(self):
        self.solver.plot()

        plt.title("Exercise 2")
        plt.legend(["Pos", "Vel"])
        plt.show()
        
class Exercise3:
    """
    Add additive Gaussian noise to the trajectory you obtained. The noise should be independent and centered, with a chosen standard deviation. Produce again two subplots: 1) show the clean solution z(t) as a line together with the noisy sampled points $(t_i, z_i)$, and 2) show the smooth trajectory in the two-dimensional phase space (z(t), v(t)) and superimpose the noisy points $(z_i, v_i)$. The goal is to clearly visualize the difference between the underlying smooth dynamics and the corrupted observations.
    """
    def __init__(self) -> None:
        system = DampedHarmonicOscillator()

        self.exercise2 = Exercise2()

        self.clean_history = self.exercise2.get_history()

        self.add_noise()

        self.plot()

    def add_noise(self):
        """
        additive Gaussian noise to the trajectory you obtained. The noise should be independent and centered, with a chosen standard deviation
        """
        # std dev for z, v
        std_dev = [0.05, 0.05]

        # make the gaussian noise generator
        def gaus(n, std_dev):
            noise = np.random.normal(0, std_dev, (n, 1))
            return noise

        N = len(self.clean_history['t'])

        z = deepcopy(self.clean_history['z'])
        v = deepcopy(self.clean_history['v'])

        # generate noise
        noise1 = gaus(N, std_dev[0])
        noise2 = gaus(N, std_dev[1])
    
        # corrupt the state
        noisy_z = z + noise1
        noisy_v = v + noise2

        # make a new history
        self.history = deepcopy(self.clean_history)

        # replace z, v
        self.history['z'] = noisy_z #type:ignore
        self.history['v'] = noisy_v #type:ignore
        pass

    def get_noisy_history(self):
        return self.history
    
    def plot(self):
        """
        Produce again two subplots: 1) show the clean solution z(t) as a line together with the noisy sampled points $(t_i, z_i)$, and 2) show the smooth trajectory in the two-dimensional phase space (z(t), v(t)) and superimpose the noisy points $(z_i, v_i)$.
        """
        ## 1)
        # clean trajectory
        # self.exercise2.plot()

        # dirty trajectory
        t = self.history['t']
        dirty_z = self.history['z']
        dirty_v = self.history['v']

        time_plot(t, self.clean_history['z'], self.clean_history['v'])
        time_plot(t, dirty_z, dirty_v)

        plt.title("Exercise 3, plot 1, time plot")
        plt.legend(["Pos", "Vel", "Noisy Pos", "Noisy Vel"])
        plt.show()

        ## 2)
        phase_plot(self.clean_history['z'], self.clean_history['v'])
        phase_plot(dirty_z, dirty_v)

        plt.title("Exercise 3, plot 2, phase plot")
        plt.legend(["Clean Phase", "Dirty Phase"])
        plt.show()



class Exercise4:
    """
    Using PyTorch, train a feedforward neural network to solve a least-squares regression problem where the input is time $t_i$ and the target is the noisy observation $z_i$. The objective is to learn a smooth approximation of the trajectory from noisy data. Start with a relatively small network (e.g. depth of 2 hidden layers and moderate width of 32 neurons).
    """
    def __init__(self) -> None:
        # make the system
        system = DampedHarmonicOscillator()
        self.exercise3 = Exercise3()

        self.history = self.exercise3.get_noisy_history()

        # time ti
        in_features = 1

        # x-value, xi = zi
        out_features = 2

        # make the model
        self.model = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, out_features),
        )

        self.train()

        self.plot()

    def train(self):
        """
        a simple pytorch training loop
        """
        opt = th.optim.Adam(self.model.parameters(), lr = 1e-3)

        # loop vars
        done = False

        # target is fixed
        target = np.hstack([self.history['z'], self.history['v']]) #type:ignore
        target = th.Tensor(target)
        target = target.reshape((-1, 2))

        # obs is fixed
        obs = self.history['t']
        obs = th.Tensor(obs)
        obs = obs.reshape((-1, 1))

        # while not done:
        for _ in range(COUNT):
            # reset optimizer
            opt.zero_grad()

            # feed it forward
            zi = self.model(obs)

            # MSE loss
            loss = nn.functional.mse_loss(zi, target)

            # back-propagate
            loss.backward()

            # step the optimizer
            opt.step()

        print("Final loss: {}".format(loss.detach().cpu().numpy())) #type:ignore

    def final_prediction(self):
        t = self.exercise3.clean_history['t']
        t = th.Tensor(t)
        t = t.reshape((-1, 1))

        # final predictions
        with th.no_grad():
            xi = self.model(t)

        # numpy
        xi = xi.cpu().numpy()
        zi = xi[:, 0]
        vi = xi[:, 1]

        return t, zi, vi

    def plot(self):
        """
        print the get the final predictions, and plot them
        """        
        t, zi, vi = self.final_prediction()

        # time plot vs obs
        # time_plot(t, clean_history['z'], clean_history['v'])
        time_plot(t, self.history['z'], self.history['v'])
        time_plot(t, zi, vi)

        plt.title("Exercise 4: time plot")
        plt.legend(["Noisy Pos", "Noisy Vel", "NN Pos", "NN Vel"])

        plt.show()

        # phase plot
        phase_plot(self.history['z'], self.history['v'])
        phase_plot(zi, vi)

        plt.title("Exercise 4: phase plot")
        plt.legend(["Noisy Phase", "NN Phase"])
        plt.show()


class Exercise5:
    """
    Increase your model capacity in two different ways: first by increasing the depth (adding layers), and second by increasing the width (adding neurons per layer). Compare the results. Explain in your own words what spectral bias means and if it appears in your experiments.
    """
    def __init__(self) -> None:
        self.exercise4 = Exercise4()

        self.plot()

    def final_prediction(self):
        # for now just call exercise4
        return self.exercise4.final_prediction()

    def plot(self):
        pass

        
class Exercise6:
    """
    Consider a ResNet-type architecture with shared weights across time steps, interpreted as a discrete dynamical system. Define a neural network $F_{theta}$ that takes as input the current state h = (z, v) in $R^2$ and outputs a vector in $R^2$. Build the predicted trajectory by iterating the Euler/ResNet update $h_{n+1} = h_n + dt * F_{theta}(h_n)$, using the same dt as in your RK4 solver. Here $F_{theta}$ is the residual block and represents the learned vector field of your ODE system in phase space. Pick the size of $F_{theta}$ (depth/width) based on what you observed in the previous capacity experiments.

    In code: define the module $F_{theta}$ for the vector field, then define a rollout function (rollout/euler) that starts from h0 and runs a loop for N steps, repeatedly applying the update and storing all states, then concatenating them into an (N,2) trajectory. Train by rolling out from the initial condition and minimizing the MSE between the full predicted trajectory and the target trajectory across all time steps. Plot z(t) (learned vs ground truth) and the phase-space trajectory (z,v) (learned vs ground truth, and optionally noisy points). Finally, compare with the previous feedforward network $z_{hat}(t)$: comment briefly on speed and on accuracy of both models.
    """
    def __init__(self) -> None:
        # my children
        self.system = DampedHarmonicOscillator()

        self.exercise5 = Exercise5()
        self.exercise4 = self.exercise5.exercise4
        self.exercise3 = self.exercise4.exercise3

        self.history = self.exercise3.get_noisy_history()

        # params
        self.N = len(self.history['t'])
        self.dt = 1e-3

        # length of the position
        len_z = 1

        # length of the velocity
        len_v = 1

        self.batch_size = 1000 # testing

        in_features = len_z + len_v

        # outputs the vector field
        out_features = 2 

        # residual network, takes as input the current state h = (z, v) in $R^2$ and outputs a vector in $R^2$
        self.F_theta = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, out_features),
        )

        # alias
        self.model = self.F_theta

        self.train()

        self.plot()

    def rollout_euler(self, h):
        """
        Build the predicted trajectory by iterating the Euler/ResNet update $h_{n+1} = h_n + dt * F_{theta}(h_n)$, using the same dt as in your RK4 solver

        starts from h0 and runs a loop for N steps, repeatedly applying the update and storing all states, then concatenating them into an (N,2) trajectory
        """

        h_n = h

        n = self.batch_size - 1

        trajectory = [h_n]

        for i in range(n):
            ftheta = self.F_theta(h_n)

            h_n_plus_one = h_n + self.dt * ftheta

            h_n = h_n_plus_one

            trajectory.append(h_n)

        # stack the trajectory
        trajectory = th.vstack(trajectory)

        return trajectory

    def train(self):
        """
        Train by rolling out from the initial condition and minimizing the MSE between the full predicted trajectory and the target trajectory across all time steps
        """
        opt = th.optim.Adam(self.model.parameters(), lr = 1e-3)

        # loop vars
        done = False

        # inital condition
        h0 = self.system.h0
        h0 = th.Tensor(h0)
        h0 = h0.reshape((1, 2))


        # target traj is static
        target = np.hstack([self.history['z'], self.history['v']]) #type:ignore
        target = th.Tensor(target)
        target = target.reshape((-1, 2))

        target = target[:self.batch_size, :]

        # while not done:
        for _ in range(COUNT):
            # reset optimizer
            opt.zero_grad()

            predicted_traj = self.rollout_euler(h0)

            # MSE loss
            loss = nn.functional.mse_loss(predicted_traj, target)

            # back-propagate
            loss.backward()

            # step the optimizer
            opt.step()

    def plot(self):
        """
        Plot z(t) (learned vs ground truth) and the phase-space trajectory (z,v) (learned vs ground truth, and optionally noisy points).

        Finally, compare with the previous feedforward network $z_{hat}(t)
        """
        clean_history = self.exercise3.clean_history
        dirty_history = self.history
        
        t = clean_history['t']
        t = th.Tensor(t)
        t = t.reshape((-1, 1))


        h0 = self.system.h0
        h0 = th.Tensor(h0)
        h0 = h0.reshape((1, 2))

        # final predictions
        with th.no_grad():
            xi = self.rollout_euler(h0)

        # numpy
        xi = xi.cpu().numpy()
        zi = xi[:, 0]
        vi = xi[:, 1]

        # time plot vs obs
        time_plot(t, clean_history['z'], clean_history['v'])
        # time_plot(t, self.history['z'], self.history['v'])
        time_plot(t, zi, vi)

        plt.title("Exercise 6: time plot")
        plt.legend(["GT Pos", "GT Vel", "RNN Pos", "RNN Vel"])

        plt.show()

        # phase plot
        # phase_plot(self.history['z'], self.history['v'])
        phase_plot(clean_history['z'], clean_history['v'])
        phase_plot(zi, vi)

        plt.title("Exercise 6: phase plot")
        plt.legend(["GT Phase", "RNN Phase"])
        plt.show()

        #####
        t, ff_zi, ff_vi = self.exercise5.final_prediction()

        # time plot vs obs
        time_plot(t, ff_zi, ff_vi)
        time_plot(t, zi, vi)

        plt.title("Exercise 6: time plot")
        plt.legend(["FF Pos", "FF Vel", "RNN Pos", "RNN Vel"])

        plt.show()

class Exercise7:
    """
    Plot the learned vector field in phase space by evaluating the neural network $F_{theta}$ on a grid of points and representing the corresponding vectors with arrows (quiver plot). 
    
    Instead of using only one initial condition, define a two-dimensional Gaussian distribution centered at your original initial state $(z_0, v_0)$, with a standard deviations of your choice. 
    
    Randomly sample three different initial conditions from this Gaussian distribution and, using your trained model in prediction mode, generate and plot the three corresponding trajectories in phase space. Comment briefly on how the learned dynamics behave for nearby initial conditions.
    """
    def __init__(self) -> None:
        self.exercise6 = Exercise6()
        
        self.plot()

    def plot(self):
        """
        Plot the learned vector field in phase space by evaluating the neural network $F_{theta}$ on a grid of points and representing the corresponding vectors with arrows (quiver plot).
        """
        h0_0 = self.exercise6.system.h0

        def gaus(std_dev):
            return np.random.normal(0, std_dev, (2, 1))
        

        def run_and_plot(h):
            h = th.Tensor(h)
            h = h.reshape((1, 2))

            # final predictions
            with th.no_grad():
                xi = self.exercise6.rollout_euler(h)

            # numpy
            xi = xi.cpu().numpy()
            zi = xi[:, 0]
            vi = xi[:, 1]

            # phase plot
            # phase_plot(self.history['z'], self.history['v'])
            phase_plot(self.exercise6.exercise3.clean_history['z'], self.exercise6.exercise3.clean_history['v'])
            phase_plot(zi, vi)

            plt.title("Exercise 7: phase plot. IC: {}".format(h))
            plt.legend(["GT Phase", "RNN Phase"])
            plt.show()

        #1
        run_and_plot(h0_0.copy() + gaus(0.05))

        #2
        run_and_plot(h0_0.copy() + gaus(0.05))

        #3
        run_and_plot(h0_0.copy() + gaus(0.05))


def main():
    Exercise7()

if __name__ == "__main__":
    main()