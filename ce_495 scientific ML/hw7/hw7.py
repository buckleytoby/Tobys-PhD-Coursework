

import numpy as np
import torch as th
from torch import nn
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import tqdm

# globals
L = 1.0
M = 1.0
G = 9.81
TEST = False
PLOT = True
SAVE_PLOTS_DONT_SHOW = True

def pendulum_potential(theta):
    return M * G * L * (1 - np.cos(theta))

def pendulum_closed_form_soln(t, theta0, omega0):
    alpha = np.sqrt(G/L)
    th = theta0 * np.cos(alpha * t) + omega0 * np.sin(alpha * t)
    omega = theta0 * -np.sin(alpha * t) + omega0 * np.cos(alpha * t)
    return th, omega

def pendulum_ode(t, y):
    th = y[0]
    pth = y[1]
    ppth = -G / L * np.sin(th)
    
    py = np.zeros_like(y)
    py[0] = pth
    py[1] = ppth
    return py

class Pendulum:
    def __init__(self) -> None:
        pass

    def generate(self):
        theta0_range = [np.deg2rad(-80), np.deg2rad(80)] # rad
        omega0_range = [np.deg2rad(-100), np.deg2rad(100)] # rad/s

        theta0 = np.random.uniform(theta0_range[0], theta0_range[1])
        omega0 = np.random.uniform(omega0_range[0], omega0_range[1])

        t = np.linspace(0, 10, 1000)

        th, omega = pendulum_closed_form_soln(t, theta0, omega0)

        # plot
        plt.figure()
        plt.plot(t, th)
        plt.title(f"θ(t) for initial conditions θ0={theta0:.2f} rad, ω0={omega0:.2f} rad/s")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(th, omega)
        plt.title(f"Phase portrait for initial conditions θ0={theta0:.2f} rad, ω0={omega0:.2f} rad/s")
        plt.xlabel("Angle (rad)")
        plt.ylabel("Angular velocity (rad/s)")
        plt.grid()
        plt.show()

        pass



class Problem2:
    def __init__(self) -> None:
        print("Problem 2")
        self.parta()
        self.partb()

    def parta(self):
        print("Part a")
        mint = 0
        maxt = 10
        n = 1200
        nb_trajs = 100
        theta0_range = [np.deg2rad(-80), np.deg2rad(80)] # rad
        omega0_range = [np.deg2rad(-100), np.deg2rad(100)] # rad/s

        ts = np.linspace(mint, maxt, n)

        trajs = {}

        if TEST:
            nb_trajs = 5

        for i in range(nb_trajs):
            theta0 = np.random.uniform(theta0_range[0], theta0_range[1])
            omega0 = np.random.uniform(omega0_range[0], omega0_range[1])

            # solve using rk45
            sol = integrate.solve_ivp(fun=pendulum_ode, t_span=(mint, maxt), y0=[theta0, omega0], t_eval=ts)

            key = (theta0, omega0)
            trajs[key] = sol.y

        # Plot the time evolution θ(t) and the phase portrait (θ, ω) for 5 representative trajectories
        keys = list(trajs.keys())
        choices = np.random.choice(len(keys), size=5, replace=False)
        for idx in choices:
            key = keys[idx]
            traj = trajs[key]

            theta_traj = traj[0]
            omega_traj = traj[1]

            if PLOT:
                # Plot θ(t)
                plt.figure()
                plt.plot(ts, theta_traj)
                plt.title(f"θ(t) for initial conditions θ0={key[0]:.2f} rad, ω0={key[1]:.2f} rad/s")
                plt.xlabel("Time (s)")
                plt.ylabel("Angle (rad)")
                plt.grid()
                if SAVE_PLOTS_DONT_SHOW:
                    plt.savefig(f"theta_t_{idx}.png")
                else:
                    plt.show()

                # Plot phase portrait (θ, ω)
                plt.figure()
                plt.plot(theta_traj, omega_traj)
                plt.title(f"Phase portrait for initial conditions θ0={key[0]:.2f} rad, ω0={key[1]:.2f} rad/s")
                plt.xlabel("Angle (rad)")
                plt.ylabel("Angular velocity (rad/s)")
                plt.grid()
                if SAVE_PLOTS_DONT_SHOW:
                    plt.savefig(f"phase_portrait_{idx}.png")
                else:
                    plt.show()

                # # for fun, plot (x, y)
                # x_traj = L * np.sin(theta_traj)
                # y_traj = -L * np.cos(theta_traj)
                # plt.figure()
                # plt.plot(x_traj, y_traj)
                # plt.title(f"Trajectory in Cartesian coordinates for initial conditions θ0={key[0]:.2f} rad, ω0={key[1]:.2f} rad/s")
                # plt.xlabel("x (m)")
                # plt.ylabel("y (m)")
                # plt.grid()
                # plt.show()

        # save necessary info
        self.trajs = trajs
        self.ts = ts

    def partb(self):
        print("Part b")
        # compute w and a using finite differences
        trajs = self.trajs
        ts = self.ts

        ws = {}
        accs = {}

        # for each traj
        for key in trajs:
            traj = trajs[key]
            theta_traj = traj[0]
            omega_traj = traj[1]

            # finite diff
            acc = np.diff(omega_traj) / np.diff(ts)

            # repeat the last value to make it the same length as ts
            acc = np.append(acc, acc[-1])

            ws[key] = omega_traj
            accs[key] = acc

        self.ws = ws
        self.accs = accs

        # split into training, val, test 70%, 15%, 15%
        train_percent = 0.7
        val_percent = 0.15
        # test_percent = 0.15

        keys = list(trajs.keys())

        train_choice = np.random.choice(len(keys), size=int(len(keys)*train_percent), replace=False)

        val_choice = np.random.choice(list(set(range(len(keys))) - set(train_choice)), size=int(len(keys)*val_percent), replace=False)

        test_choice = list(set(range(len(keys))) - set(train_choice) - set(val_choice))

        self.train_choice = train_choice
        self.val_choice = val_choice
        self.test_choice = test_choice

        # plot a sample
        idx = train_choice[0]
        key = keys[idx]
        traj = trajs[key]

        theta_traj = traj[0]
        w_traj = ws[key]
        acc_traj = accs[key]

        if PLOT:
            plt.figure()
            plt.plot(ts, theta_traj, label="θ(t)")
            plt.plot(ts, w_traj, label="ω(t)")
            plt.plot(ts, acc_traj, label="α(t)")
            plt.title(f"θ(t), ω(t), α(t) for initial conditions θ0={key[0]:.2f} rad, ω0={key[1]:.2f} rad/s")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
            if SAVE_PLOTS_DONT_SHOW:
                plt.savefig(f"theta_omega_alpha_{idx}.png")
            else:
                plt.show()

class Problem3:
    def __init__(self, p2) -> None:
        self.p2 = p2
        print("Problem 3")
        trajs = p2.trajs
        ws = p2.ws
        accs = p2.accs
        train_choice = p2.train_choice
        val_choice = p2.val_choice
        test_choice = p2.test_choice

        keys = list(trajs.keys())
        train_keys = [keys[idx] for idx in train_choice]
        val_keys = [keys[idx] for idx in val_choice]
        test_keys = [keys[idx] for idx in test_choice]


        train_traj = np.array([trajs[key] for key in train_keys])
        train_th = train_traj[:, 0, :]
        train_acc = np.array([accs[key] for key in train_keys])

        # reshape
        train_th = train_th.reshape(-1, 1)
        train_acc = train_acc.reshape(-1, 1)

        # convert to torch
        train_th = th.from_numpy(train_th).float()
        train_acc = th.from_numpy(train_acc).float()

        # now for val
        if len(val_keys) > 0:
            val_traj = np.array([trajs[key] for key in val_keys])
            val_th = val_traj[:, 0, :]
            val_acc = np.array([accs[key] for key in val_keys])

            val_th = val_th.reshape(-1, 1)
            val_acc = val_acc.reshape(-1, 1)

            val_th = th.from_numpy(val_th).float()
            val_acc = th.from_numpy(val_acc).float()
        else:
            val_th = None
            val_acc = None

        # now for test
        if len(test_keys) > 0:
            test_traj = np.array([trajs[key] for key in test_keys])
            test_th = test_traj[:, 0, :]
            test_acc = np.array([accs[key] for key in test_keys])

            test_th = test_th.reshape(-1, 1)
            test_acc = test_acc.reshape(-1, 1)

            test_th = th.from_numpy(test_th).float()
            test_acc = th.from_numpy(test_acc).float()
        else:
            test_th = None
            test_acc = None



        # setup model, input: theta [1], output Vw [1]
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.model = model

        # setup optim
        optim = th.optim.Adam(model.parameters(), lr=1e-3)

        # setup the loss
        loss_fn = nn.MSELoss()

        # simple training loop, use tqdm
        n = 200

        if TEST:
            n = 10

        train_losses = []
        val_losses = [] 
        with tqdm.tqdm(total=n) as pbar:
            for i in range(n):
                # zero the optim
                optim.zero_grad()

                train_th.requires_grad_(True)

                # forward -- prediction = Vw
                Vw_pred = model(train_th)

                # autograd on Vw to get ppth_pred, ppth = -dVw/dth
                acc_pred = th.autograd.grad(-1.0 * Vw_pred.sum(), train_th, create_graph=True)[0]

                # compute loss
                loss = loss_fn(acc_pred, train_acc)
                train_losses.append(loss.detach().numpy())

                # backprop
                loss.backward()

                # update
                optim.step()

                # update pbar
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")

                # with th.no_grad(): nvm, need grad for autograd
                ## val
                if len(val_keys) > 0:
                    assert(val_th is not None and val_acc is not None)
                    optim.zero_grad()

                    val_th.requires_grad_(True)

                    Vw_val = model(val_th)

                    acc_val_pred = th.autograd.grad(-1.0 * Vw_val.sum(), val_th, create_graph=True)[0]

                    val_loss = loss_fn(acc_val_pred, val_acc)
                    val_losses.append(val_loss.detach().numpy())

        ## plotting
        # plot train and val loss curves
        if PLOT:
            plt.figure()
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.title("Training and Validation Loss Curves")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend()
            if SAVE_PLOTS_DONT_SHOW:
                plt.savefig("loss_curves.png")
            else:
                plt.show()

        # Comparison of true vs. predicted θ(t) for selected trajectories from the test set
        self.compare_traj()
        self.plot_compare()

        # Phase portrait comparison (θ, ω) on the same test trajectories.

        # Comparison between the learned potential VW (θ) and the true potential V (θ) 
        self.compare_potential()

    def inference(self, theta0, omega0, ts):
        theta0 = th.tensor([[theta0]], requires_grad=True)

        theta = theta0
        omega = omega0

        thetas = []

        for i in range(len(ts)-1):
            Vw = self.model(theta)
            acc_pred = th.autograd.grad(-1.0 * Vw.sum(), theta, create_graph=True)[0]
            acc_pred = acc_pred.detach().numpy()[0, 0]

            # use simple first order integration
            dt = ts[i+1] - ts[i]
            omega = omega + acc_pred * dt
            theta = theta + omega * dt

            thetas.append(theta.detach().numpy()[0, 0])

        return thetas


    def plot_compare(self):
        # use the model, with a theta0, omega0, to predict the trajectory, and compare to the true trajectory
        # get a traj from the test set
        keys = list(self.p2.trajs.keys())
        test_keys = [keys[idx] for idx in self.p2.test_choice]
        
        for key in test_keys[0:5]:
            theta0, omega0 = key
            true_trajectory = self.p2.trajs[key]
            ts = self.p2.ts

            # get the th's
            true_th = true_trajectory[0]

            pred_th = self.inference(theta0, omega0, ts)

            # plot
            if PLOT:
                plt.figure()
                plt.plot(ts, true_th, label="True θ(t)")
                plt.plot(ts[:-1], pred_th, label="Predicted θ(t)")
                plt.title(f"True vs. Predicted θ(t) for initial conditions θ0={theta0:.2f} rad, ω0={omega0:.2f} rad/s")
                plt.xlabel("Time (s)")
                plt.ylabel("Angle (rad)")
                plt.legend()
                plt.grid()
                if SAVE_PLOTS_DONT_SHOW:
                    plt.savefig(f"theta_t_compare_{key}.png")
                else:
                    plt.show()

                # plot phase portrait
                true_omega = self.p2.ws[key]
                pred_omega = np.diff(pred_th) / np.diff(ts[:-1])
                plt.figure()
                plt.plot(true_th, true_omega, label="True Phase Portrait")
                plt.plot(pred_th[:-1], pred_omega, label="Predicted Phase Portrait")
                plt.title(f"True vs. Predicted Phase Portrait for initial conditions θ0={theta0:.2f} rad, ω0={omega0:.2f} rad/s")
                plt.xlabel("Angle (rad)")
                plt.ylabel("Angular velocity (rad/s)")
                plt.legend()
                plt.grid()
                if SAVE_PLOTS_DONT_SHOW:
                    plt.savefig(f"phase_portrait_compare_{key}.png")
                else:
                    plt.show()


                # get the th's
                true_th = true_trajectory[0]

                # get the true potential
                true_V = pendulum_potential(true_th)

                # get the predicted potential
                true_th_tensor = th.from_numpy(true_th).float().unsqueeze(1)
                pred_V = self.model(true_th_tensor).detach().numpy().squeeze()

                # plot it
                plt.figure()
                plt.plot(true_th, true_V, label="True Potential V(θ)")
                plt.plot(true_th, pred_V, label="Learned Potential VW(θ)")
                plt.title(f"True vs. Learned Potential for initial conditions θ0={theta0:.2f} rad, ω0={omega0:.2f} rad/s")
                plt.xlabel("Angle (rad)")
                plt.ylabel("Potential Energy (J)")
                plt.legend()
                plt.grid()
                if SAVE_PLOTS_DONT_SHOW:
                    plt.savefig(f"potential_compare_{key}.png")
                else:
                    plt.show()

    def compare_traj(self):
        keys = list(self.p2.trajs.keys())
        test_keys = [keys[idx] for idx in self.p2.test_choice]

        total_mse = 0.0
        
        for key in test_keys:
            theta0, omega0 = key
            true_trajectory = self.p2.trajs[key]
            ts = self.p2.ts

            # get the th's
            true_th = true_trajectory[0]

            pred_th = self.inference(theta0, omega0, ts)

            # calc the MSE
            mse = np.mean((true_th[:-1] - pred_th)**2)
            total_mse += mse

        # average the total_mse
        avg_mse = total_mse / len(test_keys)
        print(f"Average MSE on test trajectories: {avg_mse:.4f}")

    def compare_potential(self):
        keys = list(self.p2.trajs.keys())
        test_keys = [keys[idx] for idx in self.p2.test_choice]

        total_mse = 0.0

        for key in test_keys:
            theta0, omega0 = key
            true_trajectory = self.p2.trajs[key]
            ts = self.p2.ts

            # get the th's
            true_th = true_trajectory[0]

            # get the true potential
            true_V = pendulum_potential(true_th)

            # get the predicted potential
            true_th_tensor = th.from_numpy(true_th).float().unsqueeze(1)
            pred_V = self.model(true_th_tensor).detach().numpy().squeeze()

            # calc the MSE
            mse = np.mean((true_V - pred_V)**2)
            total_mse += mse

        # average the total_mse
        avg_mse = total_mse / len(test_keys)
        print(f"Average MSE on potential comparison: {avg_mse:.4f}")




            
        


if __name__ == "__main__":
    pendulum = Pendulum()
    pendulum.generate()
    p2 = Problem2()
    p3 = Problem3(p2)