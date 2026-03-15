

"""
all about pinns
"""

import torch as th
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from matplotlib.animation import FuncAnimation

import tqdm

PLOT = True
TEST = False
DEVICE = 'cpu'

class WaveEquation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # ORDER IS (T, X)

        nb_cells = 100
        t_nb_cells = 200

        self.v = 1.0 # testing

        self.tdomain = th.tensor([0, 5])
        self.xdomain = th.tensor([0, 1])

        # choose initial conditions
        self.ts = th.linspace(self.tdomain[0], self.tdomain[1], t_nb_cells)
        self.xs = th.linspace(self.xdomain[0], self.xdomain[1], nb_cells)

        # self.u = th.zeros((t_nb_cells, nb_cells))
        # make the parameters, random initialization
        self.u = nn.Parameter(th.rand(t_nb_cells, nb_cells))
        self.u_t = nn.Parameter(th.rand(t_nb_cells, nb_cells))
        self.u_x = nn.Parameter(th.rand(t_nb_cells, nb_cells))


        # # initial u0_t
        # u0_t = th.zeros_like(self.xs)
        # self.u0_t = u0_t.float()

        # # initial u0_x, analytical derivative of u0 w.r.t. x
        # u0_x = 2 * th.pi * th.cos(2 * th.pi * self.xs)
        # self.u0_x = u0_x.float()
        
        # move everything to device
        self.to(DEVICE)
        self.ts = self.ts.to(DEVICE)
        self.xs = self.xs.to(DEVICE)
        self.tdomain = self.tdomain.to(DEVICE)
        self.xdomain = self.xdomain.to(DEVICE)
        

    def reinforce_ics(self):
        # recall, order is (t, x)

        # set u[t=0, :] = u0
        with th.no_grad():
            # enforce the initial wave values at t=0
            self.u[0, :] = th.sin(2 * th.pi * self.xs).float()

            # (t, x=0) think of it like a boundary forcing function
            self.u[:, 0] = th.sin(2 * th.pi * self.ts).float()

            # (t, x=-1) other side forcing function so it looks nice
            self.u[:, -1] = th.sin(2 * th.pi * self.ts).float()
            # self.u[:, -1] = 0.0

    def forward(self, uxx):
        # utt = v**2 * uxx,
        utt = self.v**2 * uxx
        return utt
    
    def step_u(self):
        # use the initial conditions to step u forward in time
        # using finite differences
        self.optimizer.zero_grad()

        u = self.u
        ts = self.ts

        dt = ts[1] - ts[0]
        dx = self.xs[1] - self.xs[0]
        
        # compute u'
        u_prime_pred = th.diff(u, dim=0) / dt

        # compute the residual, detach u_t, skip the first time step since it's the initial condition
        u_t_gt = self.u_t.detach()[1:, :]
        residual = u_prime_pred - u_t_gt

        # now for x
        u_x_pred = th.diff(u, dim=1) / dx

        # ground truth u_x, skip the first spatial step since it's the initial condition
        u_x_gt = self.u_x.detach()[:, 1:] 
        residual_x = u_x_pred - u_x_gt

        # mse loss
        loss = (residual**2).mean()
        loss_x = (residual_x**2).mean()

        total_loss = loss + loss_x

        # backward
        total_loss.backward()

        # step
        self.optimizer.step()

        return loss.detach()

    def step_pde(self):
        # use the initial conditions to step u_prime forward in time
        # using finite differences
        self.optimizer_prime.zero_grad()

        ts = self.ts

        dt = ts[1] - ts[0]
        dx = self.xs[1] - self.xs[0]
        
        # compute u_tt
        u_tt = th.diff(self.u_t, dim=0) / dt

        # compute u_xx
        u_xx = th.diff(self.u_x, dim=1) / dx

        # skip the IC's
        u_tt = u_tt[:, 1:]
        u_xx = u_xx[1:, :]

        # compute the residual
        residual = u_tt - self.v**2 * u_xx

        # mse loss
        loss = (residual**2).mean()

        # backward
        loss.backward()

        # step
        self.optimizer_prime.step()

        return loss.detach()
    
    def solve(self):
        "use simple finite differences to solve it"

        self.optimizer = th.optim.Adam([self.u], lr=1e-2)
        self.optimizer_prime = th.optim.Adam([self.u_t, self.u_x], lr=1e-2)
        
        # to device
        # self.optimizer = self.optimizer.to(DEVICE)
        # self.optimizer_prime = self.optimizer_prime.to(DEVICE)

        losses = []

        nb_iters = 2500

        if TEST:
            nb_iters = 100

        with tqdm.tqdm(total=nb_iters) as pbar:
            for i in range(nb_iters):
                # step u
                loss = self.step_u()

                # step u_prime
                loss_p = self.step_pde()

                # reinforce IC's
                self.reinforce_ics()

                # for logging
                total_loss = loss + loss_p

                losses.append(total_loss.item())

                # log
                pbar.update(1)
                pbar.set_description(f"Loss: {total_loss.item():.4f}")


        self.plot(losses)
        # self.animate_solution()

    def plot(self, losses):
        if not PLOT:
            return
        
        u = self.u.detach().cpu().numpy()

        tdomain = self.tdomain.cpu().numpy()
        xdomain = self.xdomain.cpu().numpy()
        extent = (tdomain[0], tdomain[1], xdomain[0], xdomain[1])
        # extent = (xdomain[0], xdomain[1], tdomain[0], tdomain[1])

        plt.figure()
        plt.imshow(u.T, extent=extent, origin='lower', aspect='auto')
        plt.colorbar(label='u(t, x)')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Wave Equation Solution')
        plt.show()

        # plot losses
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.yscale('log')
        plt.show()
        
    def animate_solution(self, save_path=None):
        # ORDER IS (T, X)
        # Prepare data (Time x Space)
        u_data = self.u.detach().cpu().numpy()
        t_range = self.ts.cpu().numpy()
        x_range = self.xs.cpu().numpy()
        
        fig, ax = plt.subplots()
        
        # Initialize the plot with the first time step
        # We plot u[t, :] which is the 1D wave state at time t
        line, = ax.plot(x_range, u_data[0, :], lw=2)
        
        ax.set_xlim(x_range[0], x_range[-1])
        # Set y-limit based on the global min/max of the solution
        ax.set_ylim(u_data.min() * 1.1, u_data.max() * 1.1)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(t, x)')
        title = ax.set_title(f'Wave Equation Evolution (t = {t_range[0]:.2f})')

        def update(frame):
            # frame = time
            # Update the y-data of the line for the current time frame
            line.set_ydata(u_data[frame, :])
            title.set_text(f'Wave Equation Evolution (t = {t_range[frame]:.2f})')
            return line, title

        # Create animation
        # frames = number of time steps
        ani = FuncAnimation(fig, update, frames=len(t_range), 
                            interval=50, blit=True)

        if save_path:
            ani.save(save_path, writer='ffmpeg')
        
        plt.show()
    
    

class FiniteDifference:
    def __init__(self) -> None:
        pass



class PINN(nn.Module):
    def __init__(self, wave_equation: WaveEquation):
        super().__init__()
        # ORDER IS (T, X)

        self.wave_equation = wave_equation

        # extract the solution
        self.u = wave_equation.u.detach().clone()
        self.u_t = wave_equation.u_t.detach().clone()
        self.u_x = wave_equation.u_x.detach().clone()
        self.xs = wave_equation.xs.detach().clone()
        self.ts = wave_equation.ts.detach().clone()

        # totals
        self.total_data_points = self.u.numel()

        # ORDER IS (T, X)
        self.total_bc_points = self.u[:, 0].numel() + self.u[:, -1].numel() + self.u[0, :].numel()

        nb_inputs = 2
        nb_outputs = 1
        nb_hidden_features = 64
        nb_hidden_layers = 5

        first_layer = nn.Linear(nb_inputs, nb_hidden_features)

        hidden_layers = []
        for _ in range(nb_hidden_layers):
            hidden_layers.append(nn.Linear(nb_hidden_features, nb_hidden_features))
            hidden_layers.append(nn.Mish())

        final_layer = nn.Linear(nb_hidden_features, nb_outputs)

        # the model is trying to predict the state directly
        self.model = nn.Sequential(
            first_layer,
            *hidden_layers,
            final_layer
        )

        # optimizer
        self.optimizer = th.optim.Adam(self.parameters(), lr=5e-4)

        # loss
        self.loss_fn = nn.MSELoss()
        
        # move everything to device
        self.to(DEVICE)
        self.u = self.u.to(DEVICE)
        self.u_t = self.u_t.to(DEVICE)
        self.u_x = self.u_x.to(DEVICE)
        self.xs = self.xs.to(DEVICE)
        self.ts = self.ts.to(DEVICE)

    def run(self):
        nb_iters = 10000

        if TEST:
            nb_iters = 10

        losses = []

        with tqdm.tqdm(total=nb_iters) as pbar:
            for i in range(nb_iters):
                # zero grad
                self.optimizer.zero_grad()

                # compute the loss
                loss = self.compute_loss()

                # backward
                loss.backward()

                # step
                self.optimizer.step()

                losses.append(loss.detach().item())

                # update progress bar
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")

        self.losses = losses

    def get_loss_pde(self):
        # ORDER IS (T, X)
        # utt = v**2 * uxx
        # 0 = utt - v**2 * uxx

        # sample collocation points
        sample, u_gt = self.get_random_batch()

        t = sample[:, 0].unsqueeze(-1)
        x = sample[:, 1].unsqueeze(-1)

        t.requires_grad_(True)
        x.requires_grad_(True)

        # reassemble so grad knows where x, t come from
        # ORDER IS (T, X)
        sample = th.cat([t, x], dim=-1)

        # forward
        u = self.model(sample)

        # backward
        ut = th.autograd.grad(u, t, grad_outputs=th.ones_like(u), create_graph=True)[0]
        utt = th.autograd.grad(ut, t, grad_outputs=th.ones_like(ut), create_graph=True)[0]
        ux = th.autograd.grad(u, x, grad_outputs=th.ones_like(u), create_graph=True)[0]
        uxx = th.autograd.grad(ux, x, grad_outputs=th.ones_like(ux), create_graph=True)[0]
        
        # residual
        residual = utt - self.wave_equation.v**2 * uxx

        # MSE on the residual
        assert(residual.shape == th.zeros_like(residual).shape)
        loss_pde = self.loss_fn(residual, th.zeros_like(residual))

        # we're done
        return loss_pde
    
    def get_loss_bc(self):
        # sample BC points
        # just get all of them

        # (t=0, x) + (t, x=0) + (t, x=-1) + (t=-1, x)
        x_bc = th.cat([self.wave_equation.xs, 
                       self.wave_equation.xs[0].unsqueeze(0).repeat(self.wave_equation.ts.shape[0]), 
                       self.wave_equation.xs[-1].unsqueeze(0).repeat(self.wave_equation.ts.shape[0]),
                       self.wave_equation.xs,
                      ], dim=0)

        # (t=0, x) + (t, x=0) + (t, x=-1) + (t=-1, x)
        t_bc = th.cat([self.wave_equation.ts[0].unsqueeze(0).repeat(self.wave_equation.xs.shape[0]),
                        self.wave_equation.ts, 
                        self.wave_equation.ts, 
                        self.wave_equation.ts[-1].unsqueeze(0).repeat(self.wave_equation.xs.shape[0]),
                        ], dim=0)

        # (t=0, x) + (t, x=0) + (t, x=-1) + (t=-1, x)
        u_bc = th.cat([self.wave_equation.u[0, :], 
                       self.wave_equation.u[:, 0], 
                       self.wave_equation.u[:, -1],
                       self.wave_equation.u[-1, :],
                       ], dim=0).unsqueeze(-1)
        
        # move to device
        t_bc = t_bc.to(DEVICE)
        x_bc = x_bc.to(DEVICE)
        u_bc = u_bc.to(DEVICE)


        # ORDER IS (T, X)
        inputs = th.stack([t_bc, x_bc], dim=-1)

        # forward
        u_bc_pred = self.model(inputs)

        # compute the BC loss
        assert(u_bc_pred.shape == u_bc.shape)
        loss_bc = self.loss_fn(u_bc_pred, u_bc)

        # weigh it
        loss_bc = 10.0 * loss_bc

        # we're done
        return loss_bc
    
    def get_random_batch(self, batch_size=1024):
        # order should be (t, x)
        # ORDER IS (T, X)


        # sample data points
        ts = self.ts
        xs = self.xs

        # randomly choose some
        t_choices = np.random.choice(ts.shape[0], size=batch_size, replace=True)
        x_choices = np.random.choice(xs.shape[0], size=batch_size, replace=True)
        tsb = ts[t_choices]
        xsb = xs[x_choices]

        # zip the choices together
        # choices = list(zip(t_choices, x_choices))

        # get the u_gt
        # ORDER IS (T, X)
        u_gt = self.u[t_choices, x_choices].unsqueeze(-1) # shape (batch_size, 1)

        # assemble the batch
        # ORDER IS (T, X)
        sample = th.stack([tsb, xsb], dim=-1)

        return sample, u_gt
    
    def get_loss_data(self):
        sample, u_gt = self.get_random_batch()

        # forward
        u_pred = self.model(sample)

        assert(u_pred.shape == u_gt.shape)

        # compute the data loss
        assert(u_pred.shape == u_gt.shape)
        loss_data = self.loss_fn(u_pred, u_gt)

        # we're done
        return loss_data

    def compute_loss(self):
        # datapoint loss
        loss_data = self.get_loss_data()

        # PDE loss
        loss_pde = self.get_loss_pde()

        # BC loss
        loss_bc = self.get_loss_bc()

        # total loss
        loss = loss_pde + loss_data + loss_bc

        # we're done
        return loss
    
    def plot(self):
        # ORDER IS (T, X)
        if not PLOT:
            return

        # plot the final solution
        with th.no_grad():
            ts = self.ts
            xs = self.xs

            t_grid, x_grid = th.meshgrid(ts, xs, indexing='ij')
            sample = th.stack([t_grid.flatten(), x_grid.flatten()], dim=-1)
            
            u_pred = self.model(sample).cpu().numpy()

            u_pred = u_pred.reshape(t_grid.shape)

            tdomain = self.wave_equation.tdomain.cpu().numpy()
            xdomain = self.wave_equation.xdomain.cpu().numpy()
            extent = (tdomain[0], tdomain[1], xdomain[0], xdomain[1])

            plt.figure()
            plt.imshow(u_pred.T, extent=extent, origin='lower', aspect='auto')
            plt.colorbar(label='u(t, x)')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title('PINN Wave Equation Solution')
            plt.show()

            # compute the error compared to the finite difference solution
            u_gt = self.u.detach().cpu().numpy()
            error = np.mean(np.square(u_pred - u_gt))
            print(f"Mean squared error compared to finite difference solution: {error:.4f}")

        # plot loss over time
        losses = self.losses
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.yscale('log')
        plt.show()
    
class HW8:
    def __init__(self) -> None:
        wave_equation = WaveEquation()

        # generate dataset
        wave_equation.solve()

        # setup the network
        pinn = PINN(wave_equation)

        # run training
        pinn.run()

        # plot results
        pinn.plot()

if __name__ == "__main__":
    hw8 = HW8()