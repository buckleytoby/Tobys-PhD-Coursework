import numpy as np
from collections import defaultdict
from scipy.integrate import RK45

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import figure

# example
def xprimefcn(t, x):
    # xprime = fcn(t, x)
    xprime = np.zeros_like(x)
    return xprime

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
            x = self.step(x, xprime, t, dt)

            # save histories for plotting
            history['x'].append(x)
            history['t'].append(t)

        return history

    def plot(self, history, fig: figure.Figure):
        x = history['x']
        t = history['t']

        plt.plot(t, x)

class BackwardsEuler(ForwardEuler):
    def step(self, xn, xprimen, t, dt):
        # x_n+1 = x_n + dt * x'_n+1

        def g(t, x, dt, xold):
            return dt * t * x**4 - x + xold
        
        def gprime(t, x, dt):
            return 4 * dt * t * x**3 - 1
        
        n = 1000

        # initial guess
        xnplusone = xn

        # using newton's iterative root-finding method (from the notes)
        for i in range(n):
            new = xnplusone - g(t, xnplusone, dt, xold = xn) / gprime(t, xnplusone, dt)
            err = new - xnplusone

            xnplusone = new
            

        return xnplusone


class RK4(ForwardEuler):
    def __init__(self, xprimefcn) -> None:
        super().__init__(xprimefcn)

        self.rk45 = None
        self.xprimefcn

    def steps(self, t0, tf, x0, dt):
        self.rk45 = RK45(
            fun = self.xprimefcn,
            t0 = t0,
            y0 = [x0],
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

        xnplusone = self.rk45.y[0]

        return xnplusone



class Exercise3:
    def __init__(self) -> None:
        self.dts = [0.01, 0.02, 0.05, 0.1]

        # from the problem statement
        def xprimefcn(t, x):
            xp = t * x**4
            return xp
        
        self.xprimefcn = xprimefcn

        # set for simplicity
        self.x0 = 1.0
        self.t0 = 0.0
        self.tf = 0.99 * np.sqrt(2/(3 * self.x0**3)) # limit from the hw

    def part_b(self):
        fig = plt.figure()

        for dt in self.dts:
            solver = ForwardEuler(self.xprimefcn)
            history = solver.steps(self.t0, self.tf, self.x0, dt)

            solver.plot(history, fig)

        labels = [str(x) for x in self.dts]
        fig.legend(labels)
        plt.show()

    def part_c(self):
        fig = plt.figure()

        for dt in self.dts:
            solver = BackwardsEuler(self.xprimefcn)
            history = solver.steps(self.t0, self.tf, self.x0, dt)

            solver.plot(history, fig)

        labels = [str(x) for x in self.dts]
        fig.legend(labels)
        plt.show()

    def part_d(self):
        fig = plt.figure()

        for dt in self.dts:
            solver = RK4(self.xprimefcn)
            history = solver.steps(self.t0, self.tf, self.x0, dt)

            solver.plot(history, fig)

        labels = [str(x) for x in self.dts]
        fig.legend(labels)
        plt.show()

    def run(self):
        self.part_b()
        self.part_c()
        self.part_d()

def main():
    ex3 = Exercise3()
    ex3.run()

if __name__ == "__main__":
    main()
