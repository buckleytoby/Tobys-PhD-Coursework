import modern_robotics as mr
import numpy as np
from collections import defaultdict
import tqdm
from omegaconf import OmegaConf
from matplotlib import pyplot as plt


### GLOBALS
## UR5 params
M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])

# constants
Glist = [G1, G2, G3, G4, G5, G6]
Mlist = [M01, M12, M23, M34, M45, M56, M67] 
Slist = [[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]]

class State:
    def __init__(self) -> None:
        self.history = defaultdict(list)

        # current state values
        self.current = {}

    def load(self):
        name = "./input.txt"
        arr = np.loadtxt(name)

    def save_coppelia(self, prefix):
        qh = np.array(self.history["th"]).squeeze()
        np.savetxt("./{}.csv".format(prefix), qh, delimiter=',')

    def save_time_history(self, keys, label):
        nb_keys = len(keys)

        # get t
        t = self.history["t"]

        # get keys
        hs = [self.history[key] for key in keys]

        # stack
        ths = np.hstack([t,] + hs)

        # save csv
        np.savetxt("time_history_" + label, ths)

        # also plot and save
        fig1, axes = plt.subplots(nb_keys)
        
        # iterate over all non-time state variables
        for i in range(nb_keys):
            key = keys[i]
            h = self.history[key]
            axes[i].plot(t, h, color="purple", alpha=0.5)
            axes[i].set_ylabel(key)

        plt.savefig('./outputs/' + label + '.png', dpi=300, bbox_inches='tight')


class Trajectory:
    def __init__(self) -> None:
        pass

class TimeScaling:
    def __init__(self) -> None:
        pass

class Controller:
    def __init__(self) -> None:
        self.tau_damping = 0.5 # N*m/(rad/s) --> tau_damping_i = -0.5 * theta_dot_i

class Simulator:
    def __init__(self) -> None:
        

        # make my composed classes
        pass

    def forward(self, th, dth, ddth):
        """
        first order euler integration
        """
        thp, dthp = mr.EulerStep(
            th.squeeze(),
            dth.squeeze(),
            ddth.squeeze(),
            self.dt
        )
        return thp.reshape([6, 1]), dthp.reshape([6, 1])

    def step(self):
        mr.ForwardDynamics(
            thl,
            dthl,
            taul,
            g,
            ftip,
            Mlist,
            Glist,
            Slist
        )


        # first order euler integration
        thp, dthp = self.forward(thetalist, dthetalist, ddth)

        # save new state vars
        thetalist = thp
        dthetalist = dthp

    def run(self):
        pass


def main():
    # load the config
    config = OmegaConf.create("./input.yaml")

    # make the simulator
    sim = Simulator()

    # run the sim
    sim.run()

    # save the outputs
    # 1. coppelia-compatible csv
    state.save("coppelia")

    # 2. csv of time, joint-angles
    state.save_time_history("th")

    # 3. csv of time, cmd joint torques
    state.save_time_history("tau_cmd")

    # 4. csv of time, angular error, linear error
    state.save_time_history(["ang_err", "lin_err"])


    pass


if __name__ == "__main__":
    main()