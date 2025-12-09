import modern_robotics as mr
import numpy as np
from collections import defaultdict
import tqdm
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

from tqdm import tqdm


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
Glist = np.array([G1, G2, G3, G4, G5, G6])
Mlist = np.array([M01, M12, M23, M34, M45, M56, M67] )
Slist = np.array([[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]])


G = np.array([0, 0, -9.81]).T
DT = 1e-3

L1 = 0.425
L2 = 0.392
W1 = 0.109
W2 = 0.082
H1 = 0.089
H2 = 0.095
UR5_M = np.array([[-1, 0, 0, L1+L2],
              [0, 0, 1, W1+W2],
              [0, 1, 0, H1-H2],
              [0, 0, 0, 1]])


ad_Tb_s = mr.Adjoint(mr.TransInv(UR5_M))

Blist = ad_Tb_s @ Slist


# function from assignment 3
def fk(q):
    """
    returns the FK transform of q
    """
    Tsb = mr.FKinSpace(UR5_M, Slist, q)
    # Tsb = Tf(Tsb)
    return Tsb

# from assignment 3 
def error_twist(Tsb, Tsd):
    """
    computes the twist to get from T1 to T2 in 1 second. Assumes that T1 and T2 are w.r.t. the same frame
    output: space frame
    """
    Tsb_inv = mr.TransInv(Tsb)
    Tbd = Tsb_inv @ Tsd

    # convert from 4x4 transform matrix to a 4x4 log(T) mat
    log_Tbd = mr.MatrixLog6(Tbd)

    # go from the log 4x4 to the twist 6x1
    vb = mr.se3ToVec(log_Tbd)

    # ad_Ts_b = mr.Adjoint(Tsb)
    # vs = ad_Ts_b @ vb

    # v = vs
    v = vb

    v = v.reshape([6, 1])
    return v

# function from assignment 3
def IKinBodyIterates(xd, q0):    
    # params
    nb_joints = q0.shape[0]

    # loop vars
    c = 0
    eps_r = 0.001
    eps_l = 0.0001
    er_mag = 99.0
    el_mag = 99.0

    # joints
    if q0 is None:
        q = np.zeros([nb_joints, 1])
    else:
        q = np.reshape(q0, (nb_joints, 1))

    assert(q.shape == (nb_joints, 1))

    # ee config
    Tsb = fk(q)
    error_t = error_twist(Tsb, xd)

    while er_mag > eps_r or el_mag > eps_l:
        # increment the counter
        c += 1

        # early exit
        if c > 2000:
            print("Didn't converge.")
            raise

        # compute the jacobian
        Js = mr.JacobianBody(Blist, q)

        # invert the jacobian using np
        J_inv = np.linalg.pinv(Js)

        # compute the delta-x
        # dx = xd - Tsb

        # compute the delta-q
        dq = J_inv @ error_t

        # compute the new q
        q = q + dq

        # wrap the joints
        q = np.atan2(np.sin(q), np.cos(q))

        # compute new Tsb
        Tsb = fk(q)

        # compute the error twist
        error_t = error_twist(Tsb, xd)

        # compute the ang error
        er = error_t[0:3]

        # compute the linear error
        el = error_t[3:6]

        # compute the error mags
        er_mag = np.linalg.norm(er)
        el_mag = np.linalg.norm(el)
        pass

    return q.squeeze()

class State:
    def __init__(self,
                 theta_init,
                 ) -> None:
        self.theta_init = theta_init

        # my members
        self.history = defaultdict(list)

        # current state values
        self.current = {}

        self.current["t"] = 0.0
        self.current["th"] = np.array(theta_init)
        self.current["dth"] = np.zeros(6)
        self.current["ddth"] = np.zeros(6)
        self.current["tau_cmd"] = np.zeros(6)
        self.current["ang_err"] = np.zeros(1)
        self.current["lin_err"] = np.zeros(1)

    # def reset(self):
    #     """
    #     required state vars:
    #         th
    #         dth
    #         ddth
    #         tau_cmd
    #         ang_err
    #         lin_err
    #     """
    #     keys = [
    #         "th",
    #         "dth",
    #         "ddth",
    #         "tau_cmd",
    #         "ang_err",
    #         "lin_err",
    #     ]

    #     self.current = {}

    #     for key in keys:
    #         self.current[key] = 0.0

    def get_key(self, key):
        return self.current[key]

    def get_step_inputs(self):
        keys = [
            "th",
            "dth",
            "tau_cmd"
        ]
        s = [self.get_key(key) for key in keys]
        return s
    
    def get_current_config(self):
        """
        get current end effector config
        """
        th = self.get_key("th")

        # fk
        Tsb = mr.FKinSpace(UR5_M, Slist, th)

        return Tsb
    
    def update_key(self, key, val):
        # now set
        self.current[key] = val

        # first record old value
        self.record_key(key)


    def record_key(self, key):
        self.history[key].append(np.copy(self.get_key(key)))

    def record(self):
        for key in self.current.keys():
            self.record_key(key)

    def load(self):
        name = "./input.txt"
        arr = np.loadtxt(name)

    def save_coppelia(self, folder, prefix):
        qh = np.array(self.history["th"]).squeeze()
        np.savetxt("./" + folder + "/{}.csv".format(prefix), qh, delimiter=',')

    def save_time_history(self, folder, keys, label):
        nb_keys = len(keys)

        # get t
        t = self.history["t"]
        t = np.array(t)
        t = np.expand_dims(t, axis=-1)

        # get keys
        ths = [self.history[key] for key in ["t"] + keys]
        hs = [np.array(self.history[key]) for key in keys]

        # stack
        ths = np.hstack([t, np.hstack(hs)])

        # save csv
        np.savetxt("./" + folder + "/time_history_" + label + ".csv", ths, header="time, " + label)

        # also plot and save
        fig1, axes = plt.subplots(nb_keys)
        
        # iterate over all non-time state variables
        for i in range(nb_keys):
            key = keys[i]
            h = np.array(self.history[key])

            if isinstance(axes, np.ndarray):
                axis = axes[i]
            else:
                axis = axes

            axis.plot(t, h, color="purple", alpha=0.5)
            axis.set_ylabel(key)
            axis.legend()


        plt.savefig('./{}/'.format(folder) + label + '.png', dpi=300, bbox_inches='tight')


class Trajectory:
    def __init__(self,
                 type = "screw",
                 ts_start = np.zeros((4, 4)),
                 ts_end = np.zeros((4, 4)),
                 ) -> None:
        self.type = type
        self.ts_start = ts_start
        self.ts_end = ts_end

    def run(self, s):
        """
        from s in [0, 1], get the target
        """
        if self.type == "screw":
            # eq 9.6
            # Xs = Xs,start @ exp( log (Xs,start^-1 * Xs,end) * s)
            Tse = mr.TransInv(self.ts_start) @ self.ts_end

            # screw axis, start to end
            Sse = mr.se3ToVec(
                mr.MatrixLog6(
                    Tse
                )
            )

            # scale screw axis w.r.t. s to get a twist
            V = Sse * s

            # integrate the twist to get a final config, w.r.t. start frame
            Tstart_s = mr.MatrixExp6(
                    mr.VecTose3(
                        V
                    )
                )
            
            # transform to {s} frame?
            Ts_s = self.ts_start @ Tstart_s
            
            # final config as the target?
            target = Ts_s

        elif self.type == "cartesian":
            Rstart, pstart = mr.TransToRp(self.ts_start)
            Rend, pend = mr.TransToRp(self.ts_end)

            # eq 9.7 and 9.8
            p = pstart + s*(pend - pstart)

            R = Rstart * mr.MatrixExp3(
                mr.MatrixLog3(
                    Rstart.T @ Rend
                ) * s
            )

            # reconstruct the target config
            Ts_s = mr.RpToTrans(R, p)

            target = Ts_s

        else:
            raise


        return target
    
class TimeScaling:
    def __init__(self,
                 type = "quintic",
                 T = 1, # total time
                 ) -> None:
        self.type = type
        self.T = T

    def run(self, t):
        """
        cubic or quintic

        input:
            t
        output:
            s
        """
        if self.type == "cubic":
            # eq 9.10
            a0 = 0
            a1 = 0
            a2 = 3/self.T**2
            a3 = -2/self.T**3
            s = a2 * t**2 + a3 * t**3
            # s_dot = a1 + 2*a2*t + 3*a3*t**2

        elif self.type == "quintic":
            # soln of exercise 9.5
            # x=Ab
            T = self.T
            x = np.array([1, 0, 0]).T
            A = np.array([[T**3, T**4, T**5],
                          [3*T**2, 4*T**3, 5*T**4],
                          [6*T, 12*T**2, 20*T**3]])
            
            # solve the system for a3, a4, a5
            b = np.linalg.solve(A, x)
            # b = x @ A^-1

            # s(t) = a3*t**3 + a4*t**4 + a5*t**5
            s = b @ np.array([t**3, t**4, t**5])
            # s_dot = b @ np.

        else:
            raise

        assert( s >= 0.0)
        assert( s <= 1.0)
        return s



class Controller:
    def __init__(self,
                 type = "joint_space",
                 Kp = 0.0,
                 Ki = 0.0,
                 Kd = 0.0
                 ) -> None:
        self.type = type
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0

    def compute_error(self, xd, x):
        Xs_d = xd
        Xs_b = x
        V_e = mr.se3ToVec(
            mr.MatrixLog6(
                mr.TransInv(Xs_b) @ Xs_d
            )
        )
        V_e = V_e.reshape([6, 1])

        ang_err = np.linalg.norm(V_e[0:3], keepdims=True)[0] #type:ignore
        lin_err = np.linalg.norm(V_e[3:6], keepdims=True)[0] #type:ignore

        self.state.update_key("ang_err", ang_err)
        self.state.update_key("lin_err", lin_err)

        return V_e

    def joint_space_control_law(self, state: State, xd):
        th = state.get_key("th")
        dth = state.get_key("dth")

        # inverse kinematics
        th_d = IKinBodyIterates(xd, th)


        # current config, in the space frame
        Ts_b = state.get_current_config()

        # compute error - returns error twist, in the body frame
        Ve_b = self.compute_error(xd, Ts_b)
        ad_Ts_b = mr.Adjoint(Ts_b)
        Ve_s = ad_Ts_b @ Ve_b

        # eq 11.45
        Js_inv = np.linalg.pinv(mr.JacobianSpace(Slist, th))
        dth_d =  Js_inv @ Ve_s

        # squeeze
        th_d = np.squeeze(th_d)
        dth_d = np.squeeze(dth_d)

        th_e = th_d - th
        th_e_dot = dth_d - dth

        # by default don't compute the integral
        if self.Ki > 0.0:
            int_th_e = self.integral

            # increment the integral
            self.integral += th_e * DT
        else:
            int_th_e = 0.0

        g_tilde = mr.GravityForces(
            th,
            G,
            Mlist,
            Glist,
            Slist
        )

        # use eq 11.38, no feed-forward implemented (for now)
        tau = self.Kp * th_e + self.Ki * int_th_e + self.Kd * th_e_dot + g_tilde

        return tau


    def step(self, state: State, target):
        self.state = state
        xd = target

        # compute control law, section 11.4.3: task space motion control with torque inputs
        if self.type == "joint_space":
            # eq's 11.44, 11.45, 11.46

            tau_cmd = self.joint_space_control_law(state, xd)

        else:
            raise

        state.update_key("tau_cmd", tau_cmd)
        return tau_cmd



class Simulator:
    def __init__(self, 
                 ) -> None:
        self.dt = DT
        
        # my members
        self.g = G
        self.ftip = np.zeros(6) # assume no ftip
        self.tau_damping_coeff = 0.5 # N*m/(rad/s) --> tau_damping_i = -0.5 * theta_dot_i

        self.t = 0

    def forward(self, th, dth, ddth):
        """
        first order euler integration
        input: 
            th - theta
            dth - delta-theta aka theta_dot
            ddth - delta-delta-theta aka theta_dot_dot
        output:
            thp - theta-prime aka the new theta
            dthp - delta-theta-prime ak athe new delta-theta
        """
        thp, dthp = mr.EulerStep(
            th.squeeze(),
            dth.squeeze(),
            ddth.squeeze(),
            self.dt
        )

        # add onto total time
        self.t += self.dt
        self.state.update_key("t", self.t)

        # return thp.reshape([6, 1]), dthp.reshape([6, 1])
        return thp, dthp

    def step(self, state: State):
        """
        inputs:
            thl - theta list
            dthl - delta-theta list
            taul - tau-list aka joint torque cmd list
        """
        thl, dthl, taul = state.get_step_inputs()
        taul = np.copy(taul).squeeze()

        # add on damping
        tau_damping = -1.0 * self.tau_damping_coeff * dthl
        taul += tau_damping

        ddth = mr.ForwardDynamics(
            thl.squeeze(),
            dthl.squeeze(),
            taul,
            self.g,
            self.ftip,
            Mlist,
            Glist,
            Slist
        )

        # first order euler integration
        thp, dthp = self.forward(thl, dthl, ddth)


        state.update_key("th", thp)
        state.update_key("dth", dthp)

    def get_time(self):
        return self.t

    def run(self, 
            state: State,
            time_scaling: TimeScaling,
            trajectory: Trajectory,
            controller: Controller,
            T
            ):
        # reset some stuff
        self.t = 0

        # save refs
        self.state = state

        done = False
        with tqdm(total=T) as pbar:
            while not done:
                pbar.update(self.dt)

                # update/record state info
                state.record()
                
                # get s from time-scaling
                s = time_scaling.run(self.get_time())

                # get target from trajectory, fcn of s
                target = trajectory.run(s)

                # get controller feedback terms
                controller.step(state, target)

                # step the forward dynamics
                self.step(state)

                # check end condition
                t = self.get_time()

                if t > T:
                    done = True


def main():
    # load the config
    config = OmegaConf.load("./input.yaml")
    OmegaConf.save(config, "./" + config.folder + "/input.yaml")

    T = config.duration

    ts_start = np.array(config.ts_start)
    ts_end = np.array(config.ts_end)

    # get initial joint config
    th0 = IKinBodyIterates(ts_start, np.zeros(6))
    thi = th0 + config.theta_init_perturbation

    # class instances
    state = State(theta_init = thi)
    time_scaling = TimeScaling(type = config.time_scaling_type, T=T)
    trajectory = Trajectory(type = config.traj_type, ts_start=ts_start, ts_end=ts_end)
    controller = Controller(type = config.controller_type, Kp = config.Kp, Ki = config.Ki, Kd = config.Kd)

    # save initial error
    controller.state = state
    controller.compute_error(xd=ts_start, x=state.get_current_config())

    # record initial vars (not already done in compute-error)
    state.record_key("t")
    state.record_key("tau_cmd")
    state.record_key("th")



    # make the simulator
    sim = Simulator()

    # run the sim
    sim.run(state, time_scaling, trajectory, controller, T)

    # save the outputs
    # 1. coppelia-compatible csv
    state.save_coppelia(config.folder, "coppelia")

    # 2. csv of time, joint-angles
    state.save_time_history(config.folder, ["th"], "joint_angles")

    # 3. csv of time, cmd joint torques
    state.save_time_history(config.folder, ["tau_cmd"], "joint_torque_cmds")

    # 4. csv of time, angular error, linear error
    state.save_time_history(config.folder, ["ang_err", "lin_err"], "ang_and_lin_err")


    pass


if __name__ == "__main__":
    main()