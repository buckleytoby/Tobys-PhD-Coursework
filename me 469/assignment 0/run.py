# "Allowable python packages are: numpy, matplotlib, pandas, scipy, seaborn, and default packages (such as math, time, etc.)."
import os, sys, math, time, collections, csv, re, copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats
import scipy.signal

PI2 = 2.0 * math.pi
TEST = False
VERBOSE = True

"""
Forward velocity (along the x-axis of the robot body frame) commands, v
Angular velocity commands (rotation about the z-axis of the robot body frame using right hand rule)
"""

"""
two-wheel differential drive => c-space: (x, y, theta, phi_1, phi_2), with parameters wheel radius (r), and distance from wheel to body center (l)
"""

"""
robot 3
"""

def wrap(angle):
    # ref: https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    angle2 = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle2


class State:
    def __init__(self, state0=None, dt = 1e-3):
        if state0 is None:
            t0, x0, y0, theta0 = 0.0, 0.0, 0.0, 0.0
        else:
            t0, x0, y0, theta0 = state0

        # setup state variables
        self.t = t0
        self.x = x0
        self.y = y0
        self.theta = theta0

        self.state = self.bundle_state_vector()
        
    def bundle_state_vector(self):
        return [self.t, self.x, self.y, self.theta, self.phi_1, self.phi_2]
    
    def unpack_state_vector(self):
        self.t, self.x, self.y, self.theta, self.phi_1, self.phi_2 = self.state
    
class History:
    def __init__(self):
        # setup the history
        self.data = []
        self.data_frequency_in_steps = 100
        self.freq_dt = 0.5 # s
        
    def add_history(self, step_count, state):
        # if (step_count % self.data_frequency_in_steps) == 0:
        if len(self.data) == 0 or state[0] > self.data[-1][0] + self.freq_dt:
            self.data.append(state)

    def get_history(self):
        return np.array(self.data)

class MotionModel:
    def __init__(self, state0=None, dt = 1e-3, variance=None):
        if state0 is None:
            t0, x0, y0, theta0 = 0.0, 0.0, 0.0, 0.0
        else:
            t0, x0, y0, theta0 = state0
        # params
        self.dt = dt
        self.variance = variance

        # t0 = 0.0
        # x0 = 0.0
        # y0 = 0.0
        # theta0 = 0.0
        phi_10 = 0.0
        phi_20 = 0.0
        
        # setup state variables
        self.t = t0
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.phi_1 = phi_10
        self.phi_2 = phi_20

        self.state = np.array(self.bundle_state_vector())

        # noise
        if self.variance is not None:
            assert(len(self.variance) == len(self.state))
            self.gaus = scipy.stats.multivariate_normal(
                mean=np.zeros_like(self.state),
                cov=np.diag(self.variance),
                allow_singular=True
            )
        else:
            self.gaus = None


        # setup params, taken from here: https://iroboteducation.github.io/create3_docs/hw/mechanical/#dimensioned-drawings
        self.r = 72. / 1000.
        self.l = 235. / 2. / 1000.

        self.step_count = 0
        self.history = History()

        # save initial state
        self.add_history()

    def calc_gradient(self, u):
        # dt is always 1.0 second per second
        dt = 1.0

        # extract vel (m/s) and ang. vel. (rad/s)
        v, w = u

        # double check eq for dk
        dk = w

        slip_1 = 0.1
        slip_2 = 0.1

        # including slip
        # dphi_1 = <> + slip_1
        # dphi_2 = <> + slip_2

        grad = np.array([
            dt,
            dx,
            dy,
            dk,
            dphi_1,
            dphi_2
        ])

        return grad

    def propagate(self, dt, u):
        # compute the gradient, aka the rate of change of each state variable
        gradient = self.calc_gradient(u)

        # simple numerical integration
        state = self.state + dt * gradient

        # add on noise, labelled as a multi-variate gaussian
        if self.gaus is not None:
            state += self.gaus.rvs() * dt

        self.set_state(state)

    def step(self, cmd):
        """
        cmd = t, v, w
        """
        t, v, w = cmd
        u = [v, w]

        t0 = self.state[0]
        tf = t

        # propagate dynamics
        while self.t < tf:
            # get time step
            dt_full = tf - self.t
            dt_max = self.dt

            # this way dt will be at MOST, dt_max, but potentially as little as dt_full (when dt_full < dt_max)
            dt = min(dt_full, dt_max)

            self.propagate(dt, u)

            # add to history
            self.add_history()

            self.step_count += 1

    def step_until(self, cmds, tf):
        """
        cmd = t, v, w
        tf = final time
        """
        # propagate dynamics
        while self.t < tf:
            # get time step
            dt_full = tf - self.t
            dt_max = self.dt

            # this way dt will be at MOST, dt_max, but potentially as little as dt_full (when dt_full < dt_max)
            dt = min(dt_full, dt_max)

            self.propagate(dt, u)

            # add to history
            self.add_history()
        
    def bundle_state_vector(self):
        return [self.t, self.x, self.y, self.theta, self.phi_1, self.phi_2]
    
    def unpack_state_vector(self):
        self.t, self.x, self.y, self.theta, self.phi_1, self.phi_2 = self.state
        
    def add_history(self):
        self.history.add_history(self.step_count, self.bundle_state_vector())

    def get_history(self):
        return self.history.get_history()
    
    def set_state(self, state):
        self.state = state

        # save state to individual vars
        self.unpack_state_vector()

    def get_state(self):
        return self.state
    
class SimpleMotionModel(MotionModel):
    def calc_gradient(self, u):
        # dt is always 1.0 second per second
        dt = 1.0

        # extract vel (m/s) and ang. vel. (rad/s)
        v, w = u

        dtheta = w
        dx = v * math.cos(self.theta)
        dy = v * math.sin(self.theta)

        grad = np.array([dt, dx, dy, dtheta, 0.0, 0.0])
        return grad
    
class SimpleMeasurementModel():
    """
    
    """
    def __init__(self, dataloader, variance = None):
        self.dataloader = dataloader
        self.variance = variance


        ## member vars
        # likelihood
        self.w = 0
        self.history = History()
    
    def get_subject_id(self, barcode_id):
        # convert barcode_id to subject_id
        # using https://pandas.pydata.org/docs/user_guide/10min.html 
        # and here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html
        barcodes = self.dataloader.data['Barcodes']
        barcodes_pd = pd.DataFrame(barcodes, columns=["subjectid", "barcodeid"], dtype='int')
        barcodes_pd = barcodes_pd.set_index("barcodeid")

        subjectid = barcodes_pd.at[barcode_id, "subjectid"]

        return subjectid
    
    def range_bearing(self, xy1, theta, xy2):
        """
        range and bearing going FROM xy1 TO xy2.
        so xy1 should be the robot, xy2 should be the landmark
        """
        xy1 = np.array(xy1)
        xy2 = np.array(xy2)
        s = xy2 - xy1
        range = np.linalg.norm(s)

        # output in [-pi, pi]
        theta_landmark = np.atan2(s[1], s[0])

        # wrap to [-pi, pi]
        theta2 = wrap(theta)

        # heading is FROM the robot TO the landmark
        heading = theta_landmark - theta2

        return range, heading

        
    def landmark_transform(self, rxy, rtheta, subject_nb):
        landmark_gts = self.dataloader.data["Landmark_Groundtruth"]

        # first id is 6
        index = subject_nb - 6
        id, x, y, x_stddev, y_stddev = landmark_gts[index]
        
        range, heading = self.range_bearing(rxy, rtheta, [x, y])

        return range, heading
    
    def weight_m(self, xt_m, meas, landmark_xy):
        """
        given xt[m], what was the probability of zt.
        assume the probability follows a gaussian, centered on xt_m
        assume some range variance and bearing variance

        wt_m = p(zt | xt_m)
        """
        range_var = 0.1 # m
        bearing_var = 0.17 # rad

        tt, barcode_id, meas_range, meas_bearing = meas

        # compute what the range, bearing should have been, given xt_m
        _, x, y, theta = xt_m[0:4]
        xt_m_range, xt_m_bearing = self.range_bearing([x, y], theta, landmark_xy)

        if True:
            # gaus in measurement space: range, bearing
            # inspired by this post: https://stats.stackexchange.com/questions/49160/particle-filtering-importance-weights
            assert(len(self.variance) == 2)
            gaus = scipy.stats.multivariate_normal(
                mean=[xt_m_range, xt_m_bearing],
                cov=np.diag(self.variance) # [range_var, bearing_var])
            )
            p = gaus.pdf([meas_range, meas_bearing])

        # experimenting with custom error based weighting
        if False:
            s = np.array([xt_m_range, xt_m_bearing])
            s_meas = np.array([meas_range, meas_bearing])
            ds = s - s_meas
            d = np.linalg.norm(ds)
            
            # zero protection
            d = max(1e-4, d)

            # closer error is to zero, the happier we are
            p = 1.0 / (d*d)

        assert(not np.isnan(p))

        self.w = p
        return p
    
    def get_meas_landmark_xy(self, meas):
        """
        meas - Time [s]    Subject #    range [m]    bearing [rad] 
        I'm assuming the meas is the measurement from ME to `subject #`
        however, the "subject #'s" range from 5 to 81 so obviously this is misslabeled, and should actually be the barcode number
        """
        tt, barcode_id, range, bearing = meas

        subjectid = self.get_subject_id(barcode_id)

        # setup landmark groundtruth's
        landmark_gt = self.dataloader.data['Landmark_Groundtruth']
        landmark_gt_pd = pd.DataFrame(landmark_gt, columns=["subjectid", 'x', 'y', 'xstddev', 'ystddev' ])
        landmark_gt_pd = landmark_gt_pd.set_index("subjectid")

        # subjectid of <6 is another robot, so we can't use it
        if subjectid < 6:
            return False, None
        
        row = landmark_gt_pd.loc[subjectid]
        landmark_xy = np.array([row.x, row.y])

        return True, landmark_xy

    def get_next_meas(self, t):
        pass

    def weight_from_meas(self, xt_m, meas):
        # wt[m] = p(zt | xt[m]) aka weight
        valid, landmark_xy = self.get_meas_landmark_xy(meas)

        if valid:
            # wt[m] = p(zt | xt[m]) aka weight
            self.weight_m(xt_m, meas, landmark_xy)

class MotionAndMeasurement:
    def __init__(self,
                 motion,
                 meas
                 ):
        self.motion: SimpleMotionModel = motion
        self.meas: SimpleMeasurementModel = meas

class ParticleFilter:
    def __init__(self,
                 dataloader,
                 nb_particles,
                 ):
        self.dataloader: DataLoader = dataloader
        self.nb_particles = nb_particles

    def initialize(self, state0, stats, motion_variance, meas_variance, dt):
        """
        we've been given the initial ground truth location, so just use that with no variance
        """
        self.particles = []

        for m in range(self.nb_particles):
            motion = SimpleMotionModel(dt=dt, state0=state0, variance=motion_variance)
            meas = SimpleMeasurementModel(self.dataloader, variance=meas_variance)

            p = MotionAndMeasurement(
                motion,
                meas
            )

            self.particles.append(p)

    def get_control(self, idx):
        control = self.cmds[idx]

        return control

    def step_controls(self, controls, controls_idx, tf):
        # get initial control
        control = controls[controls_idx]
        t, v, w = control

        # always progress controls idx by at least one to force progression
        controls_idx += 1

        if controls_idx >= len(controls):
            return controls_idx

        while t < tf:
            # get control
            control = controls[controls_idx]
            tt, v, w = control

            # make sure we stop exactly at tf
            tt = min(tt, tf)

            control[0] = tt

            # for all particles
            for m in range(self.nb_particles):
                # sample xt[m] from p(xt | ut, xt-1[m]) aka state transition
                particle: MotionAndMeasurement = self.particles[m]

                # step the particle according to the cmd
                particle.motion.step(control)

            # update loop vars
            t = tt
            controls_idx += 1

            if controls_idx >= len(controls):
                break

        return controls_idx

    def step_controls_avg(self, controls, controls_idx, tf):
        """
        Average all the controls from t to tf to expedite compute
        """
        # get initial control
        cii = controls_idx
        control = controls[controls_idx]
        t, v, w = control

        while t < tf:
            # get control
            control = controls[controls_idx]
            tt, v, w = control

            # make sure we stop exactly at tf
            tt = min(tt, tf)

            # update loop vars
            t = tt
            controls_idx += 1

            if controls_idx >= len(controls):
                break

        controls2 = controls[cii:controls_idx-1]
        controls2_avg = np.average(controls2, axis=0)
        controls2_avg[0] = tf

        # for all particles
        for m in range(self.nb_particles):
            # sample xt[m] from p(xt | ut, xt-1[m]) aka state transition
            particle: MotionAndMeasurement = self.particles[m]

            # step the particle according to the cmd
            particle.motion.step(controls2_avg)


        return controls_idx

    def step_meas_and_redraw(self, meass, meas_idx, tf):
        """
        step through measurements until we've caught up to the next control
        redraw for each measurement so that no measurements are discarded
        """
        # get initial control
        meas = meass[meas_idx]
        t = meas[0]

        while t < tf:
            # get control
            meas = meass[meas_idx]
            tt = meas[0]
            barcodeid = int(meas[1])

            # check to see if the measurement is valid
            particle: MotionAndMeasurement = self.particles[0]
            subjectid = particle.meas.get_subject_id(barcodeid)

            # only subjects 6-20 (landmarks) are valid
            if subjectid >= 6:
                for m in range(self.nb_particles):
                    particle: MotionAndMeasurement = self.particles[m]
                    xt_m = particle.motion.bundle_state_vector()

                    # calculate the weight from the measurement and xt_m
                    particle.meas.weight_from_meas(xt_m, meas)

                # redraw for each valid measurement
                self.redraw()

            # update loop vars
            t = tt
            meas_idx += 1

            if meas_idx >= len(meass):
                break

        return meas_idx

    def extract_weights(self):
        ws = []
        for m in range(self.nb_particles):
            particle: MotionAndMeasurement = self.particles[m]
            w = particle.meas.w

            ws.append(w)

        return ws
    
    def normalize_weights(self, ws):
        ws = np.array(ws)

        sum = np.sum(ws)

        # zero protection
        if sum == 0.0:
            ws = np.ones_like(ws)

        ws /= np.sum(ws)

        return ws

    def redraw(self):
        """
        given x_bar, re-draw our particles
        """
        ws = self.extract_weights()
        ws2 = self.normalize_weights(ws)

        a = range(self.nb_particles)

        # correction
        for _ in range(self.nb_particles):
            # draw i w/prob proportional to wt[i]
            # ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            i = np.random.choice(a, p=ws2)

            # # deep copy the drawn particle
            # new_particle = copy.deepcopy(self.particles[i])
            state = self.particles[i].motion.get_state()
            self.particles[_].motion.set_state(state)

        pass

    def run(self, cmds, meass):
        """
        cmds: [tt, v, w]
        meass: [Time [s],    Subject #,    range [m],    bearing [rad] ]
        """
        self.cmds = cmds
        self.meass = meass
        controls = cmds # rename

        # first control
        ti = cmds[0][0]
        if len(meass) > 0:
            tf = max(controls[-1][0], meass[-1][0]) # end of controls or meas
        else:
            tf = controls[-1][0]

        # first measurement
        if len(meass) > 0:
            tm = meass[0][0]
        else:
            tm = tf
        
        # controls should start before measurements
        assert(ti < tm)

        # setup loop variables
        controls_idx = 0
        meas_idx = 0
        t = ti
        tm = tm

        count = 0
        freq = 20

        # until we're done
        while t < tf:
            # propagate controls until the next measurement is reached
            if controls_idx < len(controls):
                controls_idx = self.step_controls(controls, controls_idx, tm)
            
            # extract next tc
            if controls_idx < len(controls):
                tc = cmds[controls_idx][0]

            # no more commands => we are done
            else:
                break
            t = tc

            # step through measurements until the next control is reached and redraw for each measurement
            if meas_idx < len(meass):
                meas_idx = self.step_meas_and_redraw(meass, meas_idx, tc)

            # extract next tm
            if meas_idx < len(meass):
                tm = meass[meas_idx][0]
                t = tm

                # at this point the next control should always be before the next meas
                # assert(tc < tm)

            # else, go until tf
            else:
                tm = tf
            

            count +=1
            if VERBOSE and (count % freq) == 0:
                # print("{} / {} s".format(t, tf))
                print("sim time left: {:.2f}".format(tf-t))

            

    def get_state_histories(self):
        histories = []

        p: MotionAndMeasurement = None
        for p in self.particles:
            histories.append(p.motion.get_history())

        return histories
    
    def get_averaged_state_history(self):
        history = []

        p: MotionAndMeasurement = self.particles[0]

        l = len(p.motion.history.data)

        for i in range(l):

            states = [] # x, y, theta
            for p in self.particles:
                states.append(p.motion.history.data[i])

            states = np.array(states)
            state = np.mean(states, axis=0)

            history.append(state)

        return history
    
    # def get_weighted_state_history(self):
    #     history = []

    #     p: MotionAndMeasurement = self.particles[0]

    #     l = len(p.motion.history)

    #     for i in range(l):

    #         state = # x, y, theta

    def plot_error(self, h, prefix, gt=None):
        my_label = "Particle Filter"

        h = np.array(h)

        # extract the history
        t = h[:, 0]
        x = h[:, 1]
        y = h[:, 2]
        theta = h[:, 3]

        # wrap(theta) # in place
        # h[:, 3] = theta

        # extract g.t.
        if gt is not None:
            gt_t = gt[:, 0]
            gt_x = gt[:, 1]
            gt_y = gt[:, 2]
            gt_theta = gt[:, 3]

            # unwrap for plotting
            gt_theta = np.unwrap(gt_theta)
            gt[:, 3] = gt_theta

        # plotting
        nb_non_time_state_vars = 3 # h.shape[1] - 1
        names = ["x", "y", "theta"]

        # plot 1 - t vs all state variables
        fig1, axes = plt.subplots(nb_non_time_state_vars)
        
        # iterate over all non-time state variables
        for i in range(nb_non_time_state_vars):
            axes[i].plot(t, h[:, i+1], color="purple", alpha=0.5)
            axes[i].set_ylabel(names[i])

            # gt
            if gt is not None:
                axes[i].plot(gt_t, gt[:, i+1], color="orange", alpha=0.5)

                axes[i].legend([my_label, "Ground Truth"])

        plt.xlabel("Time (s)")
        plt.suptitle(prefix + ": Time History of State Variables")
        fig1.show()

    
    def plot_mm_xy(self, prefix, mm: MotionModel, fig2 = None):
        h = np.array(mm.get_history())

        # extract the history
        t = h[:, 0]
        x = h[:, 1]
        y = h[:, 2]
        theta = h[:, 3]

        # plot 2 - birds-eye view: x vs y
        width = 0.005
        my_label = "Particle Filter"
        fig2 = plotbirdseye(prefix, "_nolegend_", x, y, theta, width=width, fig2 = fig2)

    def plot(self, prefix, gt=None):
        p: MotionAndMeasurement = None

        # plot the average
        avg_history = self.get_averaged_state_history()

        self.plot_error(avg_history, prefix, gt)

    def plot_xy(self, prefix, gt = None):
        # for p in self.particles:
        #     self.plot_mm(prefix, p.motion)

        # xytheta
        fig2 = plt.figure()
        for p in self.particles:
            self.plot_mm_xy(prefix, p.motion, fig2)

        # add on ground truth
        if gt is not None:
            # extract g.t.
            n = 10
            gt_x = gt[::n, 1]
            gt_y = gt[::n, 2]
            gt_theta = gt[::n, 3]

            # unwrap for plotting
            gt_theta = np.unwrap(gt_theta)
            fig2 = plotbirdseye(prefix, "Ground Truth", gt_x, gt_y, gt_theta, fig2=fig2)
            fig2.legend()

        fig2.show()




# http://asrl.utias.utoronto.ca/datasets/mrclam/index.html#Details
class DataLoader:
    def __init__(self,
                 test=TEST,
                 max_nb_pts = None,
                 percentage = None
                 ):
        self.test = test
        self.max_nb_pts = max_nb_pts
        self.percentage = percentage

        # params
        dir = "./ds0"
        prefix = "ds0"
        postfixes = [
            "Barcodes",
            "Control",
            "Groundtruth",
            "Landmark_Groundtruth",
            "Measurement"
        ]

        filetype = ".dat"

        self.data = {}

        for postfix in postfixes:
            key = postfix
            local_fn = prefix + "_" + postfix + filetype
            fn = os.path.join(dir, local_fn)


            data = []

            df = pd.read_csv(fn, sep=r'\s+', skipinitialspace=True, comment="#", header=None)
            data = df.to_numpy()

            if self.max_nb_pts is not None:
                data = data[:self.max_nb_pts]

            self.data[key] = np.array(data, dtype="double")

        if self.percentage is not None:
            self.time_percentage()

    def time_percentage(self):
        keys = ["Measurement", "Groundtruth", "Control"]

        ti = np.inf
        tf = 0.0

        for key in keys:
            d = self.data[key]
            ti = min(ti, d[0][0])
            tf = max(tf, d[-1][0])

        tf2 = ti + self.percentage * (tf - ti)

        for key in keys:
            d = self.data[key]
            df = pd.DataFrame(d)


            idx = df[0] < tf2
            data = df[idx].to_numpy()

            self.data[key] = data

            pass

def random_color():
    # https://www.statology.org/matplotlib-random-color/
    col = (np.random.random(), np.random.random(), np.random.random())
    return col

def plotbirdseye(prefix, label, x, y, theta, fig2=None, nb_pts = 20, width = 0.005):
    if fig2 is None:
        fig2 = plt.figure()
        c1 = "purple"
        c2 = "orange"
    
    else:
        c1 = random_color()
        c2 = c1 # random_color()

    # n = int(x.shape[0] / nb_pts)
    n = x.shape[0]

    # x2 = x[::n]
    # y2 = y[::n]
    # theta2 = theta[::n]
    x2 = x
    y2 = y
    theta2 = theta
    dx = .02 * np.cos(theta2)
    dy = .02 * np.sin(theta2)
    plt.scatter(x2, y2, color=c1, marker='o', zorder=3, label=label) # ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    for i in range(0, dx.shape[0], 3): # decimate
        plt.arrow(x2[i], y2[i], dx[i], dy[i], color=c2, zorder=2, width=width)
    pass

    plt.title(prefix + ": birds-eye view")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    return fig2

def run(mm, cmds):
    # iterate over commands
    l = len(cmds)
    freq = 10
    c = 0
    for cmd in cmds:
        # propagate the motion model w.r.t. the cmd
        mm.step(cmd)

        c += 1
        if VERBOSE and (c%freq) == 0:
            print("{} of {}".format(c, l))

# def wrap(angles):
#     # wrap to [-pi, pi]
#     b1 = angles > math.pi
#     angles[b1] %= PI2

#     b2 = angles < -math.pi
#     angles[b2] %= -PI2
#     # angles[b2] = -1.0 * ( (-1.0 * angles[b2]) % math.pi)

#     pass


def q2(dataloader):
    # setup cmds, of the form (total-time, v, w). Initialize state with all zeros.
    cmds = [
        [1.0, 0.5, 0.0],
        [2.0, 0.0, -1./(PI2)],
        [3.0, 0.5, 0.0],
        [4.0, 0, 1./PI2],
        [5.0, 0.5, 0.0]
    ]

    # setup motion model
    mm = SimpleMotionModel()

    run(mm, cmds)

    # extract the history
    h = mm.get_history()
    t = h[:, 0]
    x = h[:, 1]
    y = h[:, 2]
    theta = h[:, 3]
    wrap(theta)

    h[:, 3] = theta

    # "report the values of any parameters"
    print("q2: dt used: {}".format(mm.dt))

    # plotting
    nb_non_time_state_vars = 3 # h.shape[1] - 1
    names = ["x", "y", "theta"]

    # plot 1 - t vs all state variables
    fig1, axes = plt.subplots(nb_non_time_state_vars)
    
    # iterate over all non-time state variables
    for i in range(nb_non_time_state_vars):
        axes[i].plot(t, h[:, i+1])
        axes[i].set_ylabel(names[i])

    plt.suptitle("q2: Time History of State Variables")
    plt.xlabel("Time (s)")
    fig1.show()

    # plot 2 - birds-eye view: x vs y
    fig2 = plotbirdseye("q2", "Dead Reckoning", x, y, theta)
    fig2.show()

def q3(dataloader):
    cmds = dataloader.data["Control"]
    gt = dataloader.data["Groundtruth"]

    # decimate for quicker plotting
    gt = gt[::10]

    if TEST:
        n = 2000
        cmds = cmds[:n]
        gt = gt[:n]

    # setup motion model
    mm = SimpleMotionModel(gt[0])

    run(mm, cmds)

    """ Plotting """

    # extract the history
    h = mm.get_history()
    t = h[:, 0]
    x = h[:, 1]
    y = h[:, 2]
    theta = h[:, 3]

    # wrap(theta) # in place
    # h[:, 3] = theta

    # extract g.t.
    gt_t = gt[:, 0]
    gt_x = gt[:, 1]
    gt_y = gt[:, 2]
    gt_theta = gt[:, 3]

    # unwrap for plotting
    gt_theta = np.unwrap(gt_theta)
    gt[:, 3] = gt_theta

    # "report the values of any parameters"
    print("q3: dt used: {}".format(mm.dt))

    # plotting
    nb_non_time_state_vars = 3 # h.shape[1] - 1
    names = ["x", "y", "theta"]

    # plot 1 - t vs all state variables
    fig1, axes = plt.subplots(nb_non_time_state_vars)
    
    # iterate over all non-time state variables
    for i in range(nb_non_time_state_vars):
        axes[i].plot(t, h[:, i+1], color="purple", alpha=0.5)
        axes[i].set_ylabel(names[i])

        # gt
        axes[i].plot(gt_t, gt[:, i+1], color="orange", alpha=0.5)

        axes[i].legend(["Dead Reckoning", "Ground Truth"])

    plt.xlabel("Time (s)")
    plt.suptitle("q3: Time History of State Variables")
    fig1.show()

    # plot 2 - birds-eye view: x vs y
    nb_pts = 400
    width = 0.05
    fig2 = plotbirdseye("q3", "Dead Reckoning", x, y, theta, nb_pts=nb_pts, width=width)

    # add on ground truth
    fig2 = plotbirdseye("q3", "Ground Truth", gt_x, gt_y, gt_theta, fig2=fig2, nb_pts=nb_pts, width=width)

    plt.legend()

    fig2.show()

    # metrics
    # TODO: calculate MSE (must create time-aligned data-frames, can do so easily using pandas) 


def q6(dataloader):
    # inputs
    cases = [
        [2, 3, 0, 6],
        [0, 3, 0, 13],
        [1, -2, 0, 17]
    ]

    for case in cases:
        s = case[0:3]
        id = case[3]
        mm = SimpleMeasurementModel(dataloader, s)

        range, heading = mm.landmark_transform(s[0:2], s[2], id)
        print("Landmark #{}, with respect to the robot at {} m, {} m, {} rad. Range: {:.2f} m, heading: {:.2f} rad".format(id, s[0], s[1], s[2], range, heading))

def q8():
    percent = 1.0
    dataloader = DataLoader(TEST, percentage=percent)
    # q2 commands
    # setup cmds, of the form (total-time, v, w). Initialize state with all zeros.
    cmds = [
        [1.0, 0.5, 0.0],
        [2.0, 0.0, -1./(PI2)],
        [3.0, 0.5, 0.0],
        [4.0, 0, 1./PI2],
        [5.0, 0.5, 0.0]
    ]
    meass = []

    # particle filter
    nb_particles = 200

    # [self.t, self.x, self.y, self.theta, self.phi_1, self.phi_2]
    
    # n between 0 and 1e-2 for reasonable values
    n = 1e-2
    m = 1e-3
    dt = 1e-3

    # "report the values of any parameters"
    print("q8 with q2 cmds: motion var {}, meas var {}, max dt {}".format(n, m, dt))

    motion_variance = [0.0, n, n, n, 0.0, 0.0]
    meas_variance = [m, m]

    if TEST:
        nb_particles = 100

    ### Q2 commands
    pf = ParticleFilter(dataloader, nb_particles)
    pf.initialize(np.zeros_like(dataloader.data["Groundtruth"][0]), None, motion_variance, meas_variance, dt)

    # run
    pf.run(cmds, meass)

    # plotting
    pf.plot('q8 with q2 cmds, avg of particles')
    pf.plot_xy('q8 with q2 cmds, all particles')


    ### q3 commands
    dt = 1.0 # for faster execution
    cmds = dataloader.data["Control"]
    meass = dataloader.data["Measurement"]
    gt = dataloader.data["Groundtruth"]
    
    # decimate
    gt = gt[::2]

    # "report the values of any parameters"
    print("q8 with q3 cmds: motion var {}, meas var {}, max dt {}".format(n, m, dt))

    pf = ParticleFilter(dataloader, nb_particles)
    pf.initialize(gt[0], None, motion_variance, meas_variance, dt)

    # run
    pf.run(cmds, meass)

    # plotting
    pf.plot('q8 with q3 cmds, avg of particles', gt)
    pf.plot_xy('q8 with q3 cmds, all particles', gt)

def q9():
    # only use part of the dataset so we can sweep noise values without taking too long to execute
    percentage = 0.2
    dataloader = DataLoader(TEST, percentage=percentage)

    cmds = dataloader.data["Control"]
    meass = dataloader.data["Measurement"]
    gt = dataloader.data["Groundtruth"]

    # decimate the cmds for faster execution
    n = 10
    sl = slice(0, -1, n)
    cmds = cmds[sl]

    ns = [1e-1, 1e-2, 1e-3]
    ms = [1e-2, 5e-2, 5e-3]
    dt = 1.0
    nb_particles = 10
    
    c = 0
    for n, m in zip(ns, ms):
        c += 1
        # "report the values of any parameters"
        print("q9 run {}: motion var {}, meas var {}, max dt {}".format(c, n, m, dt))
        motion_variance = [0.0, n, n, 1.0, 0.0, 0.0]
        meas_variance = [m, m]

        if TEST:
            nb_particles = 100

        pf = ParticleFilter(dataloader, nb_particles)
        pf.initialize(gt[0], None, motion_variance, meas_variance, dt)

        # run
        tic = time.time()
        pf.run(cmds, meass)
        toc = time.time()
        print("runtime: {:.2f}".format(toc - tic))

        # plotting
        pf.plot('q9 with q3 cmds, avg of particles, motion var: {}, meas var: {}'.format(n, m), gt)
        pf.plot_xy('q9 with q3 cmds, avg of particles, motion var: {}, meas var: {}'.format(n, m), gt)



def main():
    dl = DataLoader()

    # # run q2
    # q2(dl)

    # # # run q3
    # q3(dl)

    # # # run q6
    # q6(dl)

    # # run q8
    q8()

    # # run q9
    # q9()

    input("press any key")




if __name__ == "__main__":
    main()