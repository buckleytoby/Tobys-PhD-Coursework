# "Allowable python packages are: numpy, matplotlib, pandas, scipy, seaborn, and default packages (such as math, time, etc.)."
import os, sys, math, time, collections, csv, re, copy, queue

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import scipy
import scipy.stats
import scipy.signal

PI2 = 2.0 * math.pi
TEST = False
VERBOSE = True

def save(fig, label):
    # https://stackoverflow.com/questions/7986567/matplotlib-how-to-set-the-current-figure
    plt.figure(fig.number)

    # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
    plt.savefig('./outputs/' + label + '.png', dpi=300, bbox_inches='tight')

def wrap(angle):
    # ref: https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    angle2 = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle2

def real_to_grid(x, grid_width):
    gx = int(np.floor(x / grid_width))
    return gx


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
        return [self.t, self.x, self.y, self.theta]
    
    def unpack_state_vector(self):
        self.t, self.x, self.y, self.theta = self.state
    
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

# http://asrl.utias.utoronto.ca/datasets/mrclam/index.html#Details
class DataLoader:
    def __init__(self,
                 test=TEST,
                 max_nb_pts = None,
                 percentage = 1.0
                 ):
        self.test = test
        self.max_nb_pts = max_nb_pts
        self.percentage = percentage

        # params
        dir = "./ds1"
        prefix = "ds1"
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

    def __getitem__(self, idx):
        return self.data[idx]

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

        c1 = "purple"

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


class MotionModel:
    def __init__(self, state0=None, dt = 1e-3, variance=None):
        if state0 is None:
            t0, x0, y0, theta0 = 0.0, 0.0, 0.0, 0.0
        else:
            t0, x0, y0, theta0 = state0
        # params
        self.dt = dt
        self.variance = variance
        
        # setup state variables
        self.t = t0
        self.x = x0
        self.y = y0
        self.theta = theta0

        self.state = np.array(self.bundle_state_vector())

        # noise
        if self.variance is not None:
            assert(len(self.variance) == len(self.state))
            self.gaus = scipy.stats.multivariate_normal(
                mean=np.zeros_like(self.state),
                cov=np.diag(self.variance), #type:ignore
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

        dtheta = w
        dx = v * math.cos(self.theta)
        dy = v * math.sin(self.theta)

        grad = np.array([dt, dx, dy, dtheta])
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

        assert(not np.isnan(v))
        assert(not np.isnan(w))

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
        
    def bundle_state_vector(self):
        return [self.t, self.x, self.y, self.theta]
    
    def unpack_state_vector(self):
        self.t, self.x, self.y, self.theta = self.state
        
    def add_history(self):
        self.history.add_history(self.step_count, self.bundle_state_vector())

    def get_history(self):
        return self.history.get_history()
    
    def set_state(self, state):
        assert(not np.any(np.isnan(state)))
        self.state = state

        # save state to individual vars
        self.unpack_state_vector()

    def get_state(self):
        return self.state
    
    def plot_history_xy(self, prefix="Robot History", fig2 = None):
        h = np.array(self.get_history())

        # extract the history
        t = h[:, 0]
        x = h[:, 1]
        y = h[:, 2]
        theta = h[:, 3]

        # plot 2 - birds-eye view: x vs y
        width = 0.005
        my_label = "Birds-eye-view"
        fig2 = plotbirdseye(prefix, "_nolegend_", x, y, theta, width=width, fig2 = fig2)
        fig2.show()
        
    def plot_history(self, prefix="Robot History", gt=None, fig=None):

        h = self.get_history()

        # extract the history
        t = h[:, 0]
        x = h[:, 1]
        y = h[:, 2]
        theta = h[:, 3]

        # plotting
        nb_non_time_state_vars = 3 # h.shape[1] - 1
        names = ["x", "y", "theta"]

        # plot 1 - t vs all state variables
        fig1, axes = plt.subplots(nb_non_time_state_vars)
        
        # iterate over all non-time state variables
        for i in range(nb_non_time_state_vars):
            axes[i].plot(t, h[:, i+1], color="purple", alpha=0.5)
            axes[i].set_ylabel(names[i])

        plt.xlabel("Time (s)")
        plt.suptitle(prefix + ": Time History of State Variables")
        fig1.show()
    


############################## new stuff
def contains(xy, xymin, xymax):
    x, y = xy
    x1, y1 = xymin
    x2, y2 = xymax

    c = (x > x1) and (x < x2) and (y > y1) and (y < y2)
    return c

class Cell:
    def __init__(self, x, y, width = 1.0, neighbors = [], unexplored = False):
        self.width = width
        self.height = self.width
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.occupied = False
        self.unexplored = unexplored

    def get_rect(self):
        return (self.x, self.y, self.width, self.height)
    
    def set_occupied(self, b):
        self.occupied = b

    def get_neighbors(self):
        return self.neighbors
    
    def cost(self, cell):
        # scale to the cell width so I can debug more easily

        d = self.width
        # if it's not occupied, or hasn't yet been explored
        if not self.occupied or self.unexplored:
            return 1.0 * d
            
        else:
            return 1000.0 * d
        
    def __lt__(self, other):
        # used as a tie-breaker
        return self
    
    @property
    def xy(self):
        return np.array([self.x, self.y])
    
    def equal(self, other):
        assert(isinstance(other, Cell))
        return np.all(np.equal(self.xy, other.xy))
    
    @property
    def x1(self):
        return self.x
    
    @property
    def x2(self):
        return self.x + self.width
    
    @property
    def y1(self):
        return self.y
    
    @property
    def y2(self):
        return self.y + self.height
    
    @property
    def xy1(self):
        return [self.x1, self.y1]
    
    @property
    def xy2(self):
        return [self.x2, self.y2]
    
    def contains(self, x, y, object_radius = None):
        c = contains([x, y], self.xy1, self.xy2)

        if object_radius is not None:
            n = np.linalg.norm(np.array([x, y] - self.center))
            if n < object_radius:
                return True

        return c
    
    @property
    def center(self):
        return np.array([self.x1 + 0.5 * self.width, self.y1 + 0.5 * self.height])

    def copy(self):
        """
        no neighbors in the copy
        """
        c = Cell(self.x, self.y, self.width, unexplored=self.unexplored)

        return c
    
    def dist(self, other):
        delta = self.center - other.center
        d = np.linalg.norm(delta)
        return d


DL = DataLoader(TEST)

class Map:
    def __init__(self,
                 cell_width = 1.0,
                 object_width = None
                 ):
        self.cells = {}
        self.cell_width = cell_width
        self.object_width = object_width

        self.landmarks = DL["Landmark_Groundtruth"]

    def set_unexplored(self):
        for key, cell in self.cells.items():
            cell.unexplored = True

    def plot(self, f=None):
        if f is None:
            f = plt.figure()

        plt.xlim(self.minx, self.maxx + self.cell_width)
        plt.ylim(self.miny, self.maxy + self.cell_width)

        # plt.xticks(self.xrange)
        # plt.yticks(self.yrange)

        xrange, yrange = get_range(1.0)
        plt.xticks(xrange)
        plt.yticks(yrange)
        plt.grid(visible=True)

        ax = plt.gca()
        # https://stackoverflow.com/questions/17990845/how-do-i-equalize-the-scales-of-the-x-axis-and-y-axis
        ax.set_aspect('equal', adjustable='box')


        cell: Cell
        for key, cell in self.cells.items():
            if cell.occupied and not cell.unexplored:
                x, y = cell.xy
                # ref: https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
                rect = patches.Rectangle((x, y), self.cell_width, self.cell_width, fill=True, color='g')
                ax.add_patch(rect)

        return f

    def make_neighbors(self, cell: Cell): #  xr, yr):
        xr, yr = cell.xy
        n = []

        # loop vars
        x = xr - self.cell_width
        # for x in range(xr - self.cell_width, xr + self.cell_width, self.cell_width):
            # for y in range(yr - self.cell_width, yr + self.cell_width, self.cell_width):
        while x < xr + self.cell_width + 1e-3:
            y = yr - self.cell_width
            while y < yr + self.cell_width + 1e-3:
                if not np.all(np.equal([x, y], [xr, yr])):
                    # if contains([x, y], [self.minx, self.miny], [self.maxx, self.maxy]):
                    c = self.get_cell(x, y)
                    if c is not None:
                        n.append(c)

                y += self.cell_width
            x += self.cell_width

        return n


    def make_cell(self, xr, yr):
        cell = Cell(xr, yr, self.cell_width)

        # if occupied
        self.is_occupied(cell)

        self.set_cell(xr, yr, cell)

    def is_occupied(self, cell: Cell):
        """
        xr - covers xr to xr+1. ex: xr = 3 -> cell covers 3-4

        """

        # ex:
        xys = self.landmarks[:, 1:3] # all landmarks, only x, y
        for x, y in xys:
            if cell.contains(x, y, self.object_width):
                cell.set_occupied(True)

    def set_cell(self, xr, yr, cell):
        xg = real_to_grid(xr, self.cell_width)
        yg = real_to_grid(yr, self.cell_width)

        key = (xg, yg)
        assert(key not in self.cells)
        self.cells[(xg, yg)] = cell

    def get_cell(self, xr, yr) -> Cell | None:
        xr = real_to_grid(xr, self.cell_width)
        yr = real_to_grid(yr, self.cell_width)

        t = (xr, yr)

        if t in self.cells:
            return self.cells[t]
        else:
            return None
        
    def get_random_cell(self):
        i = np.random.randint(0, len(self.cells))
        k = list(self.cells.keys())[i]

        return self.cells[k]

    def init(self, range_):
        xrange, yrange = range_
        self.xrange = xrange
        self.yrange = yrange
        self.minx = xrange[0]
        self.maxx = xrange[-1]
        self.miny = yrange[0]
        self.maxy = yrange[-1]

        c = 0
        for xr in xrange:
            for yr in yrange:
                self.make_cell(xr, yr)
                c += 1
                # if c%5==0:
                #     print(c)
                
        print("nb cells: {}".format(len(self.cells.keys())))

        self.make_all_neighbors()

    def make_all_neighbors(self):
        # for xr in self.xrange:
        #     for yr in self.yrange:
        c: Cell = None #type:ignore
        for key, c in self.cells.items():
            xr, yr = c.xy
            n = self.make_neighbors(c) # xr, yr)

            if c is None:
                print("something went wrong.")
                raise
            
            if len(n) == 0:
                print("something went wrong.")
                raise

            # debug
            if key == (23, -34):
                pass

            # set neighbors set_neighbors
            c.neighbors = n

    def copy(self):
        """
        for a copy (used for state history), we just need the cells and whether they were explored, no neighbors needed
        """
        m = Map(self.cell_width, self.object_width)

        m.cells = {key: cell.copy() for key, cell in self.cells.items()}

        m.minx = self.minx
        m.maxx = self.maxx
        m.miny = self.miny
        m.maxy = self.maxy
        m.xrange = self.xrange
        m.yrange = self.yrange

        return m


class Node(Cell):
    pass

class Astar:
    def __init__(self):
        self.map = None
        self.s = None
        self.g = None

        # members
        self.plan = None

    def set_start(self, s: Node | Cell):
        self.s = s

    def set_goal(self, g: Node):
        self.g = g

    def set_map(self, map: Map):
        self.map = map

    def plan_(self) -> list[Node] | None:
        # setup
        s: Node = self.s #type:ignore
        g: Node = self.g #type:ignore
        node = Node(0, 0)

        assert(s is not None)
        assert(g is not None)
        
        # setup algorithm variables
        open_set = queue.PriorityQueue() # know this: lowest values are popped first
        open_set_nodes = {}
        closed_set = {}
        cost_hash = {}
        parent = {}

        def put(t):
            open_set.put(t)
            open_set_nodes[t[1]] = True

        # insert start into open_set
        put((0.0, s))
        cost_hash[s] = 0.0
        parent[s] = None

        # euclidean distance
        def heuristic(node: Node):
            dx, dy = np.abs(node.center - g.center)

            if dy < dx:
                dz = dy
                dy = dx
                dx = dz

            # now, dx is less than dy, so take dx diagonal moves
            h = dx

            # now take dy - dx straight moves
            h += (dy - dx)

            return h

        done = False
        print("A*: begin planning...")
        c = 0
        while not done:
            # get most promising node
            if not open_set.empty():
                prio, node = open_set.get()
            else:
                break
            closed_set[node] = True

            d = heuristic(node)

            c += 1
            if c%100==0:
                print("Evaluated {} nodes.".format(c))
                print("node ", node.center, " goal: ", g.center)

            # check if it's goal
            if node.equal(g):
                break

            # get neighbors
            neighbor_nodes = node.get_neighbors()
            neighbor_node: Node

            # iterate through neighbors
            for neighbor_node in neighbor_nodes:
                # skip a neighbor if it's in the closed set or already in the open set
                if neighbor_node in closed_set or neighbor_node in open_set_nodes:
                    continue

                # add to cost hash
                cost_hash[neighbor_node] = cost_hash[node] + neighbor_node.cost(node)

                # add parent
                parent[neighbor_node] = node

                # calculate cost
                f = cost_hash[neighbor_node] + heuristic(neighbor_node)

                # put into the open-set
                put((f, neighbor_node))
            pass

        if not node.equal(g):
            print("A* couldn't find a plan")
            return None
        else:
            print("A* plan found.")

        # get final plan by traversing from the goal up the tree
        plan = [node] # using the last node from the planning loop

        while True:
            node = parent[node]
            plan.append(node)

            if node.equal(s):
                break

        # reverse plan to go from start to goal
        plan.reverse()

        self.plan = plan
        return plan
    
    def plot(self, f=None):
        if f is None:
            f = plt.figure()

        if self.plan is None:
            return

        for i in range(len(self.plan) - 1):
            node: Cell = self.plan[i]
            node2: Cell = self.plan[i+1]

            xy1 = node.center
            xy2 = node2.center

            x, y = xy1
            dx, dy = xy2 - xy1

            xys = np.array([xy1, xy2])

            plt.arrow(x, y, dx, dy, width=0.05, color='orange')
    
    def copy(self):
        """
        only need to copy stuff for plotting
        """
        a = Astar()

        if self.plan is not None:
            a.plan = [node.copy() for node in self.plan]
        
        return a



    
def range_bearing(xy1, theta, xy2):
    """
    range and bearing going FROM xy1 TO xy2.
    so xy1 should be the robot, xy2 should be the landmark
    """
    xy1 = np.array(xy1)
    xy2 = np.array(xy2)
    s = xy2 - xy1
    range = np.linalg.norm(s)

    # output in [-pi, pi]
    theta_landmark = np.arctan2(s[1], s[0])

    # wrap to [-pi, pi]
    theta2 = wrap(theta)

    # heading is FROM the robot TO the landmark
    heading = theta_landmark - theta2

    return range, heading

NNN = 1e-2
class Robot:
    def __init__(self,
                 mm: MotionModel = MotionModel(state0=[0.0, 0.0, 0.0, -np.pi/2.0], variance=[0.0, NNN, NNN, NNN]),
                 max_accel = 0.288,
                 max_alpha = 5.579,
                 dt = 0.1,
                 v0 = 0.0,
                 w0 = 0.0,
                 kv = 0.5,
                 kw = 0.5,
                 obs_range = 1.0,
                 ):
        self.mm = mm
        self.obs_range = obs_range # m
        self.max_accel = max_accel
        self.max_alpha = max_alpha
        self.dt = dt
        self.kv = kv
        self.kw = kw
        self.radius = 0.5

        self.meas_model = None

        # state vars
        self.v = v0
        self.w = w0

    def plot(self, f=None):
        if f is None:
            f = plt.figure()

        ax = plt.gca()
        circle = patches.Circle(tuple(self.xy), self.radius, fill=True, color=random_color())
        ax.add_patch(circle)
        
    def plot_history(self, f=None):
        self.mm.plot_history(fig=f)
        self.mm.plot_history_xy(fig2=f)
        
    def plot_history_xy(self, f=None):
        self.mm.plot_history_xy(fig2=f)

    def get_state(self) -> np.ndarray:
        return self.mm.get_state()[1:4]
    
    def get_grid_xy(self, grid_width) -> tuple[int, int]:
        x, y, _ = self.get_state()

        gx = real_to_grid(x, grid_width)
        gy = real_to_grid(y, grid_width)

        return gx, gy
    
    def get_obs(self):
        x, y, _ = self.get_state()
        return x, y, self.obs_range
    
    def controller(self, G) -> tuple[float, float]:
        gx, gy = G.center

        # simple proportional controller
        x, y, theta = self.get_state() #type:ignore
        
        range, heading = range_bearing([x, y], theta, [gx, gy])
        
        # check if rotating the opposite way is shorter
        rot_ccw = heading + 2 * np.pi
        rot_cw = heading - 2 * np.pi
        if abs(rot_ccw) < abs(heading):
            heading = rot_ccw
        if abs(rot_cw) < abs(heading):
            heading = rot_cw

        v = self.kv * range
        w = self.kw * heading

        return v, w #type:ignore
    
    @property
    def xy(self):
        return self.get_state()[0:2]
    
    def reset(self, node):
        # always reset to zero time and -pi/2 theta
        x, y = node.center
        state = [0.0, x, y, -np.pi/2.0]
        
        self.mm = MotionModel(state0=state, variance=[0.0, NNN, NNN, NNN])

    def reset_xyth(self, x, y, theta):
        state = [0.0, x, y, theta]
        
        self.mm = MotionModel(state0=state, variance=[0.0, NNN, NNN, NNN])

    def step(self, G: Node):
        """
        state vars: (x, y, th)
        cmd vars: (v, w)
        """
        v, w = self.controller(G)

        self.step_vw(v, w)

    def step_vw(self, v, w):
        # delta-cmd, diff between desired and current
        vd = v - self.v
        wd = w - self.w

        # enforcing acceleration constraints
        def enforce(a, maxaa):
            return np.sign(a) * min(abs(a), maxaa * self.dt)

        vd = enforce(vd, self.max_accel)
        wd = enforce(wd, self.max_alpha)

        # update vel
        self.v += vd
        self.w += wd
        
        # take a step in the mm
        t = self.mm.t + self.dt
        cmd = [t, self.v, self.w]
        self.mm.step(cmd)


    def drive(self, plan: list[Node], reset=True, eps=5e-1):
        """
        drive through a plan
        """
        
        # assertions
        # assert(len(plan) > 1) # at least 2 nodes, start and goal
        
        # method vars
        done = False
        idx = 0
        
        # get first node
        node = plan[idx]
        goal = plan[-1]
        
        # reset robot
        if reset:
            self.reset(node)
        
        print("Starting from {}".format(self.xy))
        print("Driving to {}".format(plan[-1].center))
        count = 0
        while not done:
            # check proximity
            dist = np.linalg.norm(self.xy - node.center) #type:ignore
            close_enough = dist < eps
                
            # check for goal
            if node == goal and close_enough:
                # we're done
                break
            
            # close enough and not at the goal --> next node
            if close_enough:
                # move to next node
                idx += 1
                node = plan[idx]
                print("New goal: {}".format(node.center))
                
            # diverging
            if dist > 2.0:
                pass
                
            # drive to node
            self.step(node)
            
            # progress
            count += 1
            if count%100 == 0:
                print("dist: {}".format(dist))
        print("Done driving to {}".format(plan[-1].center))
        
        pass

class GridRobot(Robot):
    def __init__(self,
                 x0 = 0.0,
                 y0 = 0.0,
                 obs_range = 1.0,
                 ):
        super().__init__(obs_range=obs_range)

        self.x = x0
        self.y = y0



    
    def get_obs(self):
        x, y = self.get_state()
        return x, y, self.obs_range
    
    def step(self, G: Node):
        """
        G - goal
        grid robot achieves its goal automatically
        """
        x, y = G.center
        self.x = x
        self.y = y

    def get_state(self):
        return np.array([self.x, self.y])
    
    @property
    def xy(self):
        return self.get_state()
    
    def set_xy(self, xy):
        self.x, self.y = xy
        
    def reset(self, node):
        x, y = node.center
        
        self.set_xy(node.xy)
        
    def drive(self, plan, reset=False, eps=0.0):
        # go directly to final location
        n = plan[-1]
        self.step(n)
        
    def plot_history_xy(self, f=None):
        """
        no history for grid robot, only current xy
        """
        self.plot(f=f)



class OnlinePlanning:
    def __init__(self,
                 true_map: Map = None, #type:ignore
                 planner: Astar = None, #type:ignore
                 robot: Robot = None, #type:ignore
    ):
        self.true_map = true_map
        self.planner = planner
        self.robot = robot

        self.setup()

    def setup(self):
        # compute the sensor indices once, since they're repeatedly used
        _, _, r = self.robot.get_obs()

        w = self.true_map.cell_width



        indices = []
        nx = int((2* r) / w) + 1
        for x in np.linspace(-r, r, nx):
            for y in np.linspace(-r, r, nx):
                n = np.linalg.norm([x, y])
                if True:
                    gx = real_to_grid(x, w)
                    gy = real_to_grid(y, w)

                    indices.append([gx, gy])

        self.indices = np.array(indices)

    def observe(self):
        # get robot's x, y, and sensor range
        x, y, r = self.robot.get_obs()

        # to grid
        gx = real_to_grid(x, self.true_map.cell_width)
        gy = real_to_grid(y, self.true_map.cell_width)

        indices = copy.deepcopy(self.indices)

        # add on the offset
        indices[:, 0] += gx
        indices[:, 1] += gy

        # observe each index
        for gx, gy in indices:
            n = self.true_map.get_cell(gx, gy)
            if n is not None:
                n.unexplored = False

        pass

    def run(self, start: Node, G: Node):
        S = start
        # set up an online run
        self.robot.reset(S)
        map: Map = self.true_map # copy.deepcopy(self.true_map)
        map.set_unexplored()
        self.planner.set_goal(G)

        # outputs
        self.h = []

        def save_state():
            h_map = self.true_map.copy()
            h_plan = self.planner.copy()
            self.h.append((h_map, copy.deepcopy(self.robot), h_plan))

        # save initial state
        save_state()

        # loop vars
        print("run online planner")
        done = False
        c = 0
        while not done:
            if c%50==0:
                print("online planner status", S.xy, G.xy)
            # construct S
            x, y = self.robot.xy
            S = map.get_cell(x, y)

            if S is None:
                print("Something went wrong...")
                raise

            # check if done
            if S.equal(G):
                done = True
                break

            # update the start
            self.planner.set_start(S)

            # plan
            plan = self.planner.plan_()

            if plan is None:
                raise

            # take the first non-start action -- idx = 1
            # a = self.get_action(plan[1])
            a = plan[1]

            # execute
            # eps should be less than half the cell width
            eps = self.true_map.cell_width / 4.0
            self.robot.drive([a], reset=False, eps=eps)

            # take an observation
            self.observe()

            # save the state
            save_state()

        return self.h


def q3_states_and_goals():
    sg = []
    S=[0.5, -1.5]
    G=[0.5, 1.5]
    # S = Node(S[0], S[1])
    # G = Node(G[0], G[1])
    sg.append([S, G])

    # B
    S=[4.5, 3.5]
    G=[4.5, -1.5]
    # S = Node(S[0], S[1])
    # G = Node(G[0], G[1])
    sg.append([S, G])

    # C
    S=[-0.5, 5.5]
    G=[1.5, -3.5]
    # S = Node(S[0], S[1])
    # G = Node(G[0], G[1])
    sg.append([S, G])


    return sg

def get_range(cell_width = 1.0):
    eps = 1e-3
    xlim = np.array([-2 + eps, 5 + eps])
    ylim = np.array([-6 + eps, 6 + eps])
    nx = int((xlim[1] - xlim[0]) / cell_width) + 1
    ny = int((ylim[1] - ylim[0]) / cell_width) + 1
    xrange = np.linspace(xlim[0], xlim[1], nx)
    yrange = np.linspace(ylim[0], ylim[1], ny)
    range_ = (xrange, yrange)
    return range_

def q3():
    map = Map()
    map.init(get_range())

    astar = Astar()
    astar.set_map(map)

    c = 0
    def run(S, G):
        nonlocal c
        c += 1
        # executing
        astar.set_start(S)
        astar.set_goal(G)
        p = astar.plan_()

        # plot map
        f = map.plot()

        # plotting
        astar.plot(f)

        save(f, "q3: #" + str(c))

        plt.show()


    sg = q3_states_and_goals()

    for i in range(len(sg)):
        S, G = sg[i]
        S, G = map.get_cell(*S), map.get_cell(*G)
        run(S, G)

def plot_history(history, f=None, axs=None, n=1):
    if f is None:
        f = plt.figure()

    # m = max(1, len(history) / (n*n) + 1)

    r = np.linspace(0, len(history)-1, n*n, dtype='int')

    # r = range(0, len(history), m)

    map: Map = None #type:ignore
    robot: Robot = None #type:ignore
    planner: Astar = None #type:ignore
    c=-1
    for i in r:
        c += 1
        map, robot, planner = history[i]

        # plt.subplot(n, n, c)
        assert(axs is not None)
        plt.sca(axs.flatten()[c])

        map.plot(f)
        planner.plot(f)
        # robot.plot(f)
        robot.plot_history_xy(f)

        plt.title("Step {}".format(i))

    f.tight_layout()
    pass


def q5():
    map = Map()
    map.init(get_range())

    astar = Astar()
    robot = GridRobot()
    
    online_planner = OnlinePlanning(true_map=map, planner=astar, robot=robot)


    # runs
    c = 0
    def run(S, G):
        nonlocal c
        c += 1
        history = online_planner.run(S, G)
        
        # plots
        n = int(np.ceil(np.sqrt(len(history))))
        n = min(n, 3) # max 3x3 
        f, axs = plt.subplots(n, n)

        plot_history(history, f, axs, n)

        save(f, "q5: #" + str(c))

        plt.show()

        pass

    sg = q3_states_and_goals()

    for i in range(len(sg)):
        S, G = sg[i]
        S, G = map.get_cell(*S), map.get_cell(*G)
        run(S, G)



def q7_states_and_goals():
    sg = []
    S=[2.45, -3.55]
    G=[0.95, -1.55]
    sg.append([S, G])

    # B
    S=[4.95, -0.05]
    G=[2.45, 0.25]
    sg.append([S, G])

    # C
    S=[-0.55, 1.45]
    G=[1.95, 3.95]
    sg.append([S, G])

    return sg

def q7(plot=True):
    cell_width = 0.1
    obstacle_inflation = 0.3 # m

    range_ = get_range(cell_width)
    map = Map(cell_width=cell_width, object_width=obstacle_inflation)
    map.init(range_)

    astar = Astar()
    astar.set_map(map)
    
    c = 0
    def run(S, G):
        nonlocal c
        c += 1
        # executing
        astar.set_start(S)
        astar.set_goal(G)
        p = astar.plan_()

        assert(p is not None)

        if plot:
            # plot map
            f = map.plot()

            # plotting
            astar.plot(f)

            save(f, "q7: #" + str(c))

            plt.show()

        return p

    sg = q7_states_and_goals()

    plans: list[list[Node]] = []
    for i in range(len(sg)):
        S, G = sg[i]
        S, G = map.get_cell(*S), map.get_cell(*G)
        plans.append(run(S, G))

    return plans, map, astar

def q9():
    robot = Robot()

    plans, map, astar = q7(plot=False)

    c = 0
    for p in plans:
        c += 1
        robot.drive(p, eps=0.05) # eps half the cell width for q7
    
        # plot map
        f = map.plot()

        # plotting
        astar.plan = p
        astar.plot(f)
        
        # plot robot
        robot.plot_history_xy(f)

        save(f, "q9: #" + str(c))
        

def q10():
    # assume we still use q7's states and goals
    sg = q7_states_and_goals()
    S, G = sg[0]

    map = Map()
    map.init(get_range())

    astar = Astar()
    robot = Robot()
    
    online_planner = OnlinePlanning(true_map=map, planner=astar, robot=robot)

    # runs
    c = 0
    def run(S, G):
        nonlocal c
        c += 1
        history = online_planner.run(S, G)
        
        # plots
        n = int(np.ceil(np.sqrt(len(history))))
        n = min(n, 3) # max 3x3
        f, axs = plt.subplots(n, n)

        plot_history(history, f, axs, n)

        save(f, "q10: #" + str(c))

        plt.show()

        pass

    for i in range(len(sg)):
        S, G = sg[i]
        S, G = map.get_cell(*S), map.get_cell(*G)
        run(S, G)

def q11():
    sg = q3_states_and_goals()
    S, G = sg[0]

    # runs
    c = 0
    def run(S, G, cell_width):
        nonlocal c
        c += 1

        astar = Astar()
        robot = Robot()
        
        online_planner = OnlinePlanning(true_map=map, planner=astar, robot=robot)
        history = online_planner.run(S, G)
        
        # plots
        n = int(np.ceil(np.sqrt(len(history))))
        n = min(n, 3)
        f, axs = plt.subplots(n, n)

        plot_history(history, f, axs, n)
        

        save(f, "q11: #" + str(c))

        plt.show()

        pass

    # iterate over cell widths
    for cell_width in [1.0, 0.1]:
        
        # iterate over q3 goals
        for i in range(len(sg)):
            
            # make the map
            map = Map(cell_width=cell_width)
            map.init(get_range(cell_width=cell_width))
            
            S, G = sg[i]
            S, G = map.get_cell(*S), map.get_cell(*G)
            
            # do the planning
            run(S, G, map)