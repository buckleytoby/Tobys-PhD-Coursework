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
                 percentage = None
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

        dtheta = w
        dx = v * math.cos(self.theta)
        dy = v * math.sin(self.theta)

        grad = np.array([dt, dx, dy, dtheta, 0.0, 0.0])
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
        
    def bundle_state_vector(self):
        return [self.t, self.x, self.y, self.theta]
    
    def unpack_state_vector(self):
        self.t, self.x, self.y, self.theta = self.state
        
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
    
    def cost(self):
        # if it's not occupied, or hasn't yet been explored
        if not self.occupied or self.unexplored:
            return 1.0
        else:
            return 1000.0
        
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



class Map:
    def __init__(self,
                 cell_width = 1.0,
                 object_width = None
                 ):
        self.cells = {}
        self.cell_width = cell_width
        self.object_width = object_width

        dl = DataLoader(TEST)
        self.landmarks = dl["Landmark_Groundtruth"]

    def set_unexplored(self):
        for key, cell in self.cells.items():
            cell.unexplored = True

    def plot(self, f=None):
        if f is None:
            f = plt.figure()

        plt.xlim(self.minx, self.maxx + self.cell_width)
        plt.ylim(self.miny, self.maxy + self.cell_width)

        plt.grid(visible=True)
        plt.xticks(self.xrange)
        plt.yticks(self.yrange)

        ax = plt.gca()
        # https://stackoverflow.com/questions/17990845/how-do-i-equalize-the-scales-of-the-x-axis-and-y-axis
        ax.set_aspect('equal', adjustable='box')


        cell: Cell = None
        for key, cell in self.cells.items():
            if cell.occupied and not cell.unexplored:
                # ref: https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
                rect = patches.Rectangle(cell.xy, self.cell_width, self.cell_width, fill=True, color=random_color())
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

        # # if occupied
        self.is_occupied(cell)

        self.set_cell(xr, yr, cell)

    def is_occupied(self, cell: Cell):
        """
        xr - covers xr to xr+1. ex: xr = 3 -> cell covers 3-4

        """

        # ex:
        xys = self.landmarks[:, 1:3]
        for x, y in xys:
            if cell.contains(x, y, self.object_width):
                cell.set_occupied(True)

    def set_cell(self, xr, yr, cell):
        xr = real_to_grid(xr, self.cell_width)
        yr = real_to_grid(yr, self.cell_width)

        self.cells[(xr, yr)] = cell

    def get_cell(self, xr, yr) -> Cell:
        xr = real_to_grid(xr, self.cell_width)
        yr = real_to_grid(yr, self.cell_width)

        t = (xr, yr)

        if t in self.cells:
            return self.cells[(xr, yr)]
        else:
            return None

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

        self.make_all_neighbors()

    def make_all_neighbors(self):
        # for xr in self.xrange:
        #     for yr in self.yrange:
        c: Cell = None
        for key, c in self.cells.items():
            xr, yr = c.xy
            n = self.make_neighbors(c) # xr, yr)

            if c is None:
                print("something went wrong.")
                raise

            c.neighbors = n


class Node(Cell):
    pass

class Astar:
    def __init__(self):
        self.map = None
        self.s = None
        self.g = None

        # members
        self.plan = None

    def set_start(self, s: Node):
        self.s = s

    def set_goal(self, g: Node):
        self.g = g

    def set_map(self, map: Map):
        self.map = map

    def plan_(self):
        # setup
        s: Node = self.s
        g: Node = self.g
        node: Node

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
            h = np.linalg.norm(node.xy - g.xy)
            return h

        done = False
        print("A*: begin planning...")
        c = 0
        while not done:
            c += 1
            if c%100==0:
                print("Evaluated {} nodes.".format(c))
                print("node ", node.xy, " goal: ", g.xy)
            # get most promising node
            if not open_set.empty():
                prio, node = open_set.get()
            else:
                break
            closed_set[node] = True

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
                cost_hash[neighbor_node] = cost_hash[node] + neighbor_node.cost()

                # add parent
                parent[neighbor_node] = node

                # calculate cost
                f = cost_hash[neighbor_node] + heuristic(neighbor_node)

                # put into the open-set
                put((f, neighbor_node))

        if not node.equal(g):
            print("A* couldn't find a plan")
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

            plt.arrow(x, y, dx, dy, width=0.05, color=random_color())
    


    
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
    theta_landmark = np.atan2(s[1], s[0])

    # wrap to [-pi, pi]
    theta2 = wrap(theta)

    # heading is FROM the robot TO the landmark
    heading = theta_landmark - theta2

    return range, heading

class Robot:
    def __init__(self,
                 mm: MotionModel = MotionModel(state0=[0.0, 0.0, 0.0, -np.pi/2.0]),
                 max_accel = 0.288,
                 max_alpha = 5.579,
                 dt = 0.1,
                 v0 = 0.0,
                 w0 = 0.0,
                 kv = 1.0,
                 kw = 1.0,
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
        circle = patches.Circle(self.xy, self.radius, fill=True, color=random_color())
        ax.add_patch(circle)

    def get_state(self) -> tuple[float, float]:
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
        gx, gy = G

        # simple proportional controller
        x, y, theta = self.get_state()
        
        range, heading = range_bearing([x, y], theta, [gx, gy])

        v = self.kv * range
        w = self.kw * heading

        return v, w
    
    @property
    def xy(self):
        return self.get_state()[0:2]

    def step(self, G):
        v, w = self.controller(G)


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

    def drive(self, plan):
        """
        drive through a plan
        """
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
    
    def step(self, G):
        """
        G - goal
        grid robot achieves its goal automatically
        """
        self.x = G[0]
        self.y = G[1]

    def get_state(self):
        return np.array([self.x, self.y])
    
    @property
    def xy(self):
        return self.get_state()
    
    def set_xy(self, xy):
        self.x, self.y = xy



class OnlinePlanning:
    def __init__(self,
                 true_map: Map = None,
                 planner: Astar = None,
                 robot: Robot = None,
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

    def get_action(self, node: Node):
        xy = node.xy
        x, y = xy
        myxy = self.robot.xy
        myxy = np.array([real_to_grid(myxy[0], self.true_map.cell_width), real_to_grid(myxy[1], self.true_map.cell_width)])

        dxy = xy - myxy
        return dxy

    def run(self, S: Node, G: Node):
        # set up an online run
        self.robot.set_xy(S.center)
        map: Map = self.true_map # copy.deepcopy(self.true_map)
        map.set_unexplored()
        self.planner.set_goal(G)

        # outputs
        self.h = []

        def save_state():
            self.h.append((copy.deepcopy(self.true_map), copy.deepcopy(self.robot), copy.deepcopy(self.planner)))

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
            S = map.get_cell(self.robot.x, self.robot.y)

            # check if done
            if S.equal(G):
                done = True
                break

            # update the start
            self.planner.set_start(S)

            # plan
            plan = self.planner.plan_()

            # take the first non-start action -- idx = 1
            a = self.get_action(plan[1])

            # execute
            self.execute(a)

            # take an observation
            self.observe()

            # save the state
            save_state()

        return self.h


    def execute(self, a):
        """
        a = [dx, dy] - delta x and y
        """
        dx, dy = a

        assert(dx <= self.true_map.cell_width)
        assert(dy <= self.true_map.cell_width)

        self.robot.x += dx
        self.robot.y += dy


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
    xlim = np.array([-2, 5])
    ylim = np.array([-6, 6])
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

    def run(S, G):
        # executing
        astar.set_start(S)
        astar.set_goal(G)
        p = astar.plan_()

        # plot map
        f = map.plot()

        # plotting
        astar.plot(f)

        plt.show()

    sg = q3_states_and_goals()

    for i in range(len(sg)):
        S, G = sg[i]
        S, G = map.get_cell(*S), map.get_cell(*G)
        run(S, G)

def plot_history(history, f=None, axs=None, n=1):
    if f is None:
        f = plt.figure()

    m = max(1, int(len(history) / (n*n)) + 1)
    r = range(0, len(history), m)

    map: Map = None
    robot: Robot = None
    planner: Astar = None
    c=-1
    for i in r:
        c += 1
        map, robot, planner = history[i]

        # plt.subplot(n, n, c)
        plt.sca(axs.flatten()[c])

        map.plot(f)
        planner.plot(f)
        robot.plot(f)

        plt.title("Step {}".format(c))

    pass


def q5():
    map = Map()
    map.init(get_range())

    astar = Astar()
    robot = GridRobot()
    
    online_planner = OnlinePlanning(true_map=map, planner=astar, robot=robot)


    # runs
    def run(S, G):
        history = online_planner.run(S, G)
        
        # plots
        n = int(np.ceil(np.sqrt(len(history))))
        f, axs = plt.subplots(n, n)

        plot_history(history, f, axs, n*n)

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
    

    def run(S, G):
        # executing
        astar.set_start(S)
        astar.set_goal(G)
        p = astar.plan_()

        if plot:
            # plot map
            f = map.plot()

            # plotting
            astar.plot(f)

            plt.show()

        return p

    sg = q7_states_and_goals()

    plans = []
    for i in range(len(sg)):
        S, G = sg[i]
        S, G = map.get_cell(*S), map.get_cell(*G)
        plans.append(run(S, G))

    return plans

def q9():
    robot = Robot()

    pa, pb, pc = q7(plot=False)


    for p in [pa, pb, pc]:
        robot.drive(p)

def q10():
    # assume we still use q7's states and goals
    sg = q7_states_and_goals()
    S, G = sg[0]

    astar = Astar()
    robot = GridRobot()

    online_planner = OnlinePlanning(planner=astar, robot=robot)

    online_planner.run(S, G)

def q11():
    sg = q3_states_and_goals()

    def run(grid_width):
        pass

    grid_width = 0.1
    run(grid_width)

    grid_width = 1.0
    run(grid_width)


def main():
    q3()

    q5()

    q7()

    input("press any key")

if __name__ == "__main__":
    main()