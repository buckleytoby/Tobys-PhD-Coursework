import numpy as np
from collections import defaultdict
import env
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import globals


def save(fig, label, dir="outputs"):
    # https://stackoverflow.com/questions/7986567/matplotlib-how-to-set-the-current-figure
    plt.figure(fig.number)

    # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
    plt.savefig('./' + dir + '/' + label + '.png', dpi=300, bbox_inches='tight')

class Log:
    def __init__(self) -> None:
        self.d = []
        
    def __getitem__(self, idx):
        d = np.array(self.d[idx])
        
        return d
    
    def append(self, val):
        self.d.append(val)

    def __len__(self):
        return len(self.d)
class Logs():
    def __init__(self) -> None:
        self.d = defaultdict(Log)
        
    def __getitem__(self, key):
        d = np.array(self.d[key].d)
        
        return d
    
    def append(self, key, val):
        self.d[key].append(val)
        
    def items(self):
        return self.d.items()

class Logger:
    def __init__(self) -> None:
        self.logs = Logs()

    def log_one(self, key, val):
        self.logs.append(key, val)

    def plot(self, key):
        d = self.logs[key]

        f = plt.figure()
        plt.plot(d)
        plt.title(key)

        plt.show()
        
    def plot_avg(self, key, n, xlabel, label=""):
        # ref: https://learnpython.com/blog/average-in-matplotlib/
        n = int(n)
        
        d1 = self.logs[key]

        if len(d1) == 0:
            print("{} has no data".format(key))
            return
        
        # backward-fill the first value
        d1a = d1[0] * np.ones([n] + list(d1.shape[1:]))
        
        # forward-fill the last value
        d1b = d1[-1] * np.ones([n] + list(d1.shape[1:]))
        
        d3 = np.concatenate([d1a, d1, d1b], axis=0)
        
        d4 = []
        for idx in range(n, len(d3)-n):
            d5 = d3[idx-n:idx+n]
            m = np.mean(d5)
            
            d4.append(m)

        d6 = np.array(d4)

        # remove first and last n
        # d6 = d5[n:-n]

        f = plt.figure()
        plt.plot(d6)
        plt.title(key)
        plt.xlabel(xlabel)
        plt.ylabel(key)


        label2 = label + "_" + key
        save(f, label2)


        # plt.show()
        
    def print_last(self):
        for key, val in self.logs.items():
            print("{}: {}".format(key, val[-1]))
            
    def print_avg_last_n(self, n):
        n = int(n)

        for key, val in self.logs.items():
            val: Log
            l = len(val)

            if n > l:
                n2 = l
            else:
                n2 = n

            v1 = val[-n2:]
            
            v2 = np.mean(v1)
            
            print("{} avg last {}: {}".format(key, n, v2))

# global
LOG = True
logger = Logger()

class Layer:
    def __init__(self,
                 param_normalize_scale = 1e-6 # prevents param values from getting too large
                 ):
        self.param_normalize_scale = param_normalize_scale

        # my members
        self.params = {} # for storing param values
        self.grads = {} # for storing gradient values

    def init_param(self, shape):
        # this is [0, 1]
        v = np.random.random(shape)

        # change to [0, 2], then [-1, 1]
        v = v * 2.0 - 1.0

        return v
    
    def forward(self, inp):
        self.x = inp
        return inp
    
    def backward(self, upstream_grad):
        return np.array(None)

    def save(self, x):
        # save for the backward pass
        self.x = x.copy() # copy to ensure it doesn't get modified

    def scale_grad(self, scale):
        for key, val in self.grads.items():
            self.grads[key] = val * scale
            pass

    def step(self):
        for key, grad in self.grads.items():
            assert(key in self.params)

            # old value
            p = self.params[key]

            # mean the grads along the batch dim. Mean instead of sum so it doesn't scale with batch size
            g2 = np.mean(grad, axis=0, keepdims=True)

            # new val
            assert(p.shape == g2.shape)
            assert(not np.any(np.isnan(g2)))

            new_p = p + g2

            self.params[key] = new_p

        # TEST
        if True:
            self.param_normalization()

    def param_normalization(self):
        """
        compute a loss based on param size and update the param values
        """
        for key, p in self.params.items():
            # if L = 0.5p^2
            # then dL/dp = p
            grad = -p.copy()

            # scale
            grad *= self.param_normalize_scale

            # we want to min the loss, so subtract the grad
            self.params[key] += grad
            


class Dense(Layer):
    """
    Ax + B
    A in outputxinput
    B in outputx1
    Creates `output` output neurons
    """
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output

        # leading dim for the batch dim
        self.params['A'] = self.init_param([1, output, input])
        self.params['B'] = self.init_param([1, output, 1])

    def forward(self, inp):
        """
        y = Ax + B
        """
        x = inp
        self.save(x)

        x = self.params['A'] @ x

        # ignoring batch dim
        assert(x.shape[1:] == self.params['B'].shape[1:])
        x += self.params['B']

        assert(not np.any(np.isnan(x)))
        return x
    
    def backward(self, upstream_grad):
        """
        if y = Ax + B
        then dy/dA = x
        dy/dB = 1

        and if L = f(y) then dL/dA = dL/df * df/dA 

        dy/dA = x
        dy/dB = 1

        dy/dx = A
        return dy/dx * upstream_grad (chain rule)
        """
        # we expect upstream_grad to be [n, 1] and self.x.T to be [1, m] and self.grads['A'] to be [n, m]
        x2 = np.transpose(self.x, [0, 2, 1])
        self.grads['A'] = upstream_grad @ x2
        # self.grads['A'] = self.x @ upstream_grad
        
        self.grads['B'] = upstream_grad


        assert(self.grads['A'].shape[1:] == self.params['A'].shape[1:])
        assert(self.grads['B'].shape[1:] == self.params['B'].shape[1:])
        assert(not np.any(np.isnan(self.grads['A'])))
        assert(not np.any(np.isnan(self.grads['B'])))

        upstream_grad2 = np.transpose(upstream_grad, [0, 2, 1])

        out = upstream_grad2 @ self.params['A']

        out2 = np.transpose(out, [0, 2, 1])
        return out2

class LeakyReLu(Layer):
    """
    no params, but outputs x when x > 0.0, and scale * x when x < 0.0
    """
    def __init__(self, pos_scale, neg_scale):
        super().__init__()

        self.pos_scale = pos_scale
        self.neg_scale = neg_scale

    def forward(self, inp: np.ndarray):
        x = inp
        self.save(x)

        pos = x >= 0.0
        neg = x < 0.0

        x[pos] = self.pos_scale * x[pos]
        x[neg] = self.neg_scale * x[neg]


        assert(not np.any(np.isnan(x)))
        return x
    
    def backward(self, upstream_grad):
        """
        dy/dx is 1.0 when x > 0 and self.scale when x < 0 

        return dy/dx * upstream_grad (chain rule)
        """
        grad = upstream_grad.copy()

        pos = self.x > 0.0
        neg = self.x < 0.0

        # scale by scale
        grad[pos] *= self.pos_scale
        grad[neg] *= self.neg_scale

        # save no grads because I have no parameters

        return grad

class NN:
    """
    neural net class
    """
    def __init__(self, input, hidden, nb_hidden, output, needs_step=True) -> None:
        self.needs_step = needs_step
        self.nb_hidden = nb_hidden

        # RELU scales < 1.0 to reduce large default output values?
        pos_scale = 1.0
        neg_scale = 0.1

        # simple MLP
        self.dense1 = Dense(input, hidden)
        self.nonlinear1 = LeakyReLu(pos_scale, neg_scale)

        self.densen = Dense(hidden, output)

        def f():
            return [Dense(hidden, hidden), LeakyReLu(pos_scale, neg_scale)]

        self.layers: list[Layer] = [
            self.dense1,
            self.nonlinear1,
        ] 

        for _ in range(self.nb_hidden):
            self.layers += f()

        # final dense
        self.layers += [self.densen]
        
    def forward(self, inp, use_greedy=False):
        # ensure inp is 3d: [batch-dim, state-dim, 1]
        # x = np.reshape(inp, [inp.shape[0], 1])
        x = inp
        
        assert(x.ndim == 3)

        for layer in self.layers:
            x = layer.forward(x)

        # ensure output is flat
        # x = np.squeeze(x)

        assert(not np.any(np.isnan(x)))
        return x
    
    def backward(self, upstream_grad):
        """
        compute each gradient 
        """
        # make sure it's 2d
        # ug = np.reshape(upstream_grad, [upstream_grad.shape[0], 1])
        ug = upstream_grad

        # must traverse in reverse because order matters
        for layer in reversed(self.layers):
            ug = layer.backward(ug)
            pass

        return ug

    def scale_grad(self, scale):
        """
        scale each gradient 
        """
        for layer in self.layers:
            layer.scale_grad(scale)

    def step(self):
        """
        step each param 
        """
        for layer in self.layers:
            layer.step()

    # def param_length(self):
    #     l = 0.0

    #     for layer in self.layers:
    #         l += layer.param_length()
    
    def infer(self, inp):
        """
        assumes inp is a single input
        """
        x = inp.squeeze()
        
        # add state dim
        x = np.expand_dims(x, x.ndim)
        
        # add batch dim
        x = np.reshape(x, [1] + list(x.shape))
        
        y = self.forward(x)
        
        return y

class QLearning:
    def __init__(self,
                 gamma: float,
                 state_action_dim: int,
                 lr,
                 actor: NN,
                 ) -> None:
        
        self.gamma = gamma
        self.state_action_dim = state_action_dim
        self.lr = lr
        self.actor = actor

        # assertions
        assert(self.gamma < 1.0)
        assert(self.gamma >= 0.0)

        # my members
        self.Q = NN(self.state_action_dim, 32, 4, 1)

    def forward(self, state, action):
        # action: [0, 1, 2]
        # normalize action to [0, 1]
        a2 = action / 2.0

        # concat the s, a along the variable dim (1)
        sa = np.concatenate([state, a2], axis=1)

        qval = self.Q.forward(sa)

        return qval

    def bellman_eq(self, state, action, reward, statep, done):
        """
        Bellman Eq is Q(s, a) = r + gamma * Q(s', a')
        """
        # get the next action from our policy, greedy action
        ap = self.actor.forward(state, use_greedy=True)

        # # only compute future_q if the episode didn't end at this sample
        # if not done:
        #     future_q = self.gamma * self.forward(statep, ap)
        # else:
        #     future_q = 0.0
        
        # batch compatible
        qpvals = self.forward(statep, ap)
        
        # remove the row dim (dim 2)
        qpvals2 = np.reshape(qpvals, [-1, 1])
        
        # compute future-q's
        future_q = self.gamma * qpvals2 * (1.0 - done)

        qsa = reward + future_q

        return qsa

    def compute_loss(self, state, action, reward, statep, done):
        """
        Compute the QLearning loss 
        """
        # get the target q-val
        target_q = self.bellman_eq(state, action, reward, statep, done)

        # get the current q-val, do this 2nd so our NN saves the correct forward values
        qval = self.forward(state, action)
        
        # remove the row dim (dim 2)
        qval2 = np.reshape(qval, [-1, 1])


        # compute the MSE loss, 1/2(y' - y)^2 / batch_size
        # convention: final - initial, target - current
        dy = target_q - qval2

        batch_size = dy.shape[0]

        # normally I would use np.mean, but we can just divide by the batch size
        loss = 0.5 * (dy.T @ dy) / batch_size

        # add the var dim back in
        dy2 = np.expand_dims(dy, dy.ndim)

        # assuming MSE loss, the gradient of params p is dL/dp = (y' - y') * dy/dp -> dL/dy = (y' - y) * -1
        # dL/dy = (y' - y) * -1

        # must divide by batch size because we use MSE
        self.dLdy = -dy2 / batch_size

        # test
        if np.any(np.abs(loss) > 100.0):
            pass

        # logging
        logger.log_one("qval", qval.mean())
        return loss
    
    def backward(self, upstream_grad = None):
        """
        assuming MSE loss, the gradient of params p is dL/dp = (y' - y') * dy/dp -> dL/dy = (y' - y) * -1

        this is for the CRITIC network only
        """
        assert(upstream_grad is not None)
        # g = np.reshape(upstream_grad, [1, 1])
        g = upstream_grad
        assert(g.ndim == 3)
        return self.Q.backward(g)

    def step(self, state, action, reward, statep, done):
        # get the loss from the algorithm
        loss = self.compute_loss(state, action, reward, statep, done)
        logger.log_one('qloss', loss)

        # compute the gradient
        # we want to min the loss, so negate the grad
        dLdy = -self.dLdy
        self.backward(dLdy)

        # scale the gradient by the learning rate
        self.Q.scale_grad(self.lr)

        # step the parameters in the grad's direction
        self.Q.step()


class EpsGreedy(NN):
    def __init__(self,
                 max_eps, # low = less likely to explore
                 nb_steps,
                 max_nb_eps,
                 action_range,
                 needs_step = False # needs to step the actor. False for EpsGreedy
                 ) -> None:
        self.max_eps = max_eps
        self.nb_steps = nb_steps
        self.max_nb_eps = max_nb_eps
        self.action_range = action_range
        self.needs_step = needs_step

        # ref
        self.critic = None

        # my members
        self.layers = []
        self.eps = self.max_eps

    def calc_eps(self):
        # max_eps when step = 0.0
        # 0.1 when step = nb_steps
        # keep it simple, linear interp
        # alpha = globals.STEP / self.nb_steps # [0, 1]

        # turn off exploration for last 15%
        max = 0.85 * self.max_nb_eps

        # compute alpha
        alpha = globals.EPISODE / max

        # clip to [0, 1]
        alpha = np.clip(alpha, 0.0, 1.0)

        # compute epsilon
        eps = self.max_eps * (1.0 - alpha) + 0.025 * (alpha)

        # clip to [0, 1]
        eps = np.clip(eps, 0.0, 1.0)

        self.eps = eps

        return eps

    def set_critic(self, critic: QLearning):
        self.critic = critic

    def greedy(self, inp):
        assert(self.critic is not None)
        qvals = []
        s = inp

        b = inp.shape[0]

        # iterate over actions
        for a in range(self.action_range):
            a2 = a * np.ones([b, 1, 1])
            qval = self.critic.forward(s, a2)

            qvals.append(qval)

        # stack along the action dim
        qvals2 = np.concatenate(qvals, axis=1)

        # argmax along the action dim
        a = np.argmax(qvals2, axis=1, keepdims=True)

        return a
    
    def forward(self, inp, use_greedy=False):
        r = np.random.random()

        b = inp.shape[0]

        if r < self.calc_eps() and not use_greedy:
            a = np.random.randint(0, self.action_range, [b, 1, 1])
        else:
            a = self.greedy(inp)

        return a
    
class Optimizer:
    """
    takes a neural net and a loss and optimizes the neural net parameter values.

    Attempts to minimize the loss by computing a gradient and then moving each parameter in the direction of their gradient
    """
    def __init__(self,
                 algorithm: QLearning,
                 nn: NN,
                 lr: float,
                 nb_epochs: int,
                 state_dim,
                 ) -> None:
        self.algorithm = algorithm
        self.nn = nn
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.state_dim = state_dim

        # aliases
        self.critic = self.algorithm
        self.policy = self.nn


    def step_actor(self, state):
        """
        use actor to get action, use critic to evaluate the action, then propagate it all backwards
        """
        # forward pass through the actor
        action = self.policy.forward(state)

        # must do the forward pass to save state values
        qval = self.critic.forward(state, action)

        # logging
        logger.log_one("actor_qval", qval.mean())

        # now we want to maximize the qval, so L = -qval, and therefore dL/dqval = -1.0
        dL_dqval = -1.0 * np.ones_like(qval)

        # we want to min the loss, so negate the grad
        g = -dL_dqval

        # backprop through the critic
        dC_dqval = self.critic.backward(dL_dqval)
        assert(dC_dqval is not None)

        # first `state_dim` entries pertain to the state, last `action_dim` elements are the action grads
        g = dC_dqval[:, self.state_dim:]

        # backprop through the actor
        self.policy.backward(g)

        # scale the gradient by the learning rate
        self.policy.scale_grad(self.lr)

        # step the parameters in the grad's direction
        self.policy.step()

    def step(self, state, action, reward, statep, done):
        # step the critic
        self.algorithm.step(state, action, reward, statep, done)

        # step the actor
        # test
        if self.policy.needs_step:
            self.step_actor(state)
        
    def step_batch2(self, batch):
        # naive & slow, use a for loop

        for sample in batch:
            s, a, r, sp, done = sample

            self.step(s, a, r, sp, done)
        
    def step_batch(self, batch):
        # unzip it
        b1 = list(zip(*batch)) # ref: https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
        
        # np it, float it
        b2 = [np.array(b, dtype=np.float32) for b in b1]
        
        # add trailing dim to make all entries column vectors
        b3 = [np.expand_dims(b, axis=b.ndim) for b in b2]
        
        s, a, r, sp, done = b3
        
        # step it
        self.step(s, a, r, sp, done)

class Env:
    def __init__(self,
                 max_nb_steps,
                 use_motion_primitives = False,
                 ) -> None:
        self.max_nb_steps = max_nb_steps
        self.use_motion_primitives = use_motion_primitives

        self.cell_width = 2.0
        self.dt = 1.0 # unit dt

        self.map = env.Map(cell_width=self.cell_width)
        self.map.init(env.get_range(cell_width=self.cell_width))
        self.xrange = self.map.xrange
        self.yrange = self.map.yrange

        self.robot = env.Robot(dt = self.dt)

        # my members
        self.G: env.Cell
        self.nb_steps = 0

    def static_init(self):
        x = 0.0
        y = 0.0
        th = 0.0
        
        self.robot.reset_xyth(x, y, th)

        self.G = self.map.get_cell(5.0, 5.0) #type:ignore

        return self.get_state()

    def reset(self):
        self.nb_steps = 0

        if globals.TEST:
            return self.static_init()
        
        # print("finding new start and goal...")
        while True:
            # random xy
            # x = 2.5 # np.random.random() * self.xrange <>
            # y = 2.5 # np.random.random() * self.yrange <>
            th = 0.0 # np.random.random() * 2.0 * np.pi # [0, 360deg]

            # c = self.map.get_cell(x, y)
            c: env.Cell = self.map.get_random_cell()
            x, y = c.center

            # ensure it's not occupied
            assert(c is not None)
            # if c.occupied:
            #     continue

            self.robot.reset_xyth(x, y, th)

            # random non-occupied goal
            # g = self.map.get_random_cell()
            g = self.map.get_cell(3.5, 2.0) # constant goal, center of map, not occupied
            
            assert(g is not None)
            # if g.occupied:
            #     continue
            if g == c:
                # print("goal same as init state.")
                continue

            self.G = g

            # if we got here, we're good
            break

        # get initial state
        sp = self.get_state()
        return sp

    def get_state(self):
        # x, y, th
        x, y, th = self.robot.get_state()

        # wrap the angle so reduce input space
        th = env.wrap(th)

        # xg, yg = self.G.center

        s = np.array([x, y, th])
        
        # # add state dim
        # s2 = np.expand_dims(s, s.ndim)
        
        # # add batch dim
        # s3 = np.reshape(s2, [1] + list(s2.shape))
        return s
    
    def get_reward(self):
        """
        sparse reward
        """
        reward = 0.0
        eps = 0.25
        done = False

        # get our cell
        c = self.map.get_cell(*self.robot.xy)

        # large penalty for driving outside the map and end the episode
        if c is None:
            reward += -1.0 * 100.0
            # print("Went outside map.")
            done = True
        
        # inside the map
        else:
            # penalty for traversal (includes obstacle cost)
            reward += -1.0 * self.G.cost(c)

            # reward for reaching the goal
            d = self.G.dist(c)
            if d < eps:
                reward += 1000.0

                # reached goal means we're done
                # print("Reached goal.")
                done = True

        # check for time-out
        if self.nb_steps > self.max_nb_steps:
            # print("Timeout.")
            reward += -1.0 * 100.0 # penalty for timing out
            done = True

        # normalize rewards to [-1, 1]
        reward /= 100.0
        
        # add batch dim
        # r2 = np.expand_dims(reward, axis=0)
        # d2 = np.expand_dims(done, axis=0)

        return reward, done
    

    def step_control(self, action):
        """
        action: [v, w]
        """
        v, w = action
        self.nb_steps += 1

        # execute the action
        self.robot.step_vw(v, w)

        statep = self.get_state()
        reward, done = self.get_reward()

        return statep, reward, done
    
    def step(self, action):
        if self.use_motion_primitives:
            # map actions [left, right, straight] to controls
            v = 0.0
            w = 0.0
            if action == 0:
                w = np.pi / 2.0 # 90 deg
            elif action == 1:
                w = -np.pi / 2.0 # 90 deg
            else:
                v = self.cell_width

            a = (v, w)

        else:
            a = action

        return self.step_control(a)

class ReplayBuffer:
    def __init__(self) -> None:
        self.rb = []

    def store(self, state, action, reward, statep, done):
        t = (state, action, reward, statep, done)
        
        assert(len(t) == 5)
        self.rb.append(t)

    def get_batch(self, batch_size):
        # random indices
        h = len(self.rb)

        size = [batch_size]

        idxs = np.random.randint(0, h, size)

        # array indexing
        samples = [self.rb[i] for i in idxs]
        # samples = self.rb[idxs]

        return samples
    
    def __len__(self):
        return len(self.rb)

    def get_column(self, idx):
        """
        idx in [0:5]
        """
        assert(idx < 5)
        assert(idx >= 0)
        c = np.array([d[idx] for d in self.rb])
        return c
    
    def get_columns(self):
        cs = [self.get_column(idx) for idx in range(5)]

        return cs
    
    def save(self):
        # convert to np
        df = pd.DataFrame()
        # keys = ['state', 'action', 'reward', 'statep', 'done']
        keys = ['x', 'y', 'theta', 'a', 'r', 'xp', 'yp', 'thetap', 'done']

        cs = []
        for i in range(5):
            c = self.get_column(i)

            if c.ndim == 1:
                c = np.expand_dims(c, axis=-1)
            cs.append(c)

        cs2 = np.concatenate(cs, axis=1)

        df = pd.DataFrame(cs2, columns=keys)

        df.to_csv("./pd_dataset.csv", index = False)



class Training:
    def __init__(self,
                 env: Env,
                 opt: Optimizer,
                 policy: EpsGreedy,
                 action_shape,
                 state_shape,
                 batch_size,
                 ) -> None:
        self.env = env
        self.opt = opt
        self.policy = policy
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.batch_size = batch_size

        # my members
        self.rb = ReplayBuffer()
        self.val_rb = ReplayBuffer()

        self.load_into_rb()

    def outputs_to_actions(self, o):
        # 
        aidx = np.argmax(np.abs(o), axis=1, keepdims=True)

        aa = np.zeros_like(o)

        # for idx, aidx2 in enumerate(aidx):
        #     aa[idx, aidx2] = o[idx, aidx2]
        aa[aidx] = o[aidx]

        return aa, aidx

    def load_into_rb(self):
        """
        transform data into a form that is compatible with my alg
        """
        dl = env.DL

        i = dl['robot_inputs']
        o = dl['robot_outputs']

        # skip the time column
        ss = i[:, 1:]


        aa, aidx = self.outputs_to_actions(o)

        # reward is the negative loss: -1.0 * diff norm, along the var dim
        rr = -1.0 * np.linalg.norm(o - aa, axis=1)

        # normalize to 1.0ish
        rr /= np.abs(rr.mean())

        # train vs val
        l = len(ss) - 1

        # percent
        percent = 0.90

        nb_train = int(percent * l)

        # train
        for i in range(0, nb_train):
            s = ss[i]
            a = aidx[i]
            r = rr[i]
            sp = ss[i] # not actually used

            self.rb.store(s, a, r, sp, False)

        # val
        for i in range(nb_train, l):
            s = ss[i]
            a = aidx[i]
            r = rr[i]
            sp = ss[i] # not actually used

            self.val_rb.store(s, a, r, sp, False)

        pass

        
    def avg_and_print(self, dd):
        l = dd['len']
        
        # print("Ep len: {}".format(l))
        for key, val in dd.items():
            v = val / l
            
            # print("{} avg: {}".format(key, v))
            logger.log_one("avg_{}".format(key), v)

    def sum_and_print(self, dd):
        for key, val in dd.items():
            v = np.sum(val)
            
            # print("{} total: {}".format(key, v))
            logger.log_one("episodic_{}".format(key), v)

    def plot_rollout(self):
        f = self.env.map.plot()
        g = self.env.G
        g.plot(f)

        self.env.robot.plot_history_xy(f)

        label = "step_{}".format(globals.STEP)
        save(f, label, dir="rollouts")
        # plt.show()

    def val_loss(self):
        l = len(self.val_rb)

        # get val set
        s, a, r, sp, done = self.val_rb.get_columns()

        s2 = np.expand_dims(s, -1)

        # eps greedy
        aa = self.policy.forward(s2, use_greedy=True)
        aa2 = np.squeeze(aa)
        a2 = np.squeeze(a)
        # loss
        loss = np.mean(np.abs(a2 - aa2))

        logger.log_one("val_loss", loss)
        

    def run(self, nb_episodes):
        # initial values
        action = np.zeros(self.action_shape) # old action
        state = np.zeros(self.state_shape) # old state

        # reset the env, get intial state
        # state = self.env.reset()
        done = False
        ep_averager = defaultdict(float)
        c = 0
        warmup = 0

        if nb_episodes == 0:
            return
        
        def reset():
            self.val_loss()
            pass
            # nonlocal ep_averager, c, nb_episodes, state, done

            # # plotting the rollout when we're in the endgame
            # if (c / nb_episodes) > 0.999:
            #     self.plot_rollout()

            # globals.EPISODE += 1
            
            # ep_len = ep_averager['len']
            # rewards = ep_averager['reward']

            # ep_reward = np.sum(rewards)

            # if ep_reward > 0.0:
            #     success = 1.0

            #     # ep length if it was successful
            #     logger.log_one("time_to_goal", ep_len)
            # else:
            #     success = 0.0

            # # log success rate
            # logger.log_one("success_rate", success)
            # logger.log_one("epsilon", self.policy.eps)
            
            # self.avg_and_print(ep_averager)
            # self.sum_and_print(ep_averager)
            
            # # state = self.env.reset()
            # done = False
            # ep_averager = defaultdict(float)

        def print_():
            print("------------------------------------------------")
            print("Ep {} of {}".format(c, nb_episodes))
            logger.print_avg_last_n(200)
            print("------------------------------------------------")

        while c < nb_episodes:
            # whether to reset
            if done:
                c += 1
                reset()
                
                done = False

            if False:
                # query policy for an action
                action = self.policy.infer(state)
                
                # remove batch and state dim
                a2 = np.reshape(action, [1]) # action.squeeze()

                # step env
                statep, reward, done = self.env.step(a2)
                
                #
                ep_averager['reward'] += reward
                ep_averager['len'] += 1.0

                logger.log_one("reward", reward)

                # save the sample
                self.rb.store(state, a2, reward, statep, done)

            ## training
            if globals.STEP > warmup:
                if False:
                    # TEST to ensure learning
                    cc = 0
                    while True:
                        cc += 1
                        # get a batch
                        batch = self.rb.get_batch(self.batch_size)

                        # step the optimizer
                        self.opt.step_batch(batch)

                        if cc % 50 == 0:
                            print_()
                else:
                    # get a batch
                    batch = self.rb.get_batch(self.batch_size)

                    # step the optimizer
                    self.opt.step_batch(batch)


            # save the old state
            # state = statep

            # increment the step counter
            globals.STEP += 1

            # logging
            if globals.STEP % 100 == 0:
                print_()
                done = True

        # one final reset for plotting
        reset()

    def save_rb(self):
        """
        save the replay buffer to disk
        """
        self.rb.save()


def calc_map_Vvals(map: env.Map, critic):
    cell: env.Cell
    for key, cell in map.cells:
        v = 0.0
        x, y = cell.xy

        # for each action
        for i in [0, 1, 2]:
            pass




def main():
    globals.TEST = False

    # network params
    hidden_dim = 16
    nb_hidden_layers = 2

    # learning params
    lra = 1e-7 # unused
    lrc = 1e-2
    gamma = 0.0 # discount
    nb_epochs = 0 # not used
    batch_size = 256
    max_nb_episode_steps = 100 # recall, each step is 0.1 seconds
    nb_episodes = 5000
    max_epsilon = 0.0

    # env params
    state_dim = 4 # [v, w, cth, sth]
    action_range = 3 # left, right, forward
    use_motion_primitives = True

    if use_motion_primitives:
        action_dim = 1 # [a]
    else:
        action_dim = 2 # [v, w]


    state_shape = [state_dim]
    action_shape = [action_dim]

    # make the classes
    # actor = NN(state_dim, hidden_dim, action_dim) # actor
    actor = EpsGreedy(max_epsilon, nb_episodes * max_nb_episode_steps, nb_episodes, action_range)

    # make the qlearning alg, includes a critic NN
    algorithm = QLearning(gamma, state_dim + action_dim, lrc, actor) # critic

    # pass a ref
    actor.set_critic(algorithm)

    # make the optimizer
    opt = Optimizer(algorithm, actor, lra, nb_epochs, state_dim)

    # make the sim environment
    env = None # Env(max_nb_episode_steps, use_motion_primitives)

    # make the trainer and run
    trainer = Training(env, opt, actor, action_shape, state_shape, batch_size)

    # try:
    trainer.run(nb_episodes)
    # except Exception as e:
    #     print(e)
    #     pass

    # plotting
    if False:
        f = env.map.plot()
        env.G.plot(f)
        plt.legend()
        save(f, "map")
        # plt.show()


    if True:
        label = "lr{:.0E}_nb_ep{}_test".format(lrc, nb_episodes)

        n = 200
        logger.plot_avg('qloss', n, "Step Count", label=label)
        # logger.plot_avg('reward', n, "Step Count", label=label)
        logger.plot_avg("qval", n, "Step Count", label=label)
        # logger.plot_avg("epsilon", 1, "Step Count", label=label)
        # logger.plot_avg("episodic_reward", 50, "Episode Count", label=label)
        # logger.plot_avg("success_rate", 50, "Episode Count", label=label)
        logger.plot_avg("val_loss", 10, "Episode Count", label=label)

    # input("press any key to exit")

    # save the replay buffer?
    # trainer.save_rb()

if __name__ == "__main__":
    main()