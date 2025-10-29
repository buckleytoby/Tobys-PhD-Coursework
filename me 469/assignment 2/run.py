import numpy as np
from collections import defaultdict
import env
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Logger:
    def __init__(self) -> None:
        self.logs = defaultdict(list)

    def log_one(self, key, val):
        self.logs[key].append(val)

    def plot(self, key):
        d = self.logs[key]

        f = plt.figure()
        plt.plot(d)
        plt.title(key)

        plt.show()

# global
LOG = True
logger = Logger()

class Layer:
    def __init__(self,
                 param_normalize_scale = 1e-5 # prevents param values from getting too large
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
        pass

    def save(self, x):
        # save for the backward pass
        self.x = x.copy() # copy to ensure it doesn't get modified

    def scale_grad(self, scale):
        for key, val in self.grads.items():
            self.grads[key] = val * scale

    def step(self):
        for key, grad in self.grads.items():
            assert(key in self.params)

            # old value
            p = self.params[key]

            # new val
            assert(p.shape == grad.shape)
            assert(not np.any(np.isnan(grad)))

            new_p = p + grad

            self.params[key] = new_p

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

        self.params['A'] = self.init_param([output, input])
        self.params['B'] = self.init_param([output, 1])

    def forward(self, inp):
        """
        y = Ax + B
        """
        x = inp
        self.save(x)

        x = self.params['A'] @ x

        assert(x.shape == self.params['B'].shape)
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
        self.grads['A'] = upstream_grad @ self.x.T
        
        self.grads['B'] = upstream_grad


        assert(self.grads['A'].shape == self.params['A'].shape)
        assert(self.grads['B'].shape == self.params['B'].shape)
        assert(not np.any(np.isnan(self.grads['A'])))
        assert(not np.any(np.isnan(self.grads['B'])))

        out = upstream_grad.T @ self.params['A']
        return out.T

class LeakyReLu(Layer):
    """
    no params, but outputs x when x > 0.0, and scale * x when x < 0.0
    """
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, inp: np.ndarray):
        x = inp
        self.save(x)

        pos = x > 0.0
        neg = x < 0.0

        x[neg] = self.scale * x[neg]


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
        grad[neg] *= self.scale

        # save no grads because I have no parameters

        return grad

class NN:
    """
    neural net class
    """
    def __init__(self, input, hidden, output) -> None:
        # simple MLP
        self.dense1 = Dense(input, hidden)
        self.nonlinear1 = LeakyReLu(scale = 0.1)

        self.densen = Dense(hidden, output)

        def f():
            return [Dense(hidden, hidden), LeakyReLu(scale = 0.1)]

        self.layers: list[Layer] = [
            self.dense1,
            self.nonlinear1,
        ] 
        # self.layers += f()
        self.layers += [self.densen]
        
    def forward(self, inp):
        # ensure inp is 2d
        x = np.reshape(inp, [inp.shape[0], 1])

        for layer in self.layers:
            x = layer.forward(x)

        # ensure output is flat
        x = np.squeeze(x)

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



class QLearning:
    def __init__(self,
                 gamma: float,
                 policy: NN,
                 state_action_dim: int,
                 lr,
                 ) -> None:
        
        self.gamma = gamma
        self.policy = policy
        self.state_action_dim = state_action_dim
        self.lr = lr

        # assertions
        assert(self.gamma < 1.0)
        assert(self.gamma >= 0.0)

        # my members
        self.Q = NN(self.state_action_dim, 16, 1)

    def forward(self, state, action):
        # concat the s, a
        sa = np.concatenate([state, action])

        qval = self.Q.forward(sa)

        return qval

    def bellman_eq(self, state, action, reward, statep, done):
        """
        Bellman Eq is Q(s, a) = r + gamma * Q(s', a')
        """
        # get the next action from our policy
        ap = np.zeros([2]) # self.policy.forward(statep)

        # only compute future_q if the episode didn't end at this sample
        if not done:
            future_q = self.gamma * self.forward(statep, ap)
        else:
            future_q = 0.0

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

        # compute the MSE loss, 1/2(y' - y)^2
        # convention: final - initial, target - current
        dy = target_q - qval
        loss = 0.5 * np.inner(dy, dy)

        # assuming MSE loss, the gradient of params p is dL/dp = (y' - y') * dy/dp -> dL/dy = (y' - y) * -1
        # dL/dy = (y' - y) * -1
        self.dLdy = -dy

        # test
        if abs(loss) > 100.0:
            pass

        return loss
    
    def backward(self, upstream_grad = None):
        """
        assuming MSE loss, the gradient of params p is dL/dp = (y' - y') * dy/dp -> dL/dy = (y' - y) * -1

        this is for the CRITIC network only
        """
        assert(upstream_grad is not None)
        g = np.reshape(upstream_grad, [1, 1])
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
        action = self.policy.forward(state)

        qval = self.critic.forward(state, action)

        # now we want to maximize the qval, so L = -qval, and therefore dL/dqval = -1.0
        dL_dqval = -1.0

        # we want to min the loss, so negate the grad
        g = -dL_dqval

        # backprop through the critic
        dC_dqval = self.critic.backward(dL_dqval)
        assert(dC_dqval is not None)

        # first `state_dim` entries pertain to the state, last `action_dim` elements are the action grads
        g = dC_dqval[self.state_dim:]

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
        if True:
            self.step_actor(state)
        
    def step_batch(self, batch):
        # naive, use a for loop

        for sample in batch:
            s, a, r, sp, done = sample

            self.step(s, a, r, sp, done)

class Env:
    def __init__(self,
                 max_nb_steps,
                 ) -> None:
        self.max_nb_steps = max_nb_steps

        self.map = env.Map()
        self.map.init(env.get_range())
        self.xrange = self.map.xrange
        self.yrange = self.map.yrange

        self.robot = env.Robot()

        # my members
        self.G: env.Cell
        self.nb_steps = 0


    def reset(self):
        self.nb_steps = 0
        
        while True:
            # random xy
            x = 1.0 # np.random.random() * self.xrange <>
            y = 1.0 # np.random.random() * self.yrange <>
            th = np.random.random() * 2.0 * np.pi # [0, 360deg]

            c = self.map.get_cell(x, y)

            # ensure it's not occupied
            assert(c is not None)
            if c.occupied:
                continue

            self.robot.reset_xyth(x, y, th)

            # random non-occupied goal
            g = self.map.get_random_cell()
            
            assert(g is not None)
            if g.occupied:
                continue

            self.G = g

            # if we got here, we're good
            break

        # get initial state
        sp = self.get_state()
        return sp

    def get_state(self):
        # x, y, th, xg, yg
        x, y, th = self.robot.get_state()

        xg, yg = self.G.center

        s = np.array([x, y, th, xg, yg])
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

        # large penalty for driving outside the map
        if c is None:
            reward += -1.0 * 1000.0
        
        # inside the map
        else:
            # penalty for traversal
            reward += -1.0 * self.G.cost(c)

            # reward for reaching the goal
            d = self.G.dist(c)
            if d < eps:
                reward += 10.0

                # reached goal means we're done
                print("Reached goal")
                done = True

        # check for time-out
        if self.nb_steps > self.max_nb_steps:
            print("Timeout")
            done = True

        return reward, done

    def step(self, action):
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

class ReplayBuffer:
    def __init__(self) -> None:
        self.rb = []

    def store(self, state, action, reward, statep, done):
        t = (state, action, reward, statep, done)
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

class Training:
    def __init__(self,
                 env: Env,
                 opt: Optimizer,
                 policy: NN,
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

    def run(self, nb_episodes):
        # initial values
        action = np.zeros(self.action_shape) # old action
        state = np.zeros(self.state_shape) # old state

        # reset the env, get intial state
        state = self.env.reset()
        done = False

        c = 0
        while c < nb_episodes:
            # whether to reset
            if done:
                state = self.env.reset()
                c += 1
                print("Ep reset. #{} of {}".format(c, nb_episodes))

            # query policy for an action
            # action = self.policy.forward(state)

            # step env
            statep, reward, done = self.env.step(action)

            logger.log_one("reward", reward)

            # save the sample
            self.rb.store(state, action, reward, statep, done)

            # get a batch
            batch = self.rb.get_batch(self.batch_size)

            # step the optimizer
            if False: # test on overfitting
                while True:
                    self.opt.step_batch(batch)
            else:
                self.opt.step_batch(batch)

            # save the old state
            state = statep



def main():
    # network params
    hidden_dim = 16
    nb_hidden_layers = 2

    # learning params
    lra = 1e-5
    lrc = 1e-4
    gamma = 0.95 # discount
    nb_epochs = 1000
    batch_size = 32
    max_nb_episode_steps = 100 # recall, each step is 0.1 seconds
    nb_episodes = 200

    # env params
    state_dim = 5 # [x, y, th, xg, yg]
    action_dim = 2 # [v, w]

    state_shape = [state_dim]
    action_shape = [action_dim]

    # make the classes
    policy = NN(state_dim, hidden_dim, action_dim)
    algorithm = QLearning(gamma, policy, state_dim + action_dim, lrc)
    opt = Optimizer(algorithm, policy, lra, nb_epochs, state_dim)
    env = Env(max_nb_episode_steps)

    # make the trainer and run
    trainer = Training(env, opt, policy, action_shape, state_shape, batch_size)
    # try:
    trainer.run(nb_episodes)
    # except Exception as e:
    #     print(e)
    #     pass

    # plotting
    logger.plot('qloss')
    logger.plot('reward')

    input("press any key to exit")

if __name__ == "__main__":
    main()