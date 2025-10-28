import numpy as np

class Layer:
    def __init__(self):
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
        self.x = x

    def scale_grad(self, scale):
        for key, val in self.grads.items():
            self.grads[key] = val * scale

    def step(self):
        for key, grad in self.grads.items():
            assert(key in self.params)

            # old value
            p = self.params[key]

            # new value. I'm pretty sure I add on the raw grad, not a function of p
            self.params[key] += grad

class Dense(Layer):
    """
    Ax + B
    """
    def __init__(self, length):
        super().__init__()

        self.params['A'] = self.init_param(length)
        self.params['B'] = self.init_param(length)

    def forward(self, inp):
        x = inp

        x = self.params['A'] * x

        x += self.params['B']

        self.save(x)

        return x
    
    def backward(self, upstream_grad):
        """
        partial w.r.t. A is x
        partial w.r.t. B is 1
        """
        self.grads['A'] = upstream_grad * self.x
        
        self.grads['B'] = upstream_grad

        # the thing which is multiplied against self.x is the thing that propagates backwards, so return grad-A
        return self.grads['A']

class LeakyReLu(Layer):
    """
    no params, but outputs x when x > 0.0, and scale * x when x < 0.0
    """
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, inp: np.ndarray):
        x = inp.copy()

        pos = x > 0.0
        neg = x < 0.0

        x[neg] = self.scale * x[neg]

        self.save(x)
        return x
    
    def backward(self, upstream_grad):
        """
        gradient w.r.t x is 1.0 when x > 0 and self.scale when x < 0 
        """
        grad = np.zeros_like(self.x)

        pos = self.x > 0.0
        neg = self.x < 0.0

        grad[pos] = 1.0
        grad[neg] = self.scale

        # save no grads because I have no parameters

        return grad

class NN:
    """
    neural net class
    """
    def __init__(self, length) -> None:
        # simple MLP
        self.dense1 = Dense(length)
        self.nonlinear1 = LeakyReLu(scale = 0.1)
        self.dense2 = Dense(length)

        self.layers: list[Layer] = [
            self.dense1,
            self.nonlinear1,
            self.dense2
        ]
        
    def forward(self, inp):
        x = inp

        x = self.dense1.forward(x)
        x = self.nonlinear1.forward(x)
        x = self.dense2.forward(x)

        return x
    
    def backward(self, upstream_grad):
        """
        compute each gradient 
        """
        ug = upstream_grad
        # must traverse in reverse because order matters
        for layer in reversed(self.layers):
            ug = layer.backward(ug)

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



class QLearning:
    def __init__(self,
                 gamma: float,
                 policy: NN,
                 state_action_dim: int,
                 ) -> None:
        
        self.gamma = gamma
        self.policy = policy
        self.state_action_dim = state_action_dim

        # assertions
        assert(self.gamma < 1.0)
        assert(self.gamma >= 0.0)

        # my members
        self.Q = NN(self.state_action_dim)

    def forward(self, state, action):
        # concat the s, a
        sa = np.concatenate([state, action])

        qval = self.Q.forward(sa)

        return qval

    def bellman_eq(self, state, action, reward, statep):
        """
        Bellman Eq is Q(s, a) = r + gamma * Q(s', a')
        """
        # get the next action from our policy
        ap = self.policy.forward(statep)

        qsa = reward + self.gamma * self.forward(statep, ap)

        return qsa

    def compute_loss(self, state, action, reward, statep):
        """
        Compute the QLearning loss 
        """
        # get the target q-val
        target_q = self.bellman_eq(state, action, reward, statep)

        # get the current q-val, do this 2nd so our NN saves the correct forward values
        qval = self.forward(state, action)

        # compute the MSE loss, 1/2(y-y')^2
        dy = qval - target_q
        loss = 0.5 * np.inner(dy, dy)

        self.dLdy = dy

        return loss
    
    def backward(self, upstream_grad = None):
        """
        assuming MSE loss, the gradient of params p is dL/dp = (y - y') * dy/dp -> dL/dy = (y-y')
        """
        dLdy = self.dLdy

        self.policy.backward(dLdy)

    
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
                 ) -> None:
        self.algorithm = algorithm
        self.nn = nn
        self.lr = lr
        self.nb_epochs = nb_epochs


    def step(self, state, action, reward, statep):
        # get the loss from the algorithm
        loss = self.algorithm.compute_loss(state, action, reward, statep)

        # compute the gradient
        self.algorithm.backward()

        # scale the gradient by the learning rate
        self.nn.scale_grad(self.lr)

        # step the parameters in the grad's direction
        self.nn.step()
        
    def step_batch(self, batch):
        # naive, use a for loop

        for sample in batch:
            s, a, r, sp = sample

            self.step(s, a, r, sp)

class Env:
    def step(self, action):
        state = None
        action = None
        reward = None

        return state, action, reward

class ReplayBuffer:
    def __init__(self) -> None:
        self.rb = []

    def store(self, state, action, reward, statep):
        t = (state, action, reward, statep)
        self.rb.append(t)

    def get_batch(self, batch_size):
        # random indices
        h = len(self.rb)

        size = [batch_size]

        idxs = np.random.randint(0, h, size)

        # array indexing
        samples = self.rb[idxs]

        return samples

class Training:
    def __init__(self,
                 env: Env,
                 opt: Optimizer,
                 action_shape,
                 state_shape,
                 batch_size,
                 ) -> None:
        self.env = env
        self.opt = opt
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.batch_size = batch_size

        # my members
        self.rb = ReplayBuffer()

    def run(self):
        # initial values
        action = np.zeros(self.action_shape)
        statep = np.zeros(self.state_shape)

        while True:
            # step env
            state, action, reward = self.env.step(action)

            # save the sample
            self.rb.store(state, action, reward, statep)

            # get a batch
            batch = self.rb.get_batch(self.batch_size)

            # step the optimizer
            self.opt.step_batch(batch)

            # save the old state
            statep = state



def main():
    # learning params
    lr = 1e-5
    gamma = 0.95
    nb_epochs = 1000
    batch_size = 32

    # env params
    state_dim = 3 # [x, y, th]
    action_dim = 2 # [v, w]

    state_shape = [3]
    action_shape = [2]

    # make the classes
    policy = NN(state_dim)
    algorithm = QLearning(gamma, policy, state_dim + action_dim)
    opt = Optimizer(algorithm, policy, lr, nb_epochs)
    env = Env()

    # make the trainer and run
    trainer = Training(env, opt, action_shape, state_shape, batch_size)
    trainer.run()

    # plotting