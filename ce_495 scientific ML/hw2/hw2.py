import torch
from torch import nn

import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt

class Exercise2:
    def __init__(self) -> None:
        """
        T P = 30, F P = 10, T N = 150, and F N = 10
        """
        TP = 30
        FP = 10
        TN = 150
        FN = 10

        # $(TP+TN)/(TP+TN+FP+FN)$
        acc = (TP+TN)/(TP+TN+FP+FN)
        prec = TP/(TP+FP)
        recall = TP/(TP+FN)
        spec = TN / (TN+FP)
        f1 = TP/(TP+0.5*(FP+FN))

        print("acc: {}".format(acc))
        print("prec: {}".format(prec))
        print("recall: {}".format(recall))
        print("spec: {}".format(spec))
        print("f1: {}".format(f1))


e = Exercise2()

"""
Exercise 4: Physics-Based Regression and Model Fitting
Consider the problem of throwing a ball from the top of a building of height h, with initial velocity (v0x, v0y), under constant gravity. 

Plug in some reasonable real numbers for these variables. Write down the ground-truth physical model describing the position (x(t), y(t)) as a function of time.

Generate synthetic data by sampling the trajectory at discrete times and corrupt the positions with
additive Gaussian noise. 

Fit the noisy data using linear (for both x(t) and y(t)), quadratic (for
both), and cubic (for both) polynomial models.

Implement the models and optimization procedure in PyTorch (preferably using the L-BFGS-B
algorithm or another optimizer of your choice). Compare the fitted models and comment the
results.
"""
class Quadratic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # inputs: t, t^2
        # outputs: x(t), y(t)
        self.dense = nn.Linear(2, 2)

    def forward(self, t):
        # square x
        t2 = t**2

        # assemble features
        t3 = torch.concatenate((t, t2), dim=1)

        xy = self.dense(t3)

        return xy

class Cubic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # inputs: t, t^2, t^3
        # outputs: x(t), y(t)
        self.dense = nn.Linear(3, 2)

    def forward(self, t):
        # square t
        t2 = t**2

        # cube t
        t3 = t**3

        # assemble features
        t4 = torch.concatenate((t, t2, t3), dim=1)

        xy = self.dense(t4)

        return xy

class Exercise4:
    def __init__(self) -> None:
        
        self.make_models()

        self.generate_data()

        self.train()

        self.plot()

    def eom(self, t):
        # reference: https://www.physicsclassroom.com/class/1dkin/Lesson-6/Kinematic-Equations
        # equations of motion
        # assumes no air resistance
        g = -9.81
        h = 10 # m
        v0x = 2 # m/s
        v0y = 20 # m/s

        # x is just linear
        x = v0x * t

        # y is quadratic
        y = v0y * t + 0.5 * g * t**2 + h

        return x, y

    def generate_data(self):
        # generate data
        n = 100

        data = defaultdict(list)

        std = 1.5

        for i in range(n):
            t = 5.1 * np.random.random()

            x, y = self.eom(t)

            # add noise
            x += np.random.normal(0, std)
            y += np.random.normal(0, std)

            y2 = np.array([x, y])

            # add to dataset
            data['t'].append(np.array([t]))
            data['y'].append(y2)
            # data['x'].append(x)
            # data['y'].append(y)

        # convert to torch
        self.data = {}
        for key, val in data.items():
            self.data[key] = torch.tensor(data[key], dtype=torch.float32)


    def make_models(self):
        # make the models
        self.linear = nn.Linear(1, 2)
        self.quadratic = Quadratic()
        self.cubic = Cubic()

    def forward(self):
        t = self.data['t']

        y1 = self.linear(t)
        y2 = self.quadratic(t)
        y3 = self.cubic(t)

        return y1, y2, y3
    
    def loss(self, y):
        losser = nn.MSELoss()

        loss = losser(y, self.data['y'])

        return loss

    def train(self):
        n = 10000
        lr = 5.0e-4

        opt1 = torch.optim.SGD(self.linear.parameters(), lr=lr)
        opt2 = torch.optim.SGD(self.quadratic.parameters(), lr=lr)
        opt3 = torch.optim.SGD(self.cubic.parameters(), lr=lr)

        for i in range(n):
            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            
            y1, y2, y3 = self.forward()

            l1 = self.loss(y1)
            l2 = self.loss(y2)
            l3 = self.loss(y3)

            l1.backward()
            l2.backward()
            l3.backward()

            opt1.step()
            opt2.step()
            opt3.step()

    @torch.no_grad()
    def plot(self):
        t = np.expand_dims(np.linspace(0, 5.1, 100), axis=1)
        t = torch.tensor(t, dtype=torch.float32)

        y1 = self.linear(t)
        y2 = self.quadratic(t)
        y3 = self.cubic(t)

        tgt = self.data['t']
        ygt = self.data['y']

        t = t.numpy()
        tgt = tgt.numpy()
        y1 = y1.numpy()
        y2 = y2.numpy()
        y3 = y3.numpy()
        ygt = ygt.numpy()

        labels = ['linear', 'quadratic', 'cubic']
        series = [y1, y2, y3]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot t vs x1
        for y, label in zip(series, labels):
            ax1.scatter(t, y[:, 0], label=label)

        # gt
        ax1.scatter(tgt, ygt[:, 0], label='gt')
        
        ax1.set_ylabel('$x_1$')
        ax1.set_title('Time $t$ vs $x_1$')
        ax1.legend()
        ax1.grid(True)

        # Plot t vs x2
        for y, label in zip(series, labels):
            ax2.scatter(t, y[:, 1], label=label)

        # gt
        ax2.scatter(tgt, ygt[:, 1], label='gt')


        ax2.set_ylabel('$x_2$')
        ax2.set_title('Time $t$ vs $x_2$')
        ax2.set_xlabel('Time ($t$)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        
if False:
    e = Exercise4()

"""
Exercise 6: Nonlinear Activations and Learning XOR
"""
class PerceptronActivation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        1 if <= 0
        0 else
        """
        y = torch.zeros_like(x)
        b1 = x <= 0

        y[b1] = 1.0

        return y
    

def make_p_layer(activation = None):
    if activation == None:
        p1 = nn.Sequential(
            nn.Linear(2, 100),
            nn.Linear(100, 1),
        )

    else:
        p1 = nn.Sequential(
            nn.Linear(2, 1000),
            nn.Linear(1000, 1),
            activation()
        )
    return p1
    
class E5Model(nn.Module):
    def __init__(self, activation, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        
        self.p1 = make_p_layer(activation)
        self.p2 = make_p_layer(activation)
        self.p3 = make_p_layer(activation)

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x)

        x3 = torch.concatenate((x1, x2), dim=1)

        y = self.p3(x3)

        return y


class Exercise6:
    def __init__(self) -> None:
        
        x = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ])
        y = np.array([
            [0],
            [1],
            [1],
            [0]
        ])

        self.data = {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32)
        }

        self.b()

        self.train()

        self.test()

    def b(self):
        """
        b) Train the network to solve the XOR problem using:
        • a purely linear model,
        • a model with ReLU activation functions,
        • a model with sigmoid activation functions with different slopes.
        """
        self.linear = E5Model(activation=None)
        self.relu = E5Model(activation=nn.ReLU)
        self.sigmoid = E5Model(activation=nn.Sigmoid)

    def forward(self):
        t = self.data['x']

        y1 = self.linear(t)
        y2 = self.relu(t)
        y3 = self.sigmoid(t)

        return y1, y2, y3
    
    def loss(self, y):
        # for simplicity I'm going to use MSE
        losser = nn.MSELoss()

        loss = losser(y, self.data['y'])

        return loss

    def train(self):
        n = 20000
        lr = 1e-3

        opt1 = torch.optim.SGD(self.linear.parameters(), lr=lr)
        opt2 = torch.optim.SGD(self.relu.parameters(), lr=lr)
        opt3 = torch.optim.SGD(self.sigmoid.parameters(), lr=lr)

        for i in range(n):
            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            
            y1, y2, y3 = self.forward()

            l1 = self.loss(y1)
            l2 = self.loss(y2)
            l3 = self.loss(y3)

            l1.backward()
            l2.backward()
            l3.backward()

            opt1.step()
            opt2.step()
            opt3.step()

    @torch.no_grad()
    def test(self):
        y1, y2, y3 = self.forward()

        print("linear output:\n", y1)
        print("relu output:\n", y2)
        print("sigmoid output:\n", y3)

if True:
    Exercise6()