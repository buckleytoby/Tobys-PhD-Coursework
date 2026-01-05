import torch
from torch import nn
import tqdm
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal

PLOT = True

if PLOT:
    plt.ion()
    # fig = plt.figure()


BATCH_SIZE = 64
DEVICE = 'cuda' # must be cpu for backward hook to work

"""
Sandcastle 6 --- assumption: general knowledge of math, physics, robotics are beneficial for dexterous manipulation, will exhibit more generalizability / adaptivity, reduce overfitting

fundamentally, if nb params > nb datapoints then we're overconstrained => there is an exact solution => with enough time the NN will fit the data perfectly (bad)

with diffusion, and more generally generative models, no two data-points are the same => technically infinite data

Think of the simulator (or real-life) as simply being a function which takes inputs, x, and produces outputs, y. We train a policy to approximate this function.

If we trained a policy to predict auxiliary outputs, like FK, then it'll do so but it'll be worse or equal to the closed-form equation so imo it doesn't make sense to make any task which has a closed form solution (like FK) an auxiliary task. Instead those values should just be computed and added to the list of inputs

USE ML (implicit):
y = real-life(x) -- no easy closed form equation --> use ML
y = simulator(x) -- no easy closed form equation --> use ML
euler equations (CFD) -- no closed form equation (but can simulate) --> use ML
force closure = fcn(s, obs) IFF we can not (or don't want to) track objects
integrate touch + FK pos into 3d meshes / object recognition
predict next state

DON'T USE ML (explicit)
y = FK(x) -- closed form soln --> calculate explicitly and add to state
jerk = fcn(s, a)
force closure = fcn(s, obs) IFF we can track objects

desired properties:
* parameters which solve task A also solve task B
* strong correlation between the math of task A and the math of task B

ways to compute:
* calculate gradient of A or B w.r.t. parameter
* task performance of A when trained with/without B
* task performance of B when trained with/without A

references:
"Auxiliary task discovery through generate-and-test"
"AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning"
https://www.activeloop.ai/resources/glossary/auxiliary-tasks/ 
"""
