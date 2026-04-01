


import numpy as np
import matplotlib.pyplot as plt

import torch as th

"""

(z|x; s) =
(
exp 100 · (kxsk  0.2) 2  , if z = P ositive
1  exp 100 · (kxsk  0.2) 2  , if z = N egative
"""

def p(z, x, s):
    xsabs = np.linalg.norm(x-s)

    if z > 0.0:
        pz = np.exp(-100*(xsabs - 0.2**2))
    else:
        pz = 1 - np.exp(-100*(xsabs - 0.2**2))

    return pz

def sample(x, s):
    """
    draw a sample uniformly between [0, 1] and see if the sample value is smaller than p(Positive|x; s). If this is true, you get a Positive reading; otherwise, you get a Negative reading
    """
    z = np.random.rand()

    pz = p(z, x, s)

    if z < pz:
        return True
    else:
        return False
    
def p1():
    s = [0.3, 0.4]

    x = np.random.uniform(0, 1, (100, 2))
    measurements = np.array([sample(xx, s) for xx in x])

    pos = measurements > 0.0
    neg = ~pos

    pos_x = x[pos]
    neg_x = x[neg]

    # plot it 
    plt.figure()
    plt.scatter(pos_x[:, 0], pos_x[:, 1], color="g")
    plt.scatter(neg_x[:, 0], neg_x[:, 1], color="r")
    plt.show()

    # visualize ring shape

    return x, measurements

def p2():
    x, measurements = p1()

    def loss(z, x, s):
        # get prob(z given x, s)

        l = np.prod(probs)
        return l
    
    nb_cells = 100
    xs = np.linspace(0, 1, nb_cells)
    ys = np.linspace(0, 1, nb_cells)

    # meshgrid it
    X, Y = np.meshgrid(xs, ys)

    ls = np.zeros((nb_cells, nb_cells))

    ls = np.array([[loss(measurements, x, [X[i, j], Y[i, j]]) for j in range(nb_cells)] for i in range(nb_cells)])

    # plot the likelihood function
    plt.figure()
    plt.contourf(X, Y, ls, levels=50, cmap="viridis")
    plt.colorbar()
    plt.xlabel("s_x")
    plt.ylabel("s_y")
    plt.title("Likelihood Function")
    plt.show()

def p3():
    xs = np.array([[0.5, 0.5]])

def p4():
    


if __name__ == "__main__":
    p1()

