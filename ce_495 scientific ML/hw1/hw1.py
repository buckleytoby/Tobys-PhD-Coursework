import numpy as np

import matplotlib.pyplot as plt

# 1d3 - f4(x, y) = x4 + y4 − 2x2 − 4y2
xx = np.linspace(-5, 5)
yy = np.linspace(-5, 5)

x, y = np.meshgrid(xx, yy)

f4 = x**4 + y**4 - 2*x**2 - 4*y**2

plt.contour(x, y, f4)
plt.savefig("./hw1_1d3.png", dpi=400)

# exercise 7, using the function from 1b
def grad(xx: np.ndarray):
    x, y = xx.T
    dx = 2*x + 2*y
    dy = 2*x + 6*y

    return np.array([dx, dy]).T

def f(xx: np.ndarray):
    """ f2(x, y) = x**2 + 2xy + 3y**2 """
    x, y = xx.T
    v = x**2 + 2*x*y + 3*y**2
    return v

def descend(x0: np.ndarray, alpha = 1e-1):
    nb = 10

    xx = x0.copy()
    history = {'xx': [],
               'f': [],
               'i': [],
    }


    for i in range(nb):
        history['i'].append(i)
        history['f'].append(f(xx))
        history['xx'].append(xx.copy())

        g = grad(xx)
        xx = xx - g * alpha


    history['i'].append(i+1) #type:ignore
    history['f'].append(f(xx))
    history['xx'].append(xx.copy())

    return history

# init
x0 = np.array([10, 10]).T
history = descend(x0, alpha = 2e-1)

i_s = history['i']
f_s = history['f']

plt.figure()
plt.scatter(i_s, f_s)
plt.savefig("./hw1_exercise7.png", dpi=400)

# exercise 8
class NewtonsMethod:
    """
    dk = -∇^2 f(xk)^-1 * ∇f(xk)
    x_k+1 = x_k + dk
    """
    def __init__(self,
                 obj_fn,
                 grad_fn,
                 hess_fn,
                 ):
        self.obj_fn = obj_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn

    def compute_alpha(self, xx):
        h = self.hess_fn(xx)

        # invert
        h2 = np.linalg.inv(h)

        # negate
        h3 = -h2

        return h3

    
    def descend(self, x0: np.ndarray):
        nb = 10

        xx = x0.copy()
        history = {'xx': [],
                'f': [],
                'i': [],
        }

        for i in range(nb):
            history['i'].append(i)
            history['f'].append(self.obj_fn(xx))
            history['xx'].append(xx.copy())

            g = self.grad_fn(xx)
            alpha = self.compute_alpha(xx)
            xx = xx + alpha @ g


        history['i'].append(i+1) #type:ignore
        history['f'].append(self.obj_fn(xx))
        history['xx'].append(xx.copy())

        return history
    
    def plot(self, history, label):
        i_s = history['i']
        f_s = history['f']

        plt.figure()
        plt.scatter(i_s, f_s)
        plt.savefig("hw1_" + label + ".png", dpi=400)
    


def obj_fn_b(xx: np.ndarray):
    """ f2(x, y) = x**2 + 2xy + 3y**2 """
    x, y = xx.T
    v = x**2 + 2*x*y + 3*y**2
    return v

def grad_fn_b(xx: np.ndarray):
    x, y = xx.T
    dx = 2*x + 2*y
    dy = 2*x + 6*y

    return np.array([dx, dy]).T

def hess_fn_b(xx: np.ndarray):
    x, y = xx.T

    h = np.array([[2, 2],
                  [2, 6]])
    return h

nmb = NewtonsMethod(obj_fn_b, grad_fn_b, hess_fn_b)
h = nmb.descend(x0)
nmb.plot(h, "nm_b")