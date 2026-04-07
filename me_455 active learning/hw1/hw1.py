
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# import circles
import matplotlib.patches as patches
from matplotlib.patches import Circle

import torch as th

"""

(z|x; s) =
(
exp 100 · (kxsk  0.2) 2  , if z = P ositive
1  exp 100 · (kxsk  0.2) 2  , if z = N egative
"""

def p_pos(x, s):
    if s.ndim > 1:
        xsabs = np.linalg.norm(x[None, :] - s, axis=-1)
    else:
        xsabs = np.linalg.norm(x-s)
    return np.exp(-100*((xsabs - 0.2)**2))

def p_neg(x, s):
    return 1 - p_pos(x, s)

def sample(x, s):
    """
    draw a sample uniformly between [0, 1] and see if the sample value is smaller than p(Positive|x; s). If this is true, you get a Positive reading; otherwise, you get a Negative reading
    
    x: sensor location
    s: source location
    """
    z = np.random.rand()

    p_pos_val = p_pos(x, s)

    if z < p_pos_val:
        return True
    else:
        return False
    
def measure(x, s):
    return sample(x, s)

def conditional_probability(z, x, s):
    p_z_given_x_colon_s = p_pos(x, s) if z else p_neg(x, s)
    
    return p_z_given_x_colon_s
    
def likelihood(zs, xs, _colon, s):
    # get prob(z given x; s)
    N = zs.shape[0]
    
    
    probs = []
    for i in range(N):
        zi = zs[i]
        xi = xs[i]
        
        if zi > 0.0:
            p = p_pos(xi, s)
        else:
            p = p_neg(xi, s)
        
        probs.append(p)
        
    probs = np.array(probs)

    l = np.prod(probs)
    return l
    
def p1(plot=True):
    s = np.array([0.3, 0.4])

    x = np.random.uniform(0, 1, (100, 2))
    measurements = np.array([sample(xx, s) for xx in x])

    if plot:
        pos = measurements > 0.0
        neg = ~pos

        pos_x = x[pos]
        neg_x = x[neg]

        # plot it 
        fig, ax = plt.subplots()
        radius = np.linspace(0.0, 1.0, 100)
        prs = []
        for r in radius:
            x = r + s
            pr = p_pos(x, s)
            prs.append(pr)
            
            
        # normalize
        prs = np.array(prs)
        prs = prs / np.max(prs)
        
        # enumerate
        for r, pr in reversed(list(zip(radius, prs))):
            # draw a circle centered at s with radius r and color intensity proportional to pr
            circle = Circle(tuple(s), r, color="b", fill=False, alpha=pr, linewidth=5)
            ax.add_patch(circle)
            
            
        plt.scatter(pos_x[:, 0], pos_x[:, 1], color="g", label="positive")
        plt.scatter(neg_x[:, 0], neg_x[:, 1], color="r", label="negative")

        # visualize ring shape
        # place a dot at the source location
        plt.scatter(s[0], s[1], color="b", label="source")
        plt.legend()
        
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # get this current file path
        current_file_path = pathlib.Path(__file__).parent
        
        plt.savefig(current_file_path / "hw1_p1.png", dpi=400)
        # plt.show()
    
    return x, measurements

def p2(plot=True):
    x, zs = p1(plot=False)
    
    nb_cells = 100
    sxs = np.linspace(0, 1, nb_cells)
    sys = np.linspace(0, 1, nb_cells)

    # meshgrid it
    SX, SY = np.meshgrid(sxs, sys)

    ls = np.zeros((nb_cells, nb_cells))
    
    for i in range(nb_cells):
        for j in range(nb_cells):
            sxy = np.array([SX[i, j], SY[i, j]])
            ls[i, j] = likelihood(zs, x, _colon=None, s=sxy)


    if plot:
        # plot the likelihood function
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.contourf(SX, SY, ls, levels=50, cmap="viridis")
        plt.colorbar()
        plt.xlabel("s_x")
        plt.ylabel("s_y")
        plt.title("Likelihood Function")
        
        
        
        # get this current file path
        current_file_path = pathlib.Path(__file__).parent
        
        plt.savefig(current_file_path / "hw1_p2.png", dpi=400)
        
        
        # plt.show()
        
    return ls

def p3(plot=True):
    s = np.array([0.3, 0.4]) # source location, don't change

    val = 0.5
    x = np.random.uniform(val, val, (100, 2))
    zs = np.array([sample(xx, s) for xx in x])
    
    
    nb_cells = 100
    sxs = np.linspace(0, 1, nb_cells)
    sys = np.linspace(0, 1, nb_cells)

    # meshgrid it
    SX, SY = np.meshgrid(sxs, sys)

    ls = np.zeros((nb_cells, nb_cells))
    
    for i in range(nb_cells):
        for j in range(nb_cells):
            sxy = np.array([SX[i, j], SY[i, j]])
            ls[i, j] = likelihood(zs, x, _colon=None, s=sxy)


    if plot:
        # plot the likelihood function
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.contourf(SX, SY, ls, levels=50, cmap="viridis")
        plt.colorbar()
        plt.xlabel("s_x")
        plt.ylabel("s_y")
        plt.title("Likelihood Function")
        
        
        
        # get this current file path
        current_file_path = pathlib.Path(__file__).parent
        
        plt.savefig(current_file_path / "hw1_p3.png", dpi=400)
        
        
        # plt.show()
        
    return ls


def p4(plot=True):
    s_true = np.array([0.3, 0.4]) # source location, don't change, we don't know it for this one
    x = np.array([0.5, 0.5]) # sensor location, stuck at this location
    
    nb_cells = 100
    
    # init my grid
    nb_cells2 = 100*100
    
    # init probs
    sxy_grid = np.ones((100, 100)) / nb_cells2
    
    # make a list of all the sxy locations
    sxs = np.linspace(0, 1, nb_cells)
    sys = np.linspace(0, 1, nb_cells)
    SX, SY = np.meshgrid(sxs, sys)
    sxy_grid_list = np.stack([SX, SY], axis=-1).reshape(-1, 2) # shape (10000, 2)
    
    # the sxy grid is my prior, uniform to begin with 
    prior = sxy_grid 
    
    n = 10
    ns_to_plot = np.arange(0, n, n//10)
    zs = []
    posteriors = []
    
    for i in range(n):
        zi = measure(x, s_true)
        
        p_zi_given_x_colon_s = conditional_probability(zi, x, sxy_grid_list)
        
        posterior = prior * p_zi_given_x_colon_s.reshape(nb_cells, nb_cells)
        
        # normalize
        posterior = posterior / np.sum(posterior)
        
        posteriors.append(posterior)
        zs.append(zi)
        
        # my new prior is my old posterior
        prior = posterior.copy()
        
        
        
    if plot:
        fig, axs = plt.subplots(2, 5, figsize=(20, 10))
        
        sxs = np.linspace(0, 1, nb_cells)
        sys = np.linspace(0, 1, nb_cells)

        # meshgrid it
        SX, SY = np.meshgrid(sxs, sys)
        
        for i in range(len(ns_to_plot)):
            n = ns_to_plot[i]
            zi = zs[n]
            posterior = posteriors[n].copy()
            
            # divide by the max to rescale to [0, 1]
            posterior = posterior / np.max(posterior)
            
            ax = axs[i//5, i%5]
            
            # the current belief
            ax.contourf(SX, SY, posterior, levels=50, cmap="viridis")
            
            
            # add measurement location & color
            if zi:
                ax.scatter(x[0], x[1], color="g", label="positive")
            else:
                ax.scatter(x[0], x[1], color="r", label="negative")
            
            
            ax.set_title(f"Posterior after {n+1} samples")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # add a scale
            # plt.colorbar(ax.contourf(SX, SY, posterior, levels=50, cmap="viridis"), ax=ax)
            
        plt.tight_layout()
        
        # get this current file path
        current_file_path = pathlib.Path(__file__).parent
        
        plt.savefig(current_file_path / "hw1_p4.png", dpi=400)
        # plt.show()
        
def p5(plot=True):
    s_true = np.array([0.3, 0.4]) # source location, don't change, we don't know it for this one
    x = np.array([0.5, 0.5]) # sensor location, initial location, can change
    
    nb_cells = 100
    
    # init my grid
    nb_cells2 = 100*100
    
    # init probs
    sxy_grid = np.ones((100, 100)) / nb_cells2
    
    # make a list of all the sxy locations
    sxs = np.linspace(0, 1, nb_cells)
    sys = np.linspace(0, 1, nb_cells)
    SX, SY = np.meshgrid(sxs, sys)
    sxy_grid_list = np.stack([SX, SY], axis=-1).reshape(-1, 2) # shape (10000, 2)
    
    # the sxy grid is my prior, uniform to begin with 
    prior = sxy_grid 
    
    n = 10
    ns_to_plot = np.arange(0, n, n//10)
    zs = []
    xs = []
    posteriors = []
    
    for i in range(n):
        zi = measure(x, s_true)
        
        p_zi_given_x_colon_s = conditional_probability(zi, x, sxy_grid_list)
        
        posterior = prior * p_zi_given_x_colon_s.reshape(nb_cells, nb_cells)
        
        # normalize
        posterior = posterior / np.sum(posterior)
        
        posteriors.append(posterior)
        zs.append(zi)
        xs.append(x.copy())
        
        # my new prior is my old posterior
        prior = posterior.copy()
        
        # new strategy, pick the sxy with the highest posterior prob as my new sensor location
        max_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
        x = np.array([SX[max_idx], SY[max_idx]])
        pass
        
        
    if plot:
        fig, axs = plt.subplots(2, 5, figsize=(20, 10))
        
        sxs = np.linspace(0, 1, nb_cells)
        sys = np.linspace(0, 1, nb_cells)

        # meshgrid it
        SX, SY = np.meshgrid(sxs, sys)
        
        for i in range(len(ns_to_plot)):
            n = ns_to_plot[i]
            zi = zs[n]
            posterior = posteriors[n].copy()
            
            # divide by the max to rescale to [0, 1]
            posterior = posterior / np.max(posterior)
            
            ax = axs[i//5, i%5]
            
            # the current belief
            ax.contourf(SX, SY, posterior, levels=50, cmap="viridis")
            
            
            # add measurement location & color
            x = xs[n]

            if zi:
                ax.scatter(x[0], x[1], color="g", label="positive")
            else:
                ax.scatter(x[0], x[1], color="r", label="negative")
            
            
            ax.set_title(f"Posterior after {n+1} samples")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # add a scale
            # plt.colorbar(ax.contourf(SX, SY, posterior, levels=50, cmap="viridis"), ax=ax)
            
        plt.tight_layout()
        
        # get this current file path
        current_file_path = pathlib.Path(__file__).parent
        
        plt.savefig(current_file_path / "hw1_p5.png", dpi=400)
        # plt.show()
        

if __name__ == "__main__":
    p1()
    p2()
    p3()
    p4()
    p5()

