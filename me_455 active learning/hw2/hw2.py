
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# import circles
import matplotlib.patches as patches
from matplotlib.patches import Circle

import torch as th

def ok_actually_calculate_local_sensor_probs():
    # infotaxis probabilities
    sx, sy = 0, 0  # (n_sensors,)

    minsx = 0
    maxsx = 3 + 1 + 3

    minsy = 0
    maxsy = 7

    sensor_w = maxsx - minsx
    sensor_h = maxsy - minsy

    local_sensor_grid = np.zeros((sensor_w, sensor_h))

    # get the sensor xy
    sensor_XY_list = np.stack(np.meshgrid(np.arange(sensor_w), np.arange(sensor_h)), axis=-1)  # (sensor_w, sensor_h, 2)
    sensor_XY_list = sensor_XY_list.reshape(-1, 2)  # (sensor_w*sensor_h, 2)

    for sensor_XY in sensor_XY_list:
        sensor_x, sensor_y = sensor_XY

        x_dist_to_s = int(np.abs(sensor_x - sx))
        y_dist_to_s = int(np.abs(sensor_y - sy))

        prob = 1.0 / (y_dist_to_s + 1)

        if x_dist_to_s <= y_dist_to_s:
            local_sensor_grid[sensor_x, sensor_y] = prob

    return local_sensor_grid

def calculate_local_sensor_probabilities(grid, sensor_pos):
    width, height = grid.shape

    # infotaxis probabilities
    sx, sy = sensor_pos[0], sensor_pos[1]  # (n_sensors,)

    minsx = max(0, sx - 3)
    maxsx = min(width-1, sx + 3)

    minsy = max(0, sy - 3)
    maxsy = min(height-1, sy + 3)

    sensor_w = maxsx - minsx
    sensor_h = maxsy - minsy

    local_sensor_grid = np.zeros((sensor_w, sensor_h))

    # get the sensor xy
    sensor_XY_list = np.stack(np.meshgrid(np.arange(sensor_w), np.arange(sensor_h)), axis=-1)  # (sensor_w, sensor_h, 2)
    sensor_XY_list = sensor_XY_list.reshape(-1, 2)  # (sensor_w*sensor_h, 2)

    for sensor_XY in sensor_XY_list:
        sensor_x, sensor_y = sensor_XY

        x_dist_to_s = int(np.abs(sensor_x - sx))
        y_dist_to_s = int(np.abs(sensor_y - sy))

        prob = 1.0 / (y_dist_to_s + 1)

        if x_dist_to_s <= y_dist_to_s:
            local_sensor_grid[sensor_x, sensor_y] = prob

    return local_sensor_grid

def calculate_sensor_probabilities(grid, sensor_pos):
    local_sensor_grid = calculate_local_sensor_probabilities(grid, sensor_pos)

    width, height = grid.shape

    # infotaxis probabilities
    sx, sy = sensor_pos[0], sensor_pos[1]  # (n_sensors,)

    minsx = max(0, sx - 3)
    maxsx = min(width-1, sx + 3)

    minsy = max(0, sy - 3)
    maxsy = min(height-1, sy + 3)

    
    sensor_grid = np.zeros_like(grid)
    sensor_grid[minsx:maxsx, minsy:maxsy] = local_sensor_grid

    return sensor_grid

def measure(sensor_grid, A):
    # A: robot/sensor location

    # get the prob
    prob = sensor_grid[A[0], A[1]]

    # prob is the prob of getting a 1.0
    if np.random.rand() < prob:
        return 1.0
    else:
        return 0.0
    
def get_action(prior, robot_location):
    """
    as a simple algorithm, just toward the highest prob location
    """
    argmax = np.unravel_index(np.argmax(prior), prior.shape)

    if argmax[0] < robot_location[0]:
        return 'left'
    elif argmax[0] > robot_location[0]:
        return 'right'
    elif argmax[1] < robot_location[1]:
        return 'down'
    elif argmax[1] > robot_location[1]:
        return 'up'
    else:
        return 'stay'
    
def compute_conditional(measurement, robot_location, grid_shape):
    """
    intuition: if we got nothing, then assume we're not in the dumbbell.

    must compute p(sensor location | measurement)
    """
    # must compute for all possible sensor locations
    n_width, n_height = grid_shape

    # make xy list
    XY_list = np.stack(np.meshgrid(np.arange(n_width), np.arange(n_height)), axis=-1)  # (n_width, n_height, 2)

    cond_prob = np.zeros((n_width, n_height), dtype=np.float32)

    # no information if measurement was negative???? idk
    # if not measurement:
    #     return cond_prob

    # goes from 0 to 7, 0 to 7
    local_sensor_probs = ok_actually_calculate_local_sensor_probs()

    def update_cond_prob(p, sx, sy):
        # check if valid sx, sy
        if sx > 0 and sx < n_width and sy > 0 and sy < n_height:
            pass
        else:
            return

        if measurement:
            cond_prob[sx, sy] += p
        else:
            cond_prob[sx, sy] += 1 - p

    x, y = robot_location
    # well, let's start with being IF the sensor was right on top
    sx, sy = x, y
    p = 1.0
    update_cond_prob(p, sx, sy)

    # now if we were +0 x, +1 y
    sx, sy = x, y-1
    p = 0.5
    update_cond_prob(p, sx, sy)
    update_cond_prob(p, sx, x+1)

    # +0x, +2 y
    update_cond_prob(0.333, x, y-2)
    update_cond_prob(0.333, x, y+2)

    # +0x, +3 y
    update_cond_prob(0.25, x, y-3)
    update_cond_prob(0.25, x, y+3)

    # +-1x, +-1y
    update_cond_prob(0.5, x+1, y+1)
    update_cond_prob(0.5, x-1, y+1)
    update_cond_prob(0.5, x+1, y-1)
    update_cond_prob(0.5, x-1, y-1)

    # the rest
    update_cond_prob(0.333, x+1, y+2)
    update_cond_prob(0.333, x+1, y-2)
    update_cond_prob(0.333, x-1, y+2)
    update_cond_prob(0.333, x-1, y-2)

    update_cond_prob(0.333, x+2, y+2)
    update_cond_prob(0.333, x+2, y-2)
    update_cond_prob(0.333, x-2, y+2)
    update_cond_prob(0.333, x-2, y-2)


    update_cond_prob(0.25, x+1, y+3)
    update_cond_prob(0.25, x+1, y-3)
    update_cond_prob(0.25, x-1, y+3)
    update_cond_prob(0.25, x-1, y-3)

    update_cond_prob(0.25, x+2, y+3)
    update_cond_prob(0.25, x+2, y-3)
    update_cond_prob(0.25, x-2, y+3)
    update_cond_prob(0.25, x-2, y-3)

    update_cond_prob(0.25, x+3, y+3)
    update_cond_prob(0.25, x+3, y-3)
    update_cond_prob(0.25, x-3, y+3)
    update_cond_prob(0.25, x-3, y-3)

    # re-normalize
    cond_prob /= np.linalg.norm(cond_prob)

    # for plotting
    cond_prob_plot = cond_prob.copy()
    cond_prob_plot[cond_prob_plot < 0.01] = 0.0
    cond_prob_plot /= np.max(cond_prob_plot)

    return cond_prob



def p1():
    n_width = 25
    n_height = 25
    n_cells = n_width * n_height

    grid = np.zeros((n_width, n_height), dtype=np.float32)
    grid_shape = grid.shape

    grid_xy = np.stack(np.meshgrid(np.arange(n_width), np.arange(n_height)), axis=-1)  # (n_width, n_height, 2)

    # randomly sample a g.t. sensor location, in indices
    gt_sensor_location = np.random.random(size=(2,)) * np.array([n_width, n_height])
    gt_sensor_location = gt_sensor_location.astype(int)

    # make the sensor probs grid. this is the g.t. so we're not allowed to use it in the update
    sensor_probs_grid = calculate_sensor_probabilities(grid, gt_sensor_location)

    # randomly smaple the initial robot location
    robot_location = np.random.random(size=(2,)) * np.array([n_width, n_height])
    robot_location = robot_location.astype(int)

    actions = ['up', 'down', 'left', 'right', 'stay']

    # uniform prior of the sensor location over the space
    sensor_location_prior = np.ones((n_width, n_height), dtype=np.float32) / n_cells

    # control loop
    posteriors = []
    done = False
    while not done:
        # sample an action
        action = get_action(sensor_location_prior, robot_location)

        # move
        if action == 'up':
            robot_location[1] = max(0, robot_location[1] + 1)
        elif action == 'down':
            robot_location[1] = min(n_height - 1, robot_location[1] - 1)
        elif action == 'left':
            robot_location[0] = max(0, robot_location[0] - 1)
        elif action == 'right':
            robot_location[0] = min(n_width - 1, robot_location[0] + 1)

        # assert the action was valid, otherwise there's an error in our alg
        assert(robot_location[0] > 0 and robot_location[0] < n_width)
        assert(robot_location[1] > 0 and robot_location[1] < n_height)

        # measure
        measurement = measure(sensor_probs_grid, robot_location)

        # compute the conditional probability
        cond_prob = compute_conditional(measurement, robot_location, grid_shape)

        # bayes rule, update
        posterior = sensor_location_prior * cond_prob

        # normalize
        posterior = posterior / np.sum(posterior)

        # save values
        posteriors.append(posterior)

        # set for next round
        sensor_location_prior = posterior

        # for plotting
        posterior_for_plot = posterior.copy()
        posterior_for_plot[posterior_for_plot < 0.01] = 0.
        posterior_for_plot /= np.max(posterior_for_plot)

        # plot
        plt.imshow(posterior_for_plot.T, origin='lower')
        plt.scatter(gt_sensor_location[0], gt_sensor_location[1], c='red', label='gt sensor location')
        plt.scatter(robot_location[0], robot_location[1], c='blue', label='robot location')
        plt.title(f'Measurement: {measurement}, Action: {action}')
        plt.pause(0.2)




if __name__ == "__main__":
    p1()