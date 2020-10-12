#!/usr/bin python3

import numpy as np
from numpy import sin, cos
from pypoman import compute_polytope_halfspaces

import pdb

def check_collision_poly(P_1, P_2):
    pass

def get_car_poly(Z, W, L):
    # Function to get the halfspace representation of a rectangular car given its states and dimensions
    # Z: np.array of states with shape N x n_x
    # W, L: width and length of car
    N = Z.shape[0]
    obs = []

    for i in range(N):
        x = Z[i,0]
        y = Z[i,1]
        heading = Z[i,2]

        V_x = [x + L/2*cos(heading) - W/2*sin(heading),
               x + L/2*cos(heading) + W/2*sin(heading),
               x - L/2*cos(heading) + W/2*sin(heading),
               x - L/2*cos(heading) - W/2*sin(heading)]

        V_y = [y + L/2*sin(heading) + W/2*cos(heading),
               y + L/2*sin(heading) - W/2*cos(heading),
               y - L/2*sin(heading) - W/2*cos(heading),
               y - L/2*sin(heading) + W/2*cos(heading)]

        A, b = compute_polytope_halfspaces(np.vstack((V_x, V_y)).T)
        obs.append({'A': A, 'b': b})

    return obs

if __name__ == '__main__':
    states = np.random.rand(10,4)
    W = 2
    L = 4

    obs = get_car_poly(states, W, L)
