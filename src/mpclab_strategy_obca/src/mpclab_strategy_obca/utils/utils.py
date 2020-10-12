#!/usr/bin python3

import numpy as np
from numpy import sin, cos
from pypoman import compute_polytope_halfspaces, compute_polytope_vertices, intersect_polygons

import pdb

def check_collision_poly(P_1, P_2):
    A_1, b_1 = P_1['A'], P_1['b']
    A_2, b_2 = P_2['A'], P_2['b']

    V_1 = compute_polytope_vertices(A_1, b_1)
    V_2 = compute_polytope_vertices(A_2, b_2)

    isect = intersect_polygons(V_1, V_2)

    if not isect:
        return False

    return True

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
    W = 2
    L = 4

    s_1 = np.array([[0,0,0,0]])
    s_2 = np.array([[4,0,0,0]])
    obs_1 = get_car_poly(s_1, W, L)
    obs_2 = get_car_poly(s_2, W, L)

    collision = check_collision_poly(obs_1[0], obs_2[0])
    pdb.set_trace()
