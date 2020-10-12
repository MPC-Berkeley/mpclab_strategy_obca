#!/usr/bin python3

import numpy as np
from numpy import sin, cos
from pypoman import compute_polytope_halfspaces, compute_polytope_vertices, intersect_polygons

import pdb

def check_collision_poly(z_1, dim_1, z_2, dim_2):
    V_1 = get_car_verts(z_1, dim_1[0], dim_1[1])
    V_2 = get_car_verts(z_2, dim_2[0], dim_2[1])

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
        V = get_car_verts(Z[i], W, L)
        A, b = compute_polytope_halfspaces(V)
        obs.append({'A': A, 'b': b})

    return obs

def get_car_verts(z, W, L):
    x = z[0]
    y = z[1]
    heading = z[2]

    V_x = [x + L/2*cos(heading) - W/2*sin(heading),
           x + L/2*cos(heading) + W/2*sin(heading),
           x - L/2*cos(heading) + W/2*sin(heading),
           x - L/2*cos(heading) - W/2*sin(heading)]

    V_y = [y + L/2*sin(heading) + W/2*cos(heading),
           y + L/2*sin(heading) - W/2*cos(heading),
           y - L/2*sin(heading) - W/2*cos(heading),
           y - L/2*sin(heading) + W/2*cos(heading)]

    V = np.vstack((V_x, V_y)).T
    return V

if __name__ == '__main__':
    W = 2
    L = 4

    s_1 = np.array([[0,0,0,0]])
    s_2 = np.array([[1,0,0,0]])
    obs_1 = get_car_poly(s_1, W, L)
    obs_2 = get_car_poly(s_2, W, L)

    collision = check_collision_poly(s_1[0], (W, L), s_2[0], (W, L))
    print(collision)
    pdb.set_trace()
