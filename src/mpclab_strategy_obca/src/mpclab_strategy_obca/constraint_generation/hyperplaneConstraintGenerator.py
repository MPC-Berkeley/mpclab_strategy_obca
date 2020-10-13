#!/usr/bin python3

import numpy as np
import numpy.linalg as la
from numpy import sin, cos, arctan, arctan2

from mpclab_strategy_obca.constraint_generation.abstractConstraintGenerator import abstractConstraintGenerator
from mpclab_strategy_obca.utils.types import experimentParams

class hyperplaneConstraintGenerator(abstractConstraintGenerator):
    def __init__(self, exp_params=experimentParams()):
        self.L = exp_params.car_L
        self.W = exp_params.car_W
        self.r = exp_params.collision_buffer_r

        self.L_p = self.L/2
        self.L_m = self.L/2

    def generate_constraint(self, global_collision_pos, global_car_state, global_projection_direction, collision_buffer_scaling=1.0):
        assert collision_buffer_scaling >= 1, 'Collision buffer inflation scale must be >= 1'

        collision_x = global_collision_pos[0]
        collision_y = global_collision_pos[1]

        car_x = global_car_state[0]
        car_y = global_car_state[1]
        car_heading = global_car_state[2]
        car_v = global_car_state[3]

        R = self._get_rotation_matrix(car_heading)

        diff = np.array([collision_x-car_x, collision_y-car_y])
        local_collision_pos = la.solve(R, diff)

        local_projection_direction = la.solve(R, global_projection_direction)
        local_projection_angle = np.arctan2(local_projection_direction[1],local_projection_direction[0])

        v_direction = self._sign(car_v)
        if v_direction > 0:
            self.L_p = collision_buffer_scaling*self.L/2
            self.L_m = self.L/2
        else:
            self.L_p = self.L/2
            self.L_m = collision_buffer_scaling*self.L/2

        local_car_edge_pos, local_car_edge_tangent, local_collision_bound_pos = self._project_to_car_edge(local_collision_pos, local_projection_angle)

        local_hyp_w, local_hyp_b = local_car_edge_tangent, local_car_edge_tangent.dot(local_car_edge_pos)

        hyp_xy = R.dot(local_car_edge_pos) + np.array([car_x, car_y])
        hyp_w = R.dot(local_hyp_w)
        hyp_b = hyp_w.dot(np.array([car_x, car_y])) + local_hyp_b

        coll_xy = R.dot(local_collision_bound_pos) + np.array([car_x, car_y])

        return hyp_xy, hyp_w, hyp_b, coll_xy

    def check_collision_points(self, global_collision_pos, global_car_state, collision_buffer_scaling=None):
        N = global_collision_pos.shape[0]

        if collision_buffer_scaling is None:
            collision_buffer_scaling = np.ones(N)

        collisions = [False for _ in range(N)]

        A = np.vstack((np.eye(2),-np.eye(2)))

        for i in range(N):
            s = collision_buffer_scaling[i]

            car_x = global_car_state[i,0]
            car_y = global_car_state[i,1]
            car_heading = global_car_state[i,2]
            car_v = global_car_state[i,3]

            car_pos = np.array([car_x, car_y])
            R = self._get_rotation_matrix(car_heading)
            car_A = A.dot(R.T)
            t = car_A.dot(car_pos)

            v_direction = self._sign(car_v)
            if v_direction > 0:
                L_p = s*self.L/2
                L_m = self.L/2
            else:
                L_p = self.L/2
                L_m = s*self.L/2

            body_b = np.array([L_p, self.W/2, L_m, self.W/2])
            front_b = np.array([L_p+self.r, self.W/2, -L_p, self.W/2])
            back_b = np.array([-L_m, self.W/2, L_m+self.r, self.W/2])
            left_b = np.array([L_p, self.W/2+self.r, L_m, -self.W/2])
            right_b = np.array([L_p, -self.W/2, L_m, self.W/2+self.r])

            fl = np.array([L_p, self.W/2])
            fr = np.array([L_p, -self.W/2])
            bl = np.array([-L_m, self.W/2])
            br = np.array([-L_m, -self.W/2])

            car_body_b = body_b + t
            car_front_b = front_b + t
            car_back_b = back_b + t
            car_left_b = left_b + t
            car_right_b = right_b + t

            car_fl = R.dot(fl) + car_pos
            car_fr = R.dot(fr) + car_pos
            car_bl = R.dot(bl) + car_pos
            car_br = R.dot(br) + car_pos

            z = car_A.dot(global_collision_pos[i])
            if np.all(z-car_body_b <= 0) or \
                np.all(z-car_front_b <= 0) or \
                np.all(z-car_back_b <= 0) or \
                np.all(z-car_left_b <= 0) or \
                np.all(z-car_right_b <= 0) or \
                (la.norm(global_collision_pos[i]-car_fl) <= self.r) or \
                (la.norm(global_collision_pos[i]-car_fr) <= self.r) or \
                (la.norm(global_collision_pos[i]-car_bl) <= self.r) or \
                (la.norm(global_collision_pos[i]-car_br) <= self.r):
                collisions[i] =True

        return collisions

    def _project_to_car_edge(self, pos, projection_angle):
        x, y, phi = pos[0], pos[1], projection_angle

        if x > self.L_p+self.r or x < -(self.L_m+self.r):
            raise RuntimeError('Given point is outside of collision boundary')

        if phi >= 2*np.pi or phi < 0:
            phi = np.mod(phi, 2*np.pi)

        if np.abs(y) <= self.W/2:
            a_1 = arctan2(self.W/2-y,            self.L_p+self.r-x)
            a_2 = arctan2(self.W/2+self.r-y,     self.L_p-x)
            a_3 = arctan2(self.W/2+self.r-y,     -self.L_m-x)
            a_4 = arctan2(self.W/2-y,            -self.L_m-self.r-x)
            a_5 = arctan2(-self.W/2-y,           -self.L_m-self.r-x)
            a_6 = arctan2(-self.W/2-self.r-y,    -self.L_m-x)
            a_7 = arctan2(-self.W/2-self.r-y,    self.L_p-x)
            a_8 = arctan2(-self.W/2-y,           self.L_p+self.r-x)

            if (phi >= 0 and phi < a_1) or (phi >= 2*np.pi+a_8 and phi < 2*np.pi):
                equiv_phi = 0

            elif phi >= a_1 and phi < a_2:
                m = sin(phi)*(self.L_p-x) - cos(phi)*(self.W/2-y)
                p = np.array([1, 2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi <= np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = arctan(b/a)

            elif phi >= a_2 and phi < a_3:
                equiv_phi = np.pi/2

            elif phi >= a_3 and phi < a_4:
                m = sin(phi)*(-self.L_m-x) - cos(phi)*(self.W/2-y)
                p = np.array([1, -2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi >= np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = np.pi - arctan(b/a)

            elif phi >= a_4 and phi < 2*np.pi+a_5:
                equiv_phi = np.pi

            elif phi >= 2*np.pi+a_5 and phi < 2*np.pi+a_6:
                m = cos(phi)*(-self.W/2-y) - sin(phi)*(-self.L_m-x)
                p = np.array([1, 2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi <= 3*np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = np.pi + arctan(b/a)

            elif phi >= 2*np.pi+a_6 and phi < 2*np.pi+a_7:
                equiv_phi = 3*np.pi/2

            elif phi >= 2*np.pi+a_7 and phi < 2*np.pi+a_8:
                m = cos(phi)*(-self.W/2-y) - sin(phi)*(self.L_p-x)
                p = np.array([1, -2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi >= 3*np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = 2*np.pi - arctan(b/a)

        elif y > self.W/2 and y <= self.W/2+self.r:
            a_1 = arctan2(self.W/2+self.r-y,     self.L_p-x)
            a_2 = arctan2(self.W/2+self.r-y,     -self.L_m-x)
            a_3 = arctan2(self.W/2-y,            -self.L_m-self.r-x)
            a_4 = arctan2(-self.W/2-y,           -self.L_m-self.r-x)
            a_5 = arctan2(-self.W/2-self.r-y,    -self.L_m-x)
            a_6 = arctan2(-self.W/2-self.r-y,    self.L_p-x)
            a_7 = arctan2(-self.W/2-y,           self.L_p+self.r-x)
            a_8 = arctan2(self.W/2-y,            self.L_p+self.r-x)

            if (phi >= 0 and phi < a_1) or (phi >= 2*np.pi+a_8 and phi < 2*np.pi):
                m = sin(phi)*(self.L_p-x) - cos(phi)*(self.W/2-y)
                p = np.array([1, 2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi <= np.pi/2 or phi >= 3*np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = arctan(b/a)

            elif phi >= a_1 and phi < a_2:
                equiv_phi = np.pi/2

            elif phi >= a_2 and phi < 2*np.pi+a_3:
                m = sin(phi)*(-self.L_m-x) - cos(phi)*(self.W/2-y)
                p = np.array([1, -2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi >= np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = np.pi - arctan(b/a)

            elif phi >= 2*np.pi+a_3 and phi < 2*np.pi+a_4:
                equiv_phi = np.pi

            elif phi >= 2*np.pi+a_4 and phi < 2*np.pi+a_5:
                m = cos(phi)*(-self.W/2-y) - sin(phi)*(-self.L_m-x)
                p = np.array([1, 2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi <= 3*np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = np.pi + arctan(b/a)

            elif phi >= 2*np.pi+a_5 and phi < 2*np.pi+a_6:
                equiv_phi = 3*np.pi/2

            elif phi >= 2*np.pi+a_6 and phi < 2*np.pi+a_7:
                m = cos(phi)*(-self.W/2-y) - sin(phi)*(self.L_p-x)
                p = np.array([1, -2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi >= 3*np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = 2*np.pi - arctan(b/a)

            elif phi >= 2*np.pi+a_7 and phi < 2*np.pi+a_8:
                equiv_phi = 0

        elif y < -self.W/2 and y >= -self.W/2-self.r:
            a_1 = arctan2(-self.W/2-y,           self.L_p+self.r-x)
            a_2 = arctan2(self.W/2-y,            self.L_p+self.r-x)
            a_3 = arctan2(self.W/2+self.r-y,     self.L_p-x)
            a_4 = arctan2(self.W/2+self.r-y,     -self.L_m-x)
            a_5 = arctan2(self.W/2-y,            -self.L_m-self.r-x)
            a_6 = arctan2(-self.W/2-y,           -self.L_m-self.r-x)
            a_7 = arctan2(-self.W/2-self.r-y,    -self.L_m-x)
            a_8 = arctan2(-self.W/2-self.r-y,    self.L_p-x)

            if (phi >= 0 and phi < a_1) or (phi >= 2*np.pi+a_8 and phi < 2*np.pi):
                m = cos(phi)*(-self.W/2-y) - sin(phi)*(self.L_p-x)
                p = np.array([1, -2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi >= 3*np.pi/2 or phi <= np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = 2*np.pi - arctan(b/a)

            elif phi >= a_1 and phi < a_2:
                equiv_phi = 0

            elif phi >= a_2 and phi < a_3:
                m = sin(phi)*(self.L_p-x) - cos(phi)*(self.W/2-y)
                p = np.array([1, 2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi <= np.pi/2 or phi >= 3*np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = arctan(b/a)

            elif phi >= a_3 and phi < a_4:
                equiv_phi = np.pi/2

            elif phi >= a_4 and phi < a_5:
                m = sin(phi)*(-self.L_m-x) - cos(phi)*(self.W/2-y)
                p = np.array([1, -2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi >= np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = np.pi - arctan(b/a)

            elif phi >= a_5 and phi < a_6:
                equiv_phi = np.pi

            elif phi >= a_6 and phi < 2*np.pi+a_7:
                m = cos(phi)*(-self.W/2-y) - sin(phi)*(-self.L_m-x)
                p = np.array([1, 2*sin(phi)*m, m**2-cos(phi)**2*self.r**2])
                if phi <= 3*np.pi/2:
                    a = np.amax(np.roots(p))
                else:
                    a = np.amin(np.roots(p))
                b = np.sqrt(self.r**2-a**2)
                equiv_phi = np.pi + arctan(b/a)

            elif phi >= 2*np.pi+a_7 and phi < 2*np.pi+a_8:
                equiv_phi = 3*np.pi/2

        else:
            raise RuntimeError('Given point is outside of collision boundary')


        phi = equiv_phi
        if phi >= 2*np.pi or phi < 0:
            phi = np.mod(phi, 2*np.pi)

        if phi < np.pi/2 and phi >= 0:
            x_v = self.L_p
            y_v = self.W/2
        elif phi < np.pi and phi >= np.pi/2:
            x_v = -self.L_m
            y_v = self.W/2
        elif phi < 3*np.pi/2 and phi >= np.pi:
            x_v = -self.L_m
            y_v = -self.W/2
        else:
            x_v = self.L_p
            y_v = -self.W/2

        x_b = x_v + self.r*cos(phi)
        y_b = y_v + self.r*sin(phi)

        car_edge_tangent = 2*np.array([x_b-x_v, y_b-y_v])
        car_edge_pos = np.array([x_v, y_v])
        collision_bound_pos = np.array([x_b, y_b])

        return car_edge_pos, car_edge_tangent, collision_bound_pos

    def _get_rotation_matrix(self, theta):
        return np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])

    def _sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    import pdb
    import time

    L, W, r = 4.0, 1.5, 1.0
    params = experimentParams(car_L=L, car_W=W, collision_buffer_r=r)
    hyp_gen = hyperplaneConstraintGenerator(params)

    coll_pos = np.array([1, 0.5])

    car_x, car_y, car_heading, car_v = 0, 0, 5*np.pi/4, 1
    car_state = np.array([car_x,car_y,car_heading,car_v])

    proj_dir = np.array([0,1])
    scaling = 2.0
    hyp_xy, hyp_w, hyp_b, _ = hyp_gen.generate_constraint(coll_pos, car_state, proj_dir, scaling)
    plot_x = np.array([-4, 4])
    plot_y = (-hyp_w[0]*plot_x+hyp_b)/hyp_w[1]

    theta = np.linspace(0, 2*np.pi, 100)
    bound_x = []
    bound_y = []
    for t in theta:
        d = np.array([cos(t),sin(t)])
        _, _, _, xy = hyp_gen.generate_constraint(np.zeros(2), car_state, d, scaling)
        bound_x.append(xy[0])
        bound_y.append(xy[1])

    test_pts = np.vstack((np.linspace(-4, 4, 20), -2*np.ones(20))).T
    # test_pts = np.vstack((np.linspace(-4, 4, 20), np.ones(20))).T
    t_s = time.time()
    collisions = hyp_gen.check_collision_points(test_pts, np.tile(car_state, (20,1)), scaling*np.ones(20))
    print(time.time() - t_s)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    car = patches.Rectangle((-0.5*L,-0.5*W), L, W, color='blue')
    scaled_car = patches.Rectangle((-hyp_gen.L_m,-0.5*W), hyp_gen.L_m+hyp_gen.L_p, W, fill=False, color='k', linestyle='--')
    ax.add_patch(car)
    ax.add_patch(scaled_car)

    R = Affine2D().rotate_around(car_x, car_y, car_heading) + ax.transData
    car.set_xy((car_x-0.5*L, car_y-0.5*W))
    car.set_transform(R)
    scaled_car.set_xy((car_x-hyp_gen.L_m, car_y-0.5*W))
    scaled_car.set_transform(R)

    ax.plot(coll_pos[0], coll_pos[1], 'rx')
    ax.plot([coll_pos[0],coll_pos[0]+proj_dir[0]], [coll_pos[1],coll_pos[1]+proj_dir[1]], 'r')
    ax.plot(hyp_xy[0], hyp_xy[1], 'go')
    ax.plot(plot_x, plot_y, 'g')
    ax.plot(bound_x, bound_y, 'b')
    for i in range(test_pts.shape[0]):
        if collisions[i]:
            ax.plot(test_pts[i,0], test_pts[i,1], 'ro')
        else:
            ax.plot(test_pts[i,0], test_pts[i,1], 'go')

    plt.show(block=False)
    plt.draw()
    pdb.set_trace()
