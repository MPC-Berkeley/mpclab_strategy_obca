#!/usr/bin python3

import numpy as np
import time

from mpclab_strategy_obca.control.abstractController import abstractController
from mpclab_strategy_obca.control.utils.types import PIDParams

class PID(abstractController):
    def __init__(self, params=PIDParams()):
        self.dt             = params.dt

        self.Kp             = params.Kp             # proportional gain
        self.Ki             = params.Ki             # integral gain
        self.Kd             = params.Kd             # derivative gain

        # Integral action and control action saturation limits
        self.int_e_max      = params.int_e_max
        self.int_e_min      = params.int_e_min
        self.u_max          = params.u_max
        self.u_min          = params.u_min
        self.du_max         = params.du_max
        self.du_min         = params.du_min

        self.x_ref          = 0
        self.u_ref          = 0

        self.e              = 0             # error
        self.de             = 0             # finite time error difference
        self.ei             = 0             # accumulated error

        self.time = True

        self.initialized = False

    def initialize(self, x_ref=0, u_ref=0, de=0, ei=0, time=True):
        self.de = de
        self.ei = ei

        self.x_ref = x_ref         # reference point
        self.u_ref = u_ref         # control signal offset

        self.time = time

        self.initialized = True

    def solve(self, x, u_prev):
        if not self.initialized:
            raise(RuntimeError('PID controller is not initialized, run PID.initialize() before calling PID.solve()'))

        if self.time:
            t_s = time.time()

        info = {'success' : True}

        # Compute error terms
        e_t = x - self.x_ref
        de_t = (e_t - self.e)/self.dt
        ei_t = self.ei + e_t*self.dt

        # Anti-windup
        if ei_t > self.int_e_max:
            ei_t = self.int_e_max
        elif ei_t < self.int_e_min:
            ei_t = self.int_e_min

        # Compute control action terms
        P_val  = self.Kp * e_t
        I_val  = self.Ki * ei_t
        D_val  = self.Kd * de_t

        # Compute change in control action from previous timestep
        du = -(P_val + I_val + D_val) + self.u_ref - u_prev

        # Saturate change in control action
        if self.du_max is not None:
            du = self._saturate_rel_high(du)
        if self.du_min is not None:
            du = self._saturate_rel_low(du)

        u = du + u_prev

        # Saturate absolute control action
        if self.u_max is not None:
            u = self._saturate_abs_high(u)
        if self.u_min is not None:
            u = self._saturate_abs_low(u)

        # Update error terms
        self.e  = e_t
        self.de = de_t
        self.ei = ei_t

        if self.time:
            info['solve_time'] = time.time() - t_s

        return u, info

    def set_x_ref(self, x, x_ref):
        self.x_ref = x_ref
        # reset error integrator
        self.ei = 0
        # reset error, otherwise de/dt will skyrocket
        self.e = x - x_ref

    def set_u_ref(self, u_ref):
        self.u_ref = u_ref

    def clear_errors(self):
        self.ei = 0
        self.de = 0

    def set_params(self, params):
        self.dt             = params.dt

        self.Kp             = params.Kp             # proportional gain
        self.Ki             = params.Ki             # integral gain
        self.Kd             = params.Kd             # derivative gain

        # Integral action and control action saturation limits
        self.int_e_max      = params.int_e_max
        self.int_e_min      = params.int_e_min
        self.u_max          = params.u_max
        self.u_min          = params.u_min
        self.du_max         = params.du_max
        self.du_min         = params.du_min

    def get_refs(self):
        return (self.x_ref, self.u_ref)

    def get_errors(self):
        return (self.e, self.de, self.ei)

    def _saturate_abs_high(self, u):
        return np.minimum(u, self.u_max)

    def _saturate_abs_low(self, u):
        return np.maximum(u, self.u_min)

    def _saturate_rel_high(self, du):
        return np.minimum(du, self.du_max)

    def _saturate_rel_low(self, u):
        return np.maximum(du, self.du_min)

# Test script to ensure controller object is functioning properly
if __name__ == "__main__":
    import pdb

    params = PIDParams(dt=0.1, Kp=3.7, Ki=7, Kd=0.5)
    x_ref = 5
    pid = PID(params)
    # pdb.set_trace()
    pid.initialize(x_ref=x_ref)
    # pdb.set_trace()

    print('Controller instantiated successfully')
