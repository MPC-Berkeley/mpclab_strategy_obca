#!/usr/bin python3

from mpclab_strategy_obca.control.abstractController import abstractController
from mpclab_strategy_obca.control.PID import PID
from mpclab_strategy_obca.control.utils.types import safetyParams, PIDParams

class safetyController(abstractController):

    def __init__(self, params=safetyParams()):
        self.accel_pid_params = PIDParams(dt=params.dt,
            Kp=params.P_accel, Ki=params.I_accel, Kd=params.D_accel,
            int_e_max=params.accel_int_e_max, int_e_min=params.accel_int_e_min,
            u_max=params.accel_max, u_min=params.accel_min,
            du_max=params.daccel_max, du_min=params.daccel_min)
        self.steer_pid_params = PIDParams(dt=params.dt,
            Kp=params.P_speed, Ki=params.I_speed, Kd=params.D_speed,
            int_e_max=params.speed_int_e_max, int_e_min=params.speed_int_e_min,
            u_max=params.speed_max, u_min=params.speed_min,
            du_max=params.dspeed_max, du_min=params.dspeed_min)

        self.accel_ref = 0
        self.steer_ref = 0

        self.accel_pid = PID(self.accel_pid_params)
        self.accel_pid.initialize(x_ref=self.accel_ref)
        self.steer_pid = PID(self.steer_pid_params)
        self.steer_pid.initialize(x_ref=self.steer_ref)

    def initialize(self):
        pass

    def solve(self, z, Z_tv, last_u):
        x, y, heading, v = z
        x_tv, y_tv, heading_tv, v_tv = Z_tv[0]

        last_steer, last_accel = last_u

        accel, _ = self.accel_pid.solve(v, last_accel)
        if v < 0:
            steer, _ = self.steer_pid.solve(y_ev-10*heading_ev, last_steer)
        else:
            steer = 0

        u = [steer, accel]

        return u

    def set_accel_ref(self, accel_ref):
        self.accel_ref = accel_ref
        self.accel_pid.set_x_ref(self.accel_ref)

    def set_steer_ref(self, steer_ref):
        self.steer_ref = steer_ref
        self.steer_pid.set_x_ref(self.steer_ref)

class emergencyController(abstractController):

    def __init__(self, params=safetyParams()):
        self.accel_pid_params = PIDParams(dt=params.dt,
            Kp=params.P_accel, Ki=params.I_accel, Kd=params.D_accel,
            int_e_max=params.accel_int_e_max, int_e_min=params.accel_int_e_min,
            u_max=params.accel_max, u_min=params.accel_min,
            du_max=params.daccel_max, du_min=params.daccel_min)
        self.steer_pid_params = PIDParams(dt=params.dt,
            Kp=params.P_speed, Ki=params.I_speed, Kd=params.D_speed,
            int_e_max=params.speed_int_e_max, int_e_min=params.speed_int_e_min,
            u_max=params.speed_max, u_min=params.speed_min,
            du_max=params.dspeed_max, du_min=params.dspeed_min)

        self.accel_ref = 0
        self.steer_ref = 0

        self.accel_pid = PID(self.accel_pid_params)
        self.accel_pid.initialize(x_ref=self.accel_ref)
        self.steer_pid = PID(self.steer_pid_params)
        self.steer_pid.initialize(x_ref=self.steer_ref)

    def initialize(self):
        pass

    def solve(self, z, Z_tv, last_u):
        x, y, heading, v = z
        x_tv, y_tv, heading_tv, v_tv = Z_tv[0]

        last_steer, last_accel = last_u

        accel, _ = self.accel_pid.solve(v, last_accel)
        if v < 0:
            steer, _ = self.steer_pid.solve(y_ev-10*heading_ev, last_steer)
        else:
            steer = 0

        u = [steer, accel]

        return u

    def set_accel_ref(self, accel_ref):
        self.accel_ref = accel_ref
        self.accel_pid.set_x_ref(self.accel_ref)

    def set_steer_ref(self, steer_ref):
        self.steer_ref = steer_ref
        self.steer_pid.set_x_ref(self.steer_ref)

if __name__ == '__main__':
    import numpy as np

    safety_params = safetyParams()
    safety_control = safetyController(safetyParams)

    z_ev = np.random.rand(4)
    Z_tv = np.random.rand(20,4)
    last_u = np.random.rand(2)

    u = safety_control.solve(z_ev, Z_tv, last_u)
    print(u)
