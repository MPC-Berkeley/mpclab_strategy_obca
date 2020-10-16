#!/usr/bin python3

from dataclasses import dataclass, field
import numpy as np

@dataclass
class PythonMsg:
    def __setattr__(self,key,value):
        if not hasattr(self,key):
            raise TypeError ('Cannot add new field "%s" to frozen class %s' %(key,self))
        else:
            object.__setattr__(self,key,value)

@dataclass
class PIDParams(PythonMsg):
    dt: float = field(default=0.1)

    Kp: float = field(default=2.0)
    Ki: float = field(default=0.0)
    Kd: float = field(default=0.0)

    int_e_max: float = field(default=100)
    int_e_min: float = field(default=-100)
    u_max: float = field(default=None)
    u_min: float = field(default=None)
    du_max: float = field(default=None)
    du_min: float = field(default=None)

@dataclass
class safetyParams(PythonMsg):
    dt: float = field(default=0.1)

    P_accel: float = field(default=1.0)
    I_accel: float = field(default=0.0)
    D_accel: float = field(default=0.0)

    P_steer: float = field(default=1.0)
    I_steer: float = field(default=0.0)
    D_steer: float = field(default=0.0)

    accel_int_e_max: float = field(default=100)
    accel_int_e_min: float = field(default=-100)
    accel_max: float = field(default=None)
    accel_min: float = field(default=None)
    daccel_max: float = field(default=None)
    daccel_min: float = field(default=None)

    steer_int_e_max: float = field(default=100)
    steer_int_e_min: float = field(default=-100)
    steer_max: float = field(default=None)
    steer_min: float = field(default=None)
    dsteer_max: float = field(default=None)
    dsteer_min: float = field(default=None)

@dataclass
class strategyOBCAParams(PythonMsg):
    dt: float = field(default=0.1)
    n: int = field(default=4)
    d: int = field(default=2)

    N: int = field(default=20)

    n_obs: int = field(default=3)
    n_ineq: np.array = field(default=np.array([4, 1, 1]))
    d_ineq: int = field(default=2)

    G: np.array = field(default=np.array([[1,0], [-1, 0], [0, 1], [0, -1]]))
    g: np.array = field(default=np.array([2.45, 2.45, 1.03, 1.03]))

    Q: np.array = field(default=np.array([10, 10, 10, 10]))
    R: np.array = field(default=np.array([1, 1]))
    R_d: np.array = field(default=np.array([1, 10]))

    d_min: float = field(default=0.01)

    z_l: np.array = field(default=np.array([-10, -10, -10, -1]))
    z_u: np.array = field(default=np.array([10, 10, 10, 1]))
    u_l: np.array = field(default=np.array([-0.35, -2.5]))
    u_u: np.array = field(default=np.array([0.35, 1.5]))
    du_l: np.array = field(default=np.array([-0.6, -8]))
    du_u: np.array = field(default=np.array([0.6, 5]))

    optlevel: int = field(default=3)

    ws_name: str = field(default="ws_solver_strat")
    opt_name: str = field(default="opt_sovler_strat")

@dataclass
class trackingParams(PythonMsg):
    dt: float = field(default=0.1)
    n: int = field(default=4)
    d: int = field(default=2)

    N: int = field(default=50)

    Q: np.array = field(default=np.array([10, 10, 10, 10]))
    R: np.array = field(default=np.array([1, 1]))
    R_d: np.array = field(default=np.array([1, 10]))

    z_l: np.array = field(default=np.array([-10, -10, -10, -1]))
    z_u: np.array = field(default=np.array([10, 10, 10, 1]))
    u_l: np.array = field(default=np.array([-0.35, -2.5]))
    u_u: np.array = field(default=np.array([0.35, 1.5]))
    du_l: np.array = field(default=np.array([-0.6, -8]))
    du_u: np.array = field(default=np.array([0.6, 5]))

    optlevel: int = field(default=3)
    opt_name: str = field(default="tracking_solver")
