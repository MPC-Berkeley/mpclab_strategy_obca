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
class experimentParams(PythonMsg):
    car_L: float = field(default=4.0)
    car_W: float = field(default=1.5)

    N: int = field(default=20)
    dt: float = field(default=0.1)

    collision_buffer_r: float = field(default=1.0)
    confidence_thresh: float = field(default=0.55)

    x_max: float = field(default=30.)
    T: int = field(default=1500)

    T_tv: int = field(default=200)

    lock_steps: int = field(default=20)

    strategy_names: tuple = field(default=("Left", "Right", "Yield"))

    a_lim: np.array = field(default=np.array([-8.0, 2.5]))

@dataclass
class experimentStates(PythonMsg):
    t: int = field(default=0)
    EV_curr: np.array = field(default=None)
    TV_pred: np.array = field(default=None)
    score: np.array = field(default=np.zeros(3))
    feas: bool = field(default=True)
    actual_collision: bool = field(default=False)

    ref_col: list = field(default=None)
