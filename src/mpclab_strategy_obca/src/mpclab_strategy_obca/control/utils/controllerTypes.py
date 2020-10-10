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
class strategyOBCAParams(PythonMsg):
    dt: float = field(default=0.1)
    n: int = field(default=4)
    d: int = field(default=2)

    N: int = field(default=20)

    n_obs: int = field(default=3)
    n_ineq: np.array = field(default=[4, 1, 1])
    d_ineq: int = field(default=2)

    G: np.array = field(default=None)
    g: np.array = field(default=None)

    Q: np.array = field(default=None)
    R: np.array = field(default=None)

    d_min: float = field(default=0.01)

    u_l: np.array = field(default=None)
    u_u: np.array = field(default=None)
    du_l: np.array = field(default=None)
    du_u: np.array = field(default=None)

    optlevel: int = field(default=3)

    ws_name: str = field(default="ws_solver_strat")
    opt_name: str = field(default="opt_sovler_strat")
