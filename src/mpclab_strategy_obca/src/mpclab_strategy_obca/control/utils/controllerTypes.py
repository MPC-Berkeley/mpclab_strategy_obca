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

    Q: np.array = field(default=None)
    R: np.array = field(default=None)
