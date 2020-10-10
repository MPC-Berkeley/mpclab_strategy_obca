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
class dynamicsKinBikeParams(PythonMsg):
    L_f: float = field(default=1.5)
    L_r: float = field(default=1.5)

    dt: float = field(default=0.1)

    M: int = field(default=10)