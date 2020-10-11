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
    collision_buffer_r: float = field(default=1.0)

    dt: float = field(default=0.1)
