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
class visualizerParams(PythonMsg):
    dt: float = field(default=0.1)

    plot_subplots: bool = field(default=True)
    plot_sim: bool = field(default=True)
    plot_est: bool = field(default=True)
    plot_gps: bool = field(default=False)
    plot_sensor: bool = field(default=False)

    trajectory_file: str = field(default=None)
    scaling_factor: float = field(default=1.0)

@dataclass
class plotterParams(PythonMsg):
    plot_state: bool = field(default=False)
    plot_score: bool = field(default=False)
    plot_ss: bool = field(default=False)
    plot_pred: bool = field(default=False)
    global_pred: bool = field(default=True)
