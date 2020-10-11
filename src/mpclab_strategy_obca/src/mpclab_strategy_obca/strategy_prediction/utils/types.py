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
class NNParams(PythonMsg):
    d_in: int = field(default=105)
    d_layers: np.array = field(default=np.array([40,3]))
    n_layers: int = field(default=2)

    initial_weights: list = field(default=None)
    initial_biases: list = field(default=None)

    no_grad: bool = field(default=False)

@dataclass
class strategyPredictorParams(PythonMsg):
    nn_model_file: str = field(default='')

    smooth_prediction: bool = field(default=True)
    W_kf: np.array = field(default=0.01*np.eye(3))
    V_kf: np.array = field(default=0.5*np.eye(3))
    Pm_kf: np.array = field(default=0.2*np.eye(3))
    A_kf: np.array = field(default=np.eye(3))
    H_kf: np.array = field(default=np.eye(3))
