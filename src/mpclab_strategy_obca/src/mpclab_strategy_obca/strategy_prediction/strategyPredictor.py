#!/usr/bin python3

import numpy as np
import numpy.linalg as la

from mpclab_strategy_obca.strategy_prediction.abstractStrategyPredictor import abstractStrategyPredictor
from mpclab_strategy_obca.strategy_prediction.NNModel import Net, load_matlab_network
from mpclab_strategy_obca.strategy_prediction.utils.types import strategyPredictorParams

STRATEGIES = ['left', 'right', 'yield']

class strategyPredictor(abstractStrategyPredictor):
    def __init__(self, params=strategyPredictorParams()):
        nn_model_file = params.nn_model_file
        self.smooth_pred = params.smooth_prediction
        self.V_kf = params.V_kf
        self.W_kf = params.W_kf
        self.Pm_kf = params.Pm_kf
        self.A_kf = params.A_kf
        self.H_kf = params.H_kf

        self.last_pred = None

        self.nn_params = load_matlab_network(nn_model_file)
        self.net = Net(self.nn_params)

    def initialize(self):
        pass

    def predict(self, x):
        y = self.net.forward(x).detach().numpy()
        if self.smooth_pred and self.last_pred is not None:
            y = self.smooth_prediction(y, self.last_pred)
        self.last_pred = y
        return y

    def smooth_prediction(self, z, x):
        xp = self.A_kf.dot(x)
        Pp = self.Pm_kf + self.V_kf

        K = Pp.dot(self.H_kf.T)*la.inv(self.H_kf.dot(Pp.dot(self.H_kf.T))+self.W_kf)
        xm = xp + K.dot(z-self.H_kf.dot(xp))
        self.Pm_kf = Pp - K.dot(self.H_kf.dot(Pp))

        return xm/np.sum(xm)

if __name__ == '__main__':
    import torch

    strat_params = strategyPredictorParams(nn_model_file='nn_strategy_TF-trainscg_h-40_AC-tansig_ep2000_CE0.17453_2020-08-04_15-42.mat')
    strat_pred = strategyPredictor(strat_params)

    x = np.random.randn(strat_pred.nn_params.d_in)
    pred = strat_pred.predict(x)

    print(pred)
