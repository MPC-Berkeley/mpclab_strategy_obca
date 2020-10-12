#!/usr/bin python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.io import loadmat
import numpy as np
import warnings

import pdb

from mpclab_strategy_obca.strategy_prediction.utils.types import NNParams

class Net(nn.Module):
    def __init__(self, params=NNParams()):
        super(Net, self).__init__()
        d_in = params.d_in
        d_layers = params.d_layers
        n_layers = params.n_layers
        initial_weights = params.initial_weights
        initial_biases = params.initial_biases
        no_grad = params.no_grad
        self.std_max = params.standardize_max
        self.std_min = params.standardize_min

        self.layers = [nn.Linear(d_in, d_layers[0])]
        for i in range(n_layers-1):
            self.layers.append(nn.Linear(self.layers[-1].out_features, d_layers[i+1]))

        # pdb.set_trace()
        if no_grad and initial_weights is not None:
            with torch.no_grad():
                for (i, l) in enumerate(self.layers):
                    l.weight.copy_(torch.from_numpy(initial_weights[i]).float())
                    l.bias.copy_(torch.from_numpy(initial_biases[i]).float())
        elif initial_weights is not None:
            for (i, l) in enumerate(self.layers):
                l.weight.copy_(torch.from_numpy(initial_weights[i]).float())
                l.bias.copy_(torch.from_numpy(initial_biases[i]).float())

    def forward(self, x):
        x = self.standardize(x)
        x = torch.tensor(x, dtype=torch.float)
        for i in range(len(self.layers)-1):
            x = torch.tanh(self.layers[i](x))
        x = F.softmax(self.layers[-1](x))
        # x = F.log_softmax(self.layers[-1](x))

        return x

    def standardize(self, x):
        # Standardize to [-1, 1]
        if self.std_max is None or self.std_min is None:
            warnings.warn('Standardization bounds not given, passing input through unmodified')
            return x

        x_std = np.divide(x-self.std_min,self.std_max-self.std_min)*2 - 1
        return x_std

def load_matlab_network(filename):
    vars = loadmat(filename)

    # pdb.set_trace()
    if 'net' not in vars.keys():
        raise RuntimeError("The .mat file must have the variable 'net', a MATLAB 'network' object")

    net = vars['net']
    d_in = net['IW'][0,0][0][0].shape[1]
    d_layers = [net['IW'][0,0][0][0].shape[0]]
    n_layers = net['LW'][0,0].shape[0]

    weights = [net['IW'][0,0][0][0]]
    biases = [np.squeeze(net['b'][0,0][0][0])]

    standardize_bounds = net['inputs'][0][0][0][0][0][0][7] # Messy, but don't know of a better way to parse network object from MATLAB
    standardize_min = standardize_bounds[:,0]
    standardize_max = standardize_bounds[:,1]
    # pdb.set_trace()

    for i in range(n_layers):
        for j in range(1,n_layers):
            if net['LW'][0,0][j,i].size != 0:
                d_layers.append(net['LW'][0,0][j,i].shape[0])
                weights.append(net['LW'][0,0][j,i])
                biases.append(np.squeeze(net['b'][0,0][j][0]))

    d_layers = np.array(d_layers)

    return NNParams(d_in=d_in, d_layers=d_layers, n_layers=n_layers,
        initial_weights=weights, initial_biases=biases, no_grad=True,
        standardize_max=standardize_max, standardize_min=standardize_min)

if __name__ == '__main__':
    nn_params = load_matlab_network('/home/mpcbarc/strategy_classification_models/nn_strategy_TF-trainscg_h-40_AC-tansig_ep2000_CE0.17453_2020-08-04_15-42.mat')
    net = Net(nn_params)
    # pdb.set_trace()
    x = np.random.randn(nn_params.d_in)
    y = net.forward(x)
    print(y)
    pdb.set_trace()
