#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np

from mpclab_strategy_obca.visualization.plotter import barcOBCAPlotter
from mpclab_strategy_obca.visualization.types import visualizerParams

from mpclab_strategy_obca.state_machine.stateMachine import state_num_dict, num_state_dict

from mpclab_strategy_obca.utils.utils import load_vehicle_trajectory

matplotlib.use('TKAgg')

class barcOBCAVisualizer(object):
    def __init__(self, track, params=visualizerParams()):
        self.track = track

        self.visualizer_params = params
        plot_subplots = params.plot_subplots

        self.fsm_state_ids = list(state_num_dict.values())
        self.fsm_state_names = list(state_num_dict.keys())
        sort_idxs = np.argsort(self.fsm_state_ids)

        self.fsm_state_ids = [self.fsm_state_ids[i] for i in sort_idxs]
        self.fsm_state_names = [self.fsm_state_names[i] for i in sort_idxs]

        # Initialize figure
        figsize = (14, 7) if plot_subplots else (7, 7)
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle("BARC OBCA Plotter", fontsize=16)
        plt.ion()

        self.axs = dict()

        if params.trajectory_file is not None:
            scaling_factor = params.scaling_factor
            trajectory = np.multiply(load_vehicle_trajectory(params.trajectory_file), np.array([scaling_factor, scaling_factor, 1, scaling_factor]))

        ################ Trajectory Subplot ################
        if plot_subplots:
            axtr = self.fig.add_subplot(3, 2, 1)
            axtr.set_title("Trajectories")
            axtr.set_xlabel("X")
            axtr.set_ylabel("Y")
        else:
            axtr = self.fig.add_subplot(3, 1, 1)

        if params.trajectory_file is not None:
            axtr.plot(trajectory[:,0], trajectory[:,1])

        # User Defined map plotting
        self.track.plot_map(axtr)
        axtr.set_aspect('equal')
        self.axs['track'] = axtr

        if plot_subplots:
            ################ Speed Subplot ################
            axv = self.fig.add_subplot(4, 2, 2)
            axv.set_ylabel("vel")
            axv.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            self.axs['vel'] = axv

            ################ PsiDot Subplot ################
            axpsiDot = self.fig.add_subplot(4, 2, 4)
            axpsiDot.set_ylabel("yaw rate")
            axpsiDot.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            self.axs['yaw_rate'] = axpsiDot

            ################ u_a Subplot ################
            axua = self.fig.add_subplot(4, 2, 6)
            axua.set_ylabel("motor")
            axua.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            self.axs['throttle'] = axua

            ################ u_df Subplot ################
            axudf = self.fig.add_subplot(4, 2, 8)
            axudf.set_ylabel("servo")
            axudf.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            self.axs['steering'] = axudf

        plt.tight_layout()
        plt.show()

        self.plotters = []

    def attach_plotter(self, namespace, color, dimensions, plotter_params):
        if 'fsm_state' not in self.axs and plotter_params.plot_state:
            axstate = self.fig.add_subplot(3, 2, 3)
            axstate.set_ylabel("FSM state")
            axstate.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axstate.set_yticks(self.fsm_state_ids)
            axstate.set_yticklabels(self.fsm_state_names)
            self.axs['fsm_state'] = axstate

        if 'score' not in self.axs and plotter_params.plot_score:
            axscore = self.fig.add_subplot(3, 2, 5)
            axscore.set_ylabel("Confidence")
            axscore.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            self.axs['score'] = axscore

        self.plotters.append(barcOBCAPlotter(namespace, color, dimensions, plotter_params, self.axs, self.track, self.visualizer_params))

    def update(self):
        if len(self.plotters) == 0:
            raise ValueError('No plotters attached to visualizer')

        for p in self.plotters:
            p.plot_barc()

        self.fig.canvas.draw()
