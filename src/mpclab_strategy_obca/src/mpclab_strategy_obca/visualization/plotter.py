#!/usr/bin/env python3

"""
    File name: plotCarTrajectory.py
    Author: Xu Shen
    Email: xu_shen@berkeley.edu
    Python Version:
"""
# ---------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that (1) you retain this notice
# and (2) you provide clear attribution to UC Berkeley, including a link
# to http://barc-project.com
#
# Attibution Information: The barc project ROS code-base was developed
# at UC Berkeley in the Model Predictive Control (MPC) lab by Jon Gonzales
# (jon.gonzales@berkeley.edu). The cloud services integation with ROS was developed
# by Kiet Lam  (kiet.lam@berkeley.edu). The web-server app Dator was
# based on an open source project by Bruce Wootton
#
# This code provides a way to see the car's trajectory, orientation, and velocity profile in
# real time with referenced to the track defined a priori.
#
# This version is the modified based on the original barc project
# as the second generation upgrade
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpclab_strategy_obca.visualization.utils import data_listener
from mpclab_strategy_obca.state_machine.stateMachine import state_num_dict, num_state_dict

class barcOBCAPlotter(object):
    """docstring for barc_plotter"""
    def __init__(self, namespace, color, dimensions, plotter_params,
        axs, track, visualizer_params):
        # Get local plotter params
        self.plot_subplots  = visualizer_params.plot_subplots
        self.plot_sim       = visualizer_params.plot_sim
        self.plot_est       = visualizer_params.plot_est
        self.plot_sensor    = visualizer_params.plot_sensor
        self.plot_gps       = visualizer_params.plot_gps

        self.plot_ss        = plotter_params.plot_ss
        self.plot_state     = plotter_params.plot_state
        self.plot_score     = plotter_params.plot_score
        self.plot_pred      = plotter_params.plot_pred
        self.global_pred    = plotter_params.global_pred

        self.fsm_state_ids = list(state_num_dict.values())
        self.fsm_state_names = list(state_num_dict.keys())
        sort_idxs = np.argsort(self.fsm_state_ids)
        self.fsm_state_ids = [self.fsm_state_ids[i] for i in sort_idxs]
        self.fsm_state_names = [self.fsm_state_names[i] for i in sort_idxs]

        # Vehicle Dimensions
        self.barc_L = dimensions['car_length']
        self.barc_W = dimensions['car_width']
        self.n_revs = 0

        self.track = track
        self.color = color

        # Unpack axes
        self.axtr = axs['track']
        if self.plot_subplots:
            self.axv = axs['vel']
            self.axpsiDot = axs['yaw_rate']
            self.axua = axs['throttle']
            self.axudf = axs['steering']
        if self.plot_state:
            self.axstate = axs['fsm_state']
        if self.plot_score:
            self.axscore = axs['score']

        self.vis_len = 50

        # Vehicle Rectangle
        self.rec_sim = None
        self.rec_mea = None
        self.rec_est = None

        # Lines of Vehicle Trajectories
        self.traj_sim_x = []
        self.traj_sim_y = []
        self.traj_mea_x = []
        self.traj_mea_y = []
        self.traj_est_x = []
        self.traj_est_y = []

        # Lines of Vehicle Trajectories
        self.line_sim = None
        self.line_mea = None
        self.line_est = None
        self.line_pred = None
        self.line_ss = None
        self.line_gps = None

        # State and input Histories
        self.v_sim_his      = []
        self.v_mea_his      = []
        self.v_est_his      = []
        self.psiDot_sim_his = []
        self.psiDot_mea_his = []
        self.psiDot_est_his = []
        self.ua_his         = []
        self.udf_his        = []
        self.t_his          = []
        self.state_his      = []
        self.score_his       = {'left' : [], 'right' : [], 'yield' : []}

        self.linev_sim      = None
        self.linev_mea      = None
        self.linev_est      = None
        self.linev_pred     = None
        self.linepsiDot_sim = None
        self.linepsiDot_mea = None
        self.linepsiDot_est = None
        self.linepsiDot_pred= None
        self.lineua         = None
        self.lineua_pred    = None
        self.lineudf        = None
        self.lineudf_pred   = None

        # Set up subscribers to receive data for plotting
        self.data = data_listener(namespace, plotter_params, visualizer_params)

        #  Initialize polygon and lines
        v = np.array([[ 1.,  1.],
                      [ 1., -1.],
                      [-1., -1.],
                      [-1.,  1.]])

        xdata, ydata = [], []

        if self.plot_sim:
            self.rec_sim = patches.Polygon(v, alpha=0.7,closed=True, fc=self.color, ec='k', linestyle='solid', zorder=10)
            self.axtr.add_patch(self.rec_sim)
            self.line_sim, = self.axtr.plot(xdata, ydata, color=self.color, linestyle='solid')

        if self.plot_sensor:
            self.rec_mea = patches.Polygon(v, alpha=0.7,closed=True, fc=self.color, ec='k', linestyle='dotted', zorder=10)
            self.axtr.add_patch(self.rec_mea)
            self.line_mea, = self.axtr.plot(xdata, ydata, color=self.color, linestyle='dotted')

        if self.plot_est:
            self.rec_est = patches.Polygon(v, alpha=0.7,closed=True, fc=self.color, ec='k', linestyle='dashed', zorder=10)
            self.axtr.add_patch(self.rec_est)
            self.line_est, = self.axtr.plot(xdata, ydata, color=self.color, linestyle='dashed')

        if self.plot_pred:
            self.line_pred, = self.axtr.plot(xdata, ydata, 'ok-')
        if self.plot_ss:
            self.line_ss, = self.axtr.plot(xdata, ydata, color=self.color, marker='+')
        if self.plot_gps:
            self.line_gps, = self.axtr.plot(xdata, ydata, color=self.color, marker='x', linewidth=2, markersize=10)

        if self.plot_state:
            self.line_state, = self.axstate.plot(ydata, color=self.color)

        if self.plot_score:
            self.line_scores = dict()
            self.line_scores['left'], = self.axscore.plot(ydata, color='r', label='left')
            self.line_scores['right'], = self.axscore.plot(ydata, color='g', label='right')
            self.line_scores['yield'], = self.axscore.plot(ydata, color='b', label='yield')
            self.axscore.legend(loc='upper right')

        if self.plot_subplots:
            ################ v Subplot ################
            if self.plot_sim:
                self.linev_sim, = self.axv.plot(ydata, color=self.color, marker='o', label='sim')
            if self.plot_sensor:
                self.linev_mea, = self.axv.plot(ydata, color=self.color, marker='s', label='mea')
            if self.plot_est:
                self.linev_est,  = self.axv.plot(ydata, color=self.color, marker='*', label='est')
            if self.plot_pred:
                self.linev_pred, = self.axv.plot(ydata, 'ok-', label='pred')

            self.axv.legend(loc='upper left')

            ################ PsiDot Subplot ################
            if self.plot_sim:
                self.linepsiDot_sim, = self.axpsiDot.plot(ydata, color=self.color, marker='o', label='sim')
            if self.plot_sensor:
                self.linepsiDot_mea, = self.axpsiDot.plot(ydata, color=self.color, marker='s', label='mea')
            if self.plot_est:
                self.linepsiDot_est,  = self.axpsiDot.plot(ydata, color=self.color, marker='*', label='est')
            if self.plot_pred:
                self.linepsiDot_pred, = self.axpsiDot.plot(ydata, 'ok-', label='pred')

            # self.axpsiDot.legend(loc='upper left')

            ################ u_a Subplot ################
            self.lineua,  = self.axua.plot(ydata, color=self.color, marker='o')
            if self.plot_pred:
                self.lineua_pred, = self.axua.plot(ydata, 'ok-')

            ################ u_df Subplot ################
            self.lineudf,  = self.axudf.plot(ydata, color=self.color, marker='o')
            if self.plot_pred:
                self.lineudf_pred, = self.axudf.plot(ydata, 'ok-')

        self.axtr.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def plot_barc(self):
        if self.plot_sim:

            ################ Simulated Traj Subplot ###############
            car_x, car_y = self.getCarPosition(self.data.sim_states)
            self.rec_sim.set_xy(np.array([car_x, car_y]).T)

            self.traj_sim_x.append(self.data.sim_states.x)
            self.traj_sim_y.append(self.data.sim_states.y)
            self.line_sim.set_data(self.traj_sim_x, self.traj_sim_y)

            ################ Simulated State Subplots ###############
            if self.plot_subplots:
                # Velocity
                self.v_sim_his.append(self.data.sim_states.v)
                self.set_limited_data(self.linev_sim, self.t_his, self.v_sim_his)

                # psiDot
                self.psiDot_sim_his.append(self.data.sim_states.psiDot)
                self.set_limited_data(self.linepsiDot_sim, self.t_his, self.psiDot_sim_his)

        if self.plot_sensor:

            ################ Measured Traj Subplot ###############
            car_x, car_y = self.getCarPosition(self.data.mea_states)
            self.rec_mea.set_xy(np.array([car_x, car_y]).T)

            self.traj_mea_x.append(self.data.mea_states.x)
            self.traj_mea_y.append(self.data.mea_states.y)
            self.line_mea.set_data(self.traj_mea_x, self.traj_mea_y)

            if self.plot_subplots:
                ################ Measured State Subplots ###############
                self.v_mea_his.append(self.data.mea_states.v)
                self.set_limited_data(self.linev_mea, self.t_his, self.v_mea_his)

                self.psiDot_mea_his.append(self.data.mea_states.psiDot)
                self.set_limited_data(self.linepsiDot_mea, self.t_his, self.psiDot_mea_his)

        if self.plot_est:
            ################ Estimated Traj Subplot ###############
            car_x, car_y = self.getCarPosition(self.data.est_states)
            self.rec_est.set_xy(np.array([car_x, car_y]).T)

            self.traj_est_x.append(self.data.est_states.x)
            self.traj_est_y.append(self.data.est_states.y)
            self.line_est.set_data(self.traj_est_x, self.traj_est_y)

            if self.plot_subplots:
                ################ Estimated State Subplots ###############
                self.v_est_his.append(self.data.est_states.v)
                self.set_limited_data(self.linev_est, self.t_his, self.v_est_his)

                self.psiDot_est_his.append(self.data.est_states.psiDot)
                self.set_limited_data(self.linepsiDot_est, self.t_his, self.psiDot_est_his)

        if self.plot_pred:
            x_preds, y_preds, psi_preds, v_preds, psiDot_preds, df_preds, a_preds = self.getPreds(self.data.pred_states)

            self.line_pred.set_data(x_preds, y_preds)

            if self.plot_subplots:
                # predicted state subplots
                n_v_est = len(self.v_est_his)
                n_v_pred = len(v_preds)
                self.linev_pred.set_data(range(n_v_est, n_v_est+n_v_pred), v_preds)

                n_psiDot_est = len(self.psiDot_est_his)
                n_psiDot_pred = len(psiDot_preds)
                self.linepsiDot_pred.set_data(range(n_psiDot_est, n_psiDot_est+n_psiDot_pred), psiDot_preds)

        if self.plot_ss:
            x_sel = self.data.ss_states.s
            y_sel = self.data.ss_states.ey
            self.line_ss.set_data(x_sel, y_sel)

        if self.plot_gps:
            x = self.data.gps.x
            y = self.data.gps.y
            self.line_gps.set_data(x, y)

        if self.plot_subplots:
            ################ Input Subplots ##############
            self.ua_his.append(self.data.input.motor)
            self.set_limited_data(self.lineua, self.t_his, self.ua_his)

            self.udf_his.append(self.data.input.servo)
            self.set_limited_data(self.lineudf, self.t_his, self.udf_his)

            if self.plot_pred:
                n_df_est = len(self.udf_his)
                n_df_pred = len(df_preds)
                self.lineudf_pred.set_data(range(n_df_est, n_df_est+n_df_pred), df_preds)

                n_a_est = len(self.ua_his)
                n_a_pred = len(a_preds)
                self.lineua_pred.set_data(range(n_a_est, n_a_est+n_a_pred), a_preds)

            ############### Update the figure ###############
            # recompute the ax.dataLim
            self.axv.relim()
            self.axpsiDot.relim()
            self.axua.relim()
            self.axudf.relim()

            # update ax.viewLim using the new dataLim
            self.axv.autoscale_view()
            self.axpsiDot.autoscale_view()
            self.axua.autoscale_view()
            self.axudf.autoscale_view()

        if self.plot_state:
            if not self.data.fsm_state:
                self.state_his.append(-1)
            else:
                self.state_his.append(state_num_dict[self.data.fsm_state])
            self.line_state.set_data(range(len(self.state_his)), self.state_his)
            self.axstate.set_ylim(self.fsm_state_ids[0], self.fsm_state_ids[-1])
            self.axstate.set_xlim(0, len(self.state_his))

        if self.plot_score:
            if not self.data.scores:
                self.score_his['left'].append(0)
                self.score_his['right'].append(0)
                self.score_his['yield'].append(0)
            else:
                self.score_his['left'].append(self.data.scores[0])
                self.score_his['right'].append(self.data.scores[1])
                self.score_his['yield'].append(self.data.scores[2])
            self.line_scores['left'].set_data(range(len(self.score_his['left'])), self.score_his['left'])
            self.line_scores['right'].set_data(range(len(self.score_his['right'])), self.score_his['right'])
            self.line_scores['yield'].set_data(range(len(self.score_his['yield'])), self.score_his['yield'])
            self.axscore.set_ylim(0, 1)
            self.axscore.set_xlim(0, len(self.score_his['left']))

    def getCarPosition(self, states):
        x     = states.x
        y     = states.y
        psi   = states.psi
        delta = states.u_df
        l     = self.barc_L / 2
        w     = self.barc_W / 2

        car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l*np.cos(psi) + w * np.sin(psi),
                  x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
        car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
                  y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]

        return car_x, car_y

    def getPreds(self, preds):
        s_preds = preds.s
        ey_preds = preds.ey
        epsi_preds = preds.epsi
        vx_preds = preds.v_x
        vy_preds = preds.v_y
        psiDot_preds = preds.psiDot
        a_preds = preds.a
        df_preds = preds.df

        if self.global_pred:
            x_preds = preds.x
            y_preds = preds.y
            psi_preds = preds.psi
        else:
            x_preds = []
            y_preds = []
            psi_preds = []
            for i in range(len(s_preds)):
                loc_coord = (s_preds[i], ey_preds[i], epsi_preds[i])
                glob_coord = self.track.local_to_global(loc_coord)
                x_preds.append(glob_coord[0])
                y_preds.append(glob_coord[1])
                psi_preds.append(glob_coord[2])

        v_preds = np.sqrt(np.power(vx_preds, 2) + np.power(vy_preds,2))

        return x_preds, y_preds, psi_preds, v_preds, psiDot_preds, df_preds, a_preds

    def set_limited_data(self, handle, t, data):
        if len(t) > 0:
            if len(data) > self.vis_len:
                handle.set_data(t[-self.vis_len], data[-self.vis_len:])
            else:
                handle.set_data(t, data)
        else:
            idxs = range(len(data))
            if len(data) > self.vis_len:
                handle.set_data(idxs[-self.vis_len:], data[-self.vis_len:])
            else:
                handle.set_data(idxs, data)
