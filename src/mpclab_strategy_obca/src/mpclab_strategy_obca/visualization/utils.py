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

from barc.utils.envs import BlankMap

import rospy
from mpclab_strategy_obca.msg import FSMState, Scores
from barc.msg import States, ECU, Prediction
from nav_msgs.msg import Odometry

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter

class data_listener(object):
    """Object collecting closed loop data points
    Attributes:
        updateInitialConditions: function which updates initial conditions and clear the memory
    """
    def __init__(self, namespace, plotter_params, visualizer_params):
        """Initialization
        Arguments:

        """

        plot_sim      = visualizer_params.plot_sim
        plot_est      = visualizer_params.plot_est
        plot_sensor   = visualizer_params.plot_sensor
        plot_gps      = visualizer_params.plot_gps

        plot_state    = plotter_params.plot_state
        plot_score    = plotter_params.plot_score
        plot_pred     = plotter_params.plot_pred
        plot_ss       = plotter_params.plot_ss

        if plot_sim:
            rospy.Subscriber(namespace + '/sim_states', States, self.simState_callback)
        if plot_sensor:
            rospy.Subscriber(namespace + '/mea_states', States, self.meaState_callback)
        if plot_est:
            rospy.Subscriber(namespace + '/est_states', States, self.estState_callback)
        if plot_pred:
            rospy.Subscriber(namespace + '/pred_states', Prediction, self.predState_callback)
        if plot_ss:
            rospy.Subscriber(namespace + '/sel_ss', Prediction, self.SS_callback)
        if plot_gps:
            rospy.Subscriber(namespace + '/gps_pos', States, self.gps_callback)
        if plot_state:
            rospy.Subscriber(namespace + '/fsm_state', FSMState, self.fsm_state_callback)
        if plot_score:
            rospy.Subscriber(namespace + '/strategy_scores', Scores, self.scores_callback)

        rospy.Subscriber(namespace + '/ecu', ECU, self.ecu_callback)

        self.sim_states     = States()
        self.mea_states     = States()
        self.est_states     = States()
        self.pred_states    = Prediction()
        self.ss_states      = Prediction()
        self.gps            = States()
        self.input          = ECU()
        self.fsm_state      = 'Start'
        self.scores         = [0, 0, 0]

    def simState_callback(self, msg):
        self.sim_states = msg

    def meaState_callback(self, msg):
        self.mea_states = msg

    def estState_callback(self, msg):
        self.est_states = msg

    def predState_callback(self, msg):
        self.pred_states = msg

    def SS_callback(self, msg):
        self.ss_states = msg

    def ecu_callback(self, msg):
        self.input = msg

    def gps_callback(self, msg):
        self.gps = msg

    def fsm_state_callback(self, msg):
        self.fsm_state = msg.fsm_state

    def scores_callback(self, msg):
        self.scores = msg.scores

class PlotBARC(object):
    """docstring for PlotBARC"""
    def __init__(self):
        super(PlotBARC, self).__init__()

        self.plot_subplots = rospy.get_param('/visualization/plot_subplots', False)
        self.plot_sim     = rospy.get_param("/visualization/plot_sim")
        self.plot_sensor  = rospy.get_param("/visualization/plot_sensor")
        self.plot_est     = rospy.get_param("/visualization/plot_est")
        self.plot_pred = rospy.get_param('controller/plot_pred', False)
        self.plot_ss = rospy.get_param('controller/plot_ss', False)
        self.plot_gps = rospy.get_param('visualization/plot_gps', True)

        # Vehicle Dimensions
        self.barc_L = rospy.get_param("car/plot/L")
        self.barc_W = rospy.get_param("car/plot/W")
        self.n_revs = 0

        # Figure and Axis
        self.fig     = None
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

        # Variable to receive simulated, measured and estimated data
        self.data = SimMeasureEstInputData(self.plot_sim, self.plot_sensor,
            self.plot_est, self.plot_gps, self.plot_pred, self.plot_ss)

        self.map = self.load_map()

        self.init_fig()

    def load_map(self):
        return BlankMap()

    def init_fig(self):
        figsize = (14, 7) if self.plot_subplots else (7, 7)
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle("BARC Plotter", fontsize=16)
        plt.ion()

        ################ Trajectory Subplot ################
        if self.plot_subplots:
            self.axtr = self.fig.add_subplot(1, 2, 1)
            self.axtr.set_title("Trajectories")
            self.axtr.set_xlabel("X")
            self.axtr.set_ylabel("Y")
        else:
            self.axtr = self.fig.add_subplot(1, 1, 1)
        # User Defined map plotting
        self.map.plot_map(self.axtr)

        # Vehicle Init
        v = np.array([[ 1.,  1.],
                      [ 1., -1.],
                      [-1., -1.],
                      [-1.,  1.]])

        xdata, ydata = [], []

        if self.plot_sim:
            self.rec_sim = patches.Polygon(v, alpha=0.7,closed=True, fc='r', ec='k',zorder=10)
            self.axtr.add_patch(self.rec_sim)

            self.line_sim, = self.axtr.plot(xdata, ydata, 'r-')

        if self.plot_sensor:
            self.rec_mea = patches.Polygon(v, alpha=0.7,closed=True, fc='g', ec='k',zorder=10)
            self.axtr.add_patch(self.rec_mea)

            self.line_mea, = self.axtr.plot(xdata, ydata, 'g-')

        if self.plot_est:
            self.rec_est = patches.Polygon(v, alpha=0.7,closed=True, fc='b', ec='k',zorder=10)
            self.axtr.add_patch(self.rec_est)

            self.line_est, = self.axtr.plot(xdata, ydata, 'b-')

        if self.plot_pred:
            self.line_pred, = self.axtr.plot(xdata, ydata, 'ok-')

        if self.plot_ss:
            self.line_ss, = self.axtr.plot(xdata, ydata, 'mo')

        if self.plot_gps:
            self.line_gps, = self.axtr.plot(xdata, ydata, 'gx', linewidth=2, markersize=10)

        if self.plot_subplots:
            ################ Speed Subplot ################
            self.axv = self.fig.add_subplot(4, 2, 2)
            self.axv.set_title("Velocities")
            self.axv.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            if self.plot_sim:
                self.linev_sim, = self.axv.plot(ydata, 'or-', label='sim')

            if self.plot_sensor:
                self.linev_mea, = self.axv.plot(ydata, 'og-', label='mea')

            if self.plot_est:
                self.linev_est,  = self.axv.plot(ydata, 'ob-', label='est')

            if self.plot_pred:
                self.linev_pred, = self.axv.plot(ydata, 'ok-', label='pred')

            self.axv.legend(loc='upper left')

            ################ PsiDot Subplot ################
            self.axpsiDot = self.fig.add_subplot(4, 2, 4)
            self.axpsiDot.set_title("psiDot")
            self.axpsiDot.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            if self.plot_sim:
                self.linepsiDot_sim, = self.axpsiDot.plot(ydata, 'or-', label='sim')

            if self.plot_sensor:
                self.linepsiDot_mea, = self.axpsiDot.plot(ydata, 'og-', label='mea')

            if self.plot_est:
                self.linepsiDot_est,  = self.axpsiDot.plot(ydata, 'ob-', label='est')

            if self.plot_pred:
                self.linepsiDot_pred, = self.axpsiDot.plot(ydata, 'ok-', label='pred')


            self.axpsiDot.legend(loc='upper left')

            ################ u_a Subplot ################
            self.axua = self.fig.add_subplot(4, 2, 6)
            self.axua.set_title("motor")
            self.axua.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            self.lineua,  = self.axua.plot(ydata, 'or-')

            if self.plot_pred:
                self.lineua_pred, = self.axua.plot(ydata, 'ok-')

            ################ u_df Subplot ################
            self.axudf = self.fig.add_subplot(4, 2, 8)
            self.axudf.set_title("servo")
            self.axudf.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            self.lineudf,  = self.axudf.plot(ydata, 'or-')

            if self.plot_pred:
                self.lineudf_pred, = self.axudf.plot(ydata, 'ok-')

        self.axtr.set_aspect('equal')
        plt.tight_layout()
        plt.show()


    def plot_car(self):

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

        self.fig.canvas.draw()

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

        x_preds = []
        y_preds = []
        psi_preds = []
        for i in range(len(s_preds)):
            loc_coord = (s_preds[i], ey_preds[i], epsi_preds[i])
            glob_coord = self.map.local_to_global(loc_coord)
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

class PlotError(object):
    def __init__(self):

        # Figure and Axis
        self.fig     = None
        self.vis_len = 50

        # State and input Histories
        self.e_psi_his      = []
        self.e_y_his     = []
        self.s_his     = []

        self.line_e_psi  = None
        self.line_e_y    = None
        self.line_s      = None

        # Variable to receive simulated, measured and estimated data
        self.data = LocStateData()

        self.init_fig()

    def init_fig(self):
        self.fig = plt.figure(figsize=(7, 7))
        # self.fig.suptitle("BARC Simulator Errors", fontsize=16)
        plt.ion()

        xdata, ydata = [], []

        ################ e_psi Subplot ################
        self.ax_epsi = self.fig.add_subplot(3, 1, 1)
        self.ax_epsi.set_title("e_psi")
        self.ax_epsi.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        self.line_e_psi, = self.ax_epsi.plot(ydata, 'or-')


        ################ e_y Subplot ################
        self.ax_ey = self.fig.add_subplot(3, 1, 2)
        self.ax_ey.set_title("e_y")
        self.ax_ey.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        self.line_e_y, = self.ax_ey.plot(ydata, 'or-')

        ################ s Subplot ################
        self.ax_s = self.fig.add_subplot(3, 1, 3)
        self.ax_s.set_title("s")
        self.ax_s.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        self.line_s,  = self.ax_s.plot(ydata, 'or-')

        plt.tight_layout()
        plt.show()

    def plot(self):

        ################ e_psi Subplot ###############
        self.e_psi_his.append(self.data.locStates.epsi)
        self.set_limited_data(self.line_e_psi, self.e_psi_his)

        ################ e_y Subplot ###############
        self.e_y_his.append(self.data.locStates.ey)
        self.set_limited_data(self.line_e_y, self.e_y_his)

        ################ s Subplot ##############
        self.s_his.append(self.data.locStates.s)
        self.set_limited_data(self.line_s, self.s_his)

        ############### Update the figure ###############
        # recompute the ax.dataLim
        self.ax_s.relim()
        self.ax_epsi.relim()
        self.ax_ey.relim()

        # update ax.viewLim using the new dataLim
        self.ax_s.autoscale_view()
        self.ax_epsi.autoscale_view()
        self.ax_ey.autoscale_view()

        self.fig.canvas.draw()

    def set_limited_data(self, handle, data):
        if len(data) > self.vis_len:
            handle.set_data(range(self.vis_len), data[-self.vis_len:])
        else:
            handle.set_data(range(len(data)), data)


class LocStateData(object):

    def __init__(self):
        """Initialization
        Arguments:
        """
        rospy.Subscriber("/log_states", States, self.locState_callback)

        self.locStates = States()

    def locState_callback(self, msg):
        epsi = msg.epsi
        ey = msg.ey
        s = msg.s

        self.locStates.epsi = epsi
        self.locStates.ey = ey
        self.locStates.s = s
