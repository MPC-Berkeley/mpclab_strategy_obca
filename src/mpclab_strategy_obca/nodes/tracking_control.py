#!/usr/bin python3

import rospy
from bondpy import bondpy
import numpy as np
import numpy.linalg as la

from barc.msg import ECU, States, Prediction

from mpclab_strategy_obca.control.trackingController import trackingController
from mpclab_strategy_obca.control.utils.types import trackingParams

from mpclab_strategy_obca.dynamics.dynamicsModels import bike_dynamics_rk4
from mpclab_strategy_obca.dynamics.utils.types import dynamicsKinBikeParams

from mpclab_strategy_obca.utils.utils import load_vehicle_trajectory

class trackingControlNode(object):
    def __init__(self):
        # Read parameter values from ROS parameter server
        rospy.init_node('tracking_control')

        ns = rospy.get_namespace()
        vehicle_ns = '/'.join(ns.split('/')[:-1])

        self.dt = rospy.get_param('controller/dt')
        self.init_time = rospy.get_param('controller/init_time')
        self.max_time = rospy.get_param('controller/max_time')
        self.x_max = rospy.get_param('controller/x_max')
        self.x_min = rospy.get_param('controller/x_min')
        self.y_max = rospy.get_param('controller/y_max')
        self.y_min = rospy.get_param('controller/y_min')
        self.heading_max = rospy.get_param('controller/heading_max')
        self.heading_min = rospy.get_param('controller/heading_min')
        self.v_max = rospy.get_param('controller/v_max')
        self.v_min = rospy.get_param('controller/v_min')
        self.accel_max = rospy.get_param('controller/accel_max')
        self.accel_min = rospy.get_param('controller/accel_min')
        self.steer_max = rospy.get_param('controller/steer_max')
        self.steer_min = rospy.get_param('controller/steer_min')
        self.daccel_max = rospy.get_param('controller/daccel_max')
        self.daccel_min = rospy.get_param('controller/daccel_min')
        self.dsteer_max = rospy.get_param('controller/dsteer_max')
        self.dsteer_min = rospy.get_param('controller/dsteer_min')
        self.lanewidth = rospy.get_param('controller/lanewidth')
        trajectory_file = rospy.get_param('controller/trajectory_file', None)
        x_scaling = rospy.get_param('controller/x_scaling', 1.0)
        y_scaling = rospy.get_param('controller/y_scaling', 1.0)
        v_scaling = rospy.get_param('controller/v_scaling', 1.0)

        self.n_x = rospy.get_param('controller/tracking/n')
        self.n_u = rospy.get_param('controller/tracking/d')
        self.N = rospy.get_param('controller/tracking/N')
        self.Q = rospy.get_param('controller/tracking/Q')
        self.R = rospy.get_param('controller/tracking/R')
        self.R_d = rospy.get_param('controller/tracking/R_d')
        self.optlevel = rospy.get_param('controller/tracking/optlevel')

        self.L_r = rospy.get_param('controller/dynamics/L_r')
        self.L_f = rospy.get_param('controller/dynamics/L_f')
        self.M = rospy.get_param('controller/dynamics/M')

        self.EV_L = rospy.get_param('car/plot/L')
        self.EV_W = rospy.get_param('car/plot/W')

        x_init = rospy.get_param('car/car_init/x', 0.0)
        y_init = rospy.get_param('car/car_init/y', 0.0)
        heading_init = rospy.get_param('car/car_init/heading', 0.0)
        if type(heading_init) is str:
            heading_init = eval(heading_init)
        v_init = rospy.get_param('car/car_init/v', 0.0)

        self.x_boundary = rospy.get_param('/track/x_boundary')
        self.y_boundary = rospy.get_param('/track/y_boundary')

        if trajectory_file is not None:
            # Load reference trajectory from file and set initial conditions
            traj = load_vehicle_trajectory(trajectory_file)
            traj -= np.array([traj[0,0], traj[0,1], 0, traj[0,3]])
            traj = np.multiply(traj, np.array([x_scaling, y_scaling, 1, v_scaling]))
            traj += np.array([x_init, y_init, 0, v_init])
            self.trajectory = traj

            self.traj_len = self.trajectory.shape[0]
            rospy.set_param('/'.join((vehicle_ns,'car/car_init/x')), float(self.trajectory[0,0]))
            rospy.set_param('/'.join((vehicle_ns,'car/car_init/y')), float(self.trajectory[0,1]))
            rospy.set_param('/'.join((vehicle_ns,'car/car_init/z')), 0.0)
            rospy.set_param('/'.join((vehicle_ns,'car/car_init/heading')), float(self.trajectory[0,2]))
            rospy.set_param('/'.join((vehicle_ns,'car/car_init/v')), 0.0)

            rospy.loginfo('Initial state has been set to - X: %g, Y: %g, heading: %g, v: %g' %
                (float(self.trajectory[0,0]), float(self.trajectory[0,1]), float(self.trajectory[0,2]), 0.0))
        else:
            # Generate simple reference trajectory
            self.traj_len = rospy.get_param('controller/tracking/T')
            v_ref = rospy.get_param('controller/tracking/v_ref')
            y_ref = rospy.get_param('controller/tracking/y_ref')
            x_init = rospy.get_param('car/car_init/x')
            y_init = rospy.get_param('car/car_init/y')
            heading_init = rospy.get_param('car/car_init/heading')
            if type(heading_init) is str:
                heading_init = eval(heading_init)

            # x_ref = x_init + v_ref*np.cos(heading_init)*np.arange(0, self.traj_len*self.dt, self.dt)
            x_ref = np.zeros(self.traj_len)
            y_ref = y_ref*np.ones(self.traj_len)
            heading_ref = heading_init*np.ones(self.traj_len)
            v_ref = v_ref*np.ones(self.traj_len)

            self.trajectory = np.vstack((x_ref, y_ref, heading_ref, v_ref)).T

        dyn_params = dynamicsKinBikeParams(dt=self.dt, L_r=self.L_r, L_f=self.L_f, M=self.M)
        self.dynamics = bike_dynamics_rk4(dyn_params)

        tracking_params = trackingParams(dt=self.dt, N=self.N, n=self.n_x, d=self.n_u,
            Q=self.Q, R=self.R, R_d=self.R_d,
            z_l=np.array([self.x_min, self.y_min, self.heading_min, self.v_min]), z_u=np.array([self.x_max, self.y_max, self.heading_max, self.v_max]),
            u_l=np.array([self.steer_min,self.accel_min]), u_u=np.array([self.steer_max,self.accel_max]),
            du_l=np.array([self.dsteer_min,self.daccel_min]), du_u=np.array([self.dsteer_max,self.daccel_max]),
            optlevel=self.optlevel)
        self.tracking_controller = trackingController(self.dynamics, tracking_params)
        self.tracking_controller.initialize()

        self.state = np.zeros(self.n_x)
        self.last_state = np.zeros(self.n_x)
        self.input = np.zeros(self.n_u)
        self.last_input = np.zeros(self.n_u)
        self.state_prediction = None
        self.input_prediction = None

        rospy.Subscriber('est_states', States, self.estimator_callback, queue_size=1)

        # Publisher for steering and motor control
        self.ecu_pub = rospy.Publisher('ecu', ECU, queue_size=1)
        # Publisher for state predictions
        self.pred_pub = rospy.Publisher('pred_states', Prediction, queue_size=1)
        # Publisher for data logger
        # self.log_pub = rospy.Publisher('log_states', States, queue_size=1)

        # Create bond to shutdown data logger and arduino interface when controller stops
        bond_id = rospy.get_param('car/name')
        self.bond_log = bondpy.Bond('controller_logger', bond_id)
        self.bond_ard = bondpy.Bond('controller_arduino', bond_id)

        self.start_time = 0
        self.task_finished = False

        self.rate = rospy.Rate(1.0/self.dt)

    def estimator_callback(self, msg):
        self.state = np.array([msg.x, msg.y, msg.psi, np.sign(msg.v_x)*np.sqrt(msg.v_x**2+msg.v_y**2)])

    def spin(self):
        rospy.sleep(self.init_time)

        rospy.loginfo('============ Controler START ============')
        self.start_time = rospy.get_rostime().to_sec()
        counter = 0

        while not rospy.is_shutdown():
            # Get state at current time
            state = self.state
            x, y, heading, v = state
            t = rospy.get_rostime().to_sec() - self.start_time

            ecu_msg = ECU()
            ecu_msg.servo = 0.0
            ecu_msg.motor = 0.0
            if t >= self.max_time and not self.task_finished:
                self.task_finished = True
                shutdown_msg = '============ Max time of %g reached. Controler SHUTTING DOWN ============' % self.max_time
            elif (x < self.x_boundary[0] or x > self.x_boundary[1]) or (y < self.y_boundary[0] or y > self.y_boundary[1]) and not self.task_finished:
                # Check if car has left the experiment area
                self.task_finished = True
                shutdown_msg = '============ Track bounds exceeded reached. Controler SHUTTING DOWN ============'
            elif la.norm(np.array([x,y])-self.trajectory[-1,:2]) <= 0.10:
                # self.task_finished = True
                shutdown_msg = '============ Goal position reached. Controler SHUTTING DOWN ============'

            if self.task_finished:
                self.ecu_pub.publish(ecu_msg)
                self.bond_log.break_bond()
                self.bond_ard.break_bond()
                rospy.loginfo(shutdown_msg)
                rospy.signal_shutdown(shutdown_msg)

            if counter >= self.traj_len - 1:
                Z_ref = np.tile(self.trajectory[-1], (self.N+1,1))
                Z_ref[:,3] = 0
                rospy.loginfo('TRACKING: End of trajectory reached')
            elif counter >= self.traj_len - (self.N+1):
                Z_ref = np.vstack((self.trajectory[counter:], np.tile(self.trajectory[-1], ((self.N+1)-(self.traj_len-counter),1))))
                Z_ref[:,3] = 0
            else:
                Z_ref = self.trajectory[counter:counter+self.N+1]

            if self.state_prediction is None:
                Z_ws = Z_ref
                U_ws = np.zeros((self.N, self.n_u))
            else:
                Z_ws = np.vstack((self.state_prediction[1:],
                    self.dynamics.f_dt(self.state_prediction[-1], self.input_prediction[-1], type='numpy')))
                U_ws = np.vstack((self.input_prediction[1:],
                    self.input_prediction[-1]))

            Z_pred, U_pred, status_sol = self.tracking_controller.solve(state, self.last_input, Z_ref, Z_ws, U_ws)

            if not status_sol['success']:
                rospy.loginfo('============ TRACKING: MPC not feasible ============')
                ecu_msg.servo = 0.0
                ecu_msg.motor = 0.0
            else:
                ecu_msg.servo = U_pred[0,0]
                ecu_msg.motor = U_pred[0,1]

            self.ecu_pub.publish(ecu_msg)
            self.last_input = U_pred[0]

            self.state_prediction = Z_pred
            self.input_prediction = U_pred

            pred_msg = Prediction()
            pred_msg.x = Z_pred[:,0]
            pred_msg.y = Z_pred[:,1]
            pred_msg.psi = Z_pred[:,2]
            pred_msg.v = Z_pred[:,3]
            pred_msg.df = U_pred[:,0]
            pred_msg.a = U_pred[:,1]
            self.pred_pub.publish(pred_msg)

            counter += 1

            self.rate.sleep()

if __name__ == '__main__':
    tracking_node = trackingControlNode()
    try:
        tracking_node.spin()
    except rospy.ROSInterruptException: pass
