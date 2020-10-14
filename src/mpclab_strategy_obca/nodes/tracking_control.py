#!/usr/bin python3

import rospy
from bondpy import bondpy
import numpy as np

from barc.msg import ECU, States, Prediction

from mpclab_strategy_obca.control.trackingController import trackingController
from mpclab_strategy_obca.control.utils.types import trackingParams

from mpclab_strategy_obca.dynamics.dynamicsModels import bike_dynamics_rk4
from mpclab_strategy_obca.dynamics.utils.types import dynamicsKinBikeParams

from mpclab_strategy_obca.utils.utils import load_vehicle_trajectory

class trackingControlNode(object):
    def __init__(self):
        # Read parameter values from ROS parameter server
        rospy.init_node('naive_obca_control')

        self.dt = rospy.get_param('controller/dt')
        self.init_time = rospy.get_param('controller/init_time')
        self.max_time = rospy.get_param('controller/max_time')
        self.accel_max = rospy.get_param('controller/accel_max')
        self.accel_min = rospy.get_param('controller/accel_min')
        self.steer_max = rospy.get_param('controller/steer_max')
        self.steer_min = rospy.get_param('controller/steer_min')
        self.daccel_max = rospy.get_param('controller/daccel_max')
        self.daccel_min = rospy.get_param('controller/daccel_min')
        self.dsteer_max = rospy.get_param('controller/dsteer_max')
        self.dsteer_min = rospy.get_param('controller/dsteer_min')
        self.lanewidth = rospy.get_param('controller/lanewidth')
        self.trajectory_file = rospy.get_param('controller/trajectory_file')

        self.n_x = rospy.get_param('controller/tracking/n')
        self.n_u = rospy.get_param('controller/tracking/d')
        self.N = rospy.get_param('controller/tracking/N')
        self.Q = np.diag(rospy.get_param('controller/tracking/Q'))
        self.R = np.diag(rospy.get_param('controller/tracking/R'))
        self.optlevel = rospy.get_param('controller/tracking/optlevel')

        self.L_r = rospy.get_param('controller/dynamics/L_r')
        self.L_f = rospy.get_param('controller/dynamics/L_f')
        self.M = rospy.get_param('controller/dynamics/M')

        self.EV_L = rospy.get_param('car/plot/L')
        self.EV_W = rospy.get_param('car/plot/W')

        self.trajectory = load_vehicle_trajectory(self.trajectory_file)
        rospy.set_param('car/car_init/x')
        rospy.set_param('car/car_init/y')
        rospy.set_param('car/car_init/heading')
        rospy.set_param('car/car_init/v')
        
        dyn_params = dynamicsKinBikeParams(dt=self.dt, L_r=self.L_r, L_f=self.L_f, M=self.M)
        self.dynamics = bike_dynamics_rk4(dyn_params)

        tracking_params = trackingParams(dt=self.dt, N=self.N, n=self.n_x, d=self.n_u,
            Q=self.Q, R=self.R,
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
        self.pred_pub = rospy.Publisher('prediction', Prediction, queue_size=1)
        # Publisher for data logger
        # self.log_pub = rospy.Publisher('log_states', States, queue_size=1)

        # Create bond to shutdown data logger and arduino interface when controller stops
        bond_id = rospy.get_param('car/name')
        self.bond_log = bondpy.Bond('controller_logger', bond_id)
        self.bond_ard = bondpy.Bond('controller_arduino', bond_id)

        self.start_time = 0

        self.rate = rospy.Rate(1.0/self.dt)

    def estimator_callback(self, msg):
        self.state = np.array([msg.x, msg.y, msg.psi, np.sign(msg.v_x)*np.sqrt(msg.v_x**2+msg.v_y**2)])

    def spin(self):
        rospy.sleep(self.init_time)
        self.start_time = rospy.get_rostime().to_sec()
        counter = 0

        while not rospy.is_shutdown():
            t = rospy.get_rostime().to_sec()
            ecu_msg = ECU()
            if t-self.start_time >= self.max_time:
                ecu_msg.servo = 0.0
                ecu_msg.motor = 0.0
                # Publish the final motor and steering commands
                self.ecu_pub.publish(ecu_msg)

                self.bond_log.break_bond()
                self.bond_ard.break_bond()
                rospy.signal_shutdown('Max time of %g reached, controller shutting down...' % self.max_time)

            state = self.state

            x, y, heading, v = state

            Z_ref = self.trajectory[counter:counter+self.N+1]

            if self.state_prediction is None:
                Z_ws = Z_ref
                U_ws = np.zeros((self.N, self.n_u))
            else:
                Z_ws = np.vstack((self.state_prediction[1:],
                    self.dynamics.f_dt(self.state_prediction[-1], self.ev_input_prediction[-1], type='numpy')))
                U_ws = np.vstack((self.input_prediction[1:],
                    self.input_prediction[-1]))

            Z_pred, U_pred, status_sol = self.tracking_controller.solve(state, self.last_input, Z_ref, Z_ws, U_ws)

            if not status_sol['success']:
                rospy.loginfo('Tracking MPC not feasible')

            self.ev_state_prediction = Z_pred
            self.ev_input_prediction = U_pred

            self.last_input = U_pred[0]

            ecu_msg = ECU()
            ecu_msg.servo = U_pred[0,0]
            ecu_msg.motor = U_pred[0,1]
            self.ecu_pub.publish(ecu_msg)

            pred_msg = Prediction()
            pred_msg.x = Z_pred[:,0]
            pred_msg.y = Z_pred[:,1]
            pred_msg.psi = Z_pred[:,2]
            pred_msg.v = Z_pred[:,3]
            pred_pub.publish(pred_msg)

            self.rate.sleep()

if __name__ == '__main__':
    tracking_node = trackingControlNode()
    try:
        tracking_node.spin()
    except rospy.ROSInterruptException: pass
