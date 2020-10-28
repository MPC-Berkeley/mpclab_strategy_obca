#!/usr/bin python3

import rospy
from bondpy import bondpy
import numpy as np
import numpy.linalg as la

from std_msgs.msg import Float64
from barc.msg import ECU, States, Prediction

from mpclab_strategy_obca.control.safetyController import safetyController
from mpclab_strategy_obca.control.utils.types import safetyParams

from mpclab_strategy_obca.dynamics.dynamicsModels import bike_dynamics_rk4
from mpclab_strategy_obca.dynamics.utils.types import dynamicsKinBikeParams

from mpclab_strategy_obca.utils.utils import load_vehicle_trajectory, get_trajectory_waypoints

class safetyControlNode(object):
    def __init__(self):
        # Read parameter values from ROS parameter server
        rospy.init_node('safety_control')

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
        self.a_lim = np.array([self.accel_min, self.accel_max])

        self.v_ref = rospy.get_param('controller/safety/v_ref')
        self.n_x = rospy.get_param('controller/safety/n')
        self.n_u = rospy.get_param('controller/safety/d')
        self.N = rospy.get_param('controller/safety/N')

        self.P_accel = rospy.get_param('controller/safety/P_accel')
        self.I_accel = rospy.get_param('controller/safety/I_accel')
        self.D_accel = rospy.get_param('controller/safety/D_accel')
        self.P_steer = rospy.get_param('controller/safety/P_steer')
        self.I_steer = rospy.get_param('controller/safety/I_steer')
        self.D_steer = rospy.get_param('controller/safety/D_steer')
        self.deadband_accel = rospy.get_param('controller/safety/deadband_accel')
        self.deadband_steer = rospy.get_param('controller/safety/deadband_steer')

        self.L_r = rospy.get_param('controller/dynamics/L_r')
        self.L_f = rospy.get_param('controller/dynamics/L_f')
        self.M = rospy.get_param('controller/dynamics/M')

        self.EV_L = rospy.get_param('car/plot/L')
        self.EV_W = rospy.get_param('car/plot/W')
        self.collision_buffer_r = np.sqrt(self.EV_L**2+self.EV_W**2)

        x_init = rospy.get_param('car/car_init/x', 0.0)
        y_init = rospy.get_param('car/car_init/y', 0.0)
        heading_init = rospy.get_param('car/car_init/heading', 0.0)
        if type(heading_init) is str:
            heading_init = eval(heading_init)
        v_init = rospy.get_param('car/car_init/v', 0.0)

        self.x_boundary = rospy.get_param('/track/x_boundary')
        self.y_boundary = rospy.get_param('/track/y_boundary')

        dyn_params = dynamicsKinBikeParams(dt=self.dt, L_r=self.L_r, L_f=self.L_f, M=self.M)
        self.dynamics = bike_dynamics_rk4(dyn_params)

        safety_params = safetyParams(dt=self.dt,
            P_accel=self.P_accel, I_accel=self.I_accel, D_accel=self.D_accel,
            P_steer=self.P_steer, I_steer=self.I_steer, D_steer=self.D_steer,
            accel_max=self.accel_max, accel_min=self.accel_min,
            daccel_max=self.daccel_max, daccel_min=self.daccel_min,
            steer_max=self.steer_max, steer_min=self.steer_min,
            dsteer_max=self.dsteer_max, dsteer_min=self.dsteer_min,
            deadband_accel=self.deadband_accel, deadband_steer=self.deadband_steer)
        self.safety_controller = safetyController(safety_params)

        self.state = np.zeros(self.n_x)
        self.last_state = np.zeros(self.n_x)
        self.input = np.zeros(self.n_u)
        self.last_input = np.zeros(self.n_u)
        self.state_prediction = None
        self.input_prediction = None

        rospy.Subscriber('est_states', States, self.estimator_callback, queue_size=1)
        rospy.Subscriber('/target_vehicle/pred_states', Prediction, self.prediction_callback, queue_size=1)

        # Publisher for steering and motor control
        self.ecu_pub = rospy.Publisher('ecu', ECU, queue_size=1)
        # Publisher for tracking controller node start time (used as reference start time)
        self.start_pub = rospy.Publisher('/start_time', Float64, queue_size=1, latch=True)
        # Publisher for data logger
        # self.log_pub = rospy.Publisher('log_states', States, queue_size=1)

        # Create bond to shutdown data logger and arduino interface when controller stops
        bond_id = rospy.get_param('car/name')
        self.bond_log = bondpy.Bond('controller_logger', bond_id)
        self.bond_ard = bondpy.Bond('controller_arduino', bond_id)

        self.start_time = None
        self.task_finish = False
        self.task_start = False

        self.rate = rospy.Rate(1.0/self.dt)

    def estimator_callback(self, msg):
        self.state = np.array([msg.x, msg.y, msg.psi, np.sign(msg.v_x)*np.sqrt(msg.v_x**2+msg.v_y**2)])

    def prediction_callback(self, msg):
        self.tv_state_prediction = np.vstack((msg.x, msg.y, msg.psi, msg.v)).T

    def spin(self):
        start_time_msg = rospy.wait_for_message('/start_time', Float64, timeout=5.0)
        self.start_time = start_time_msg.data
        rospy.sleep(0.5)
        counter = 0
        rospy.loginfo('============ SAFETY: Node START, time %g ============' % self.start_time)

        while not rospy.is_shutdown():
            t = rospy.get_rostime().to_sec() - self.start_time

            # Get EV state and TV prediction at current time
            EV_state = self.state
            TV_pred = self.tv_state_prediction[:self.N+1]

            EV_x, EV_y, EV_heading, EV_v = EV_state
            TV_x, TV_y, TV_heading, TV_v = TV_pred[0]

            ecu_msg = ECU()
            ecu_msg.servo = 0.0
            ecu_msg.motor = 0.0

            if t >= self.init_time and not self.task_start:
                self.task_start = True
                rospy.loginfo('============ SAFETY: Controler START ============')

            if t >= self.max_time and not self.task_finish:
                self.task_finish = True
                shutdown_msg = '============ SAFETY: Max time of %g reached. Controler SHUTTING DOWN ============' % self.max_time
            elif (EV_x < self.x_boundary[0] or EV_x > self.x_boundary[1]) or (EV_y < self.y_boundary[0] or EV_y > self.y_boundary[1]) and not self.task_finish:
                # Check if car has left the experiment area
                self.task_finish = True
                shutdown_msg = '============ SAFETY: Track bounds exceeded. Controler SHUTTING DOWN ============'
            # elif np.abs(y) > 0.7 and np.abs(heading - np.pi/2) <= 10*np.pi/180:
            elif np.abs(EV_y) > 0.7:
                self.task_finish = True
                shutdown_msg = '============ SAFETY: Goal position reached. Controler SHUTTING DOWN ============'

            if self.task_finish:
                self.ecu_pub.publish(ecu_msg)
                self.bond_log.break_bond()
                self.bond_ard.break_bond()
                rospy.loginfo(shutdown_msg)
                rospy.signal_shutdown(shutdown_msg)

            # Compute the distance threshold for applying braking assuming max
    	    # decceleration is applied

            rel_vx = TV_v * np.cos(TV_heading) - EV_v * np.cos(EV_heading)
            min_ts = np.ceil(-rel_vx / np.abs(self.a_lim[0]) / self.dt) # Number of timesteps requred for relative velocity to be zero
            v_brake = np.abs(rel_vx) + np.arange(min_ts+1) * self.dt * self.a_lim[0] # Velocity when applying max decceleration
            brake_thresh = np.sum( np.abs(v_brake) * self.dt ) + 5 * self.collision_buffer_r # Distance threshold for safety controller to be applied
            d = la.norm(np.array([EV_x-TV_x, EV_y-TV_y])) # Distance between ego and target vehicles

            if d <= brake_thresh:
                self.safety_controller.set_accel_ref(EV_v*np.cos(EV_heading), TV_v*np.cos(TV_heading))
            else:
                self.safety_controller.set_accel_ref(EV_v*np.cos(EV_heading), self.v_ref)

            u_safe = self.safety_controller.solve(EV_state, TV_pred, self.last_input)

            if not self.task_start:
                rospy.loginfo('============ SAFETY: Node up time: %g s. Controller not yet started ============' % t)
                self.last_input = np.zeros(self.n_u)
            else:
                rospy.loginfo('============ SAFETY: Applying safety PID control ============')
                ecu_msg.servo = u_safe[0]
                ecu_msg.motor = u_safe[1]
                self.last_input = u_safe

            self.ecu_pub.publish(ecu_msg)

            if self.task_start:
                counter += 1

            self.rate.sleep()

if __name__ == '__main__':
    safety_node = safetyControlNode()
    try:
        safety_node.spin()
    except rospy.ROSInterruptException: pass
