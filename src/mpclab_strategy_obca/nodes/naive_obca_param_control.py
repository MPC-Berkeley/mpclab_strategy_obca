#!/usr/bin python3

import rospy
from bondpy import bondpy
import numpy as np

from barc.msg import ECU, States, Prediction

from mpclab_strategy_obca.control.OBCAController import NaiveOBCAParameterizedController
from mpclab_strategy_obca.control.safetyController import safetyController, emergencyController
from mpclab_strategy_obca.control.utils.types import strategyOBCAParams, safetyParams

from mpclab_strategy_obca.dynamics.dynamicsModels import bike_dynamics_rk4
from mpclab_strategy_obca.dynamics.utils.types import dynamicsKinBikeParams

from mpclab_strategy_obca.utils.utils import get_car_poly, check_collision_poly

class naiveOBCAParameterizedControlNode(object):
    def __init__(self):
        # Read parameter values from ROS parameter server
        rospy.init_node('naive_obca_param_control')

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

        self.v_ref = rospy.get_param('controller/obca/v_ref')
        self.n_x = rospy.get_param('controller/obca/n')
        self.n_u = rospy.get_param('controller/obca/d')
        self.N = rospy.get_param('controller/obca/N')
        self.n_obs = rospy.get_param('controller/obca/n_obs')
        self.n_ineq = rospy.get_param('controller/obca/n_ineq')
        self.d_ineq = rospy.get_param('controller/obca/d_ineq')
        self.d_min = rospy.get_param('controller/obca/d_min')
        self.Q = rospy.get_param('controller/obca/Q')
        self.R = rospy.get_param('controller/obca/R')
        self.R_d = rospy.get_param('controller/obca/R_d')
        self.optlevel = rospy.get_param('controller/obca/optlevel')

        self.P_accel = rospy.get_param('controller/safety/P_accel')
        self.I_accel = rospy.get_param('controller/safety/I_accel')
        self.D_accel = rospy.get_param('controller/safety/D_accel')
        self.P_steer = rospy.get_param('controller/safety/P_steer')
        self.I_steer = rospy.get_param('controller/safety/I_steer')
        self.D_steer = rospy.get_param('controller/safety/D_steer')

        self.L_r = rospy.get_param('controller/dynamics/L_r')
        self.L_f = rospy.get_param('controller/dynamics/L_f')
        self.M = rospy.get_param('controller/dynamics/M')

        self.EV_L = rospy.get_param('car/plot/L')
        self.EV_W = rospy.get_param('car/plot/W')
        self.TV_L = rospy.get_param('/target_vehicle/car/plot/L')
        self.TV_W = rospy.get_param('/target_vehicle/car/plot/W')

        dyn_params = dynamicsKinBikeParams(dt=self.dt, L_r=self.L_r, L_f=self.L_f, M=self.M)
        self.dynamics = bike_dynamics_rk4(dyn_params)

        G = np.array([[1,0], [-1, 0], [0, 1], [0, -1]])
        g = np.array([self.EV_L/2, self.EV_L/2, self.EV_W/2, self.EV_W/2])
        obca_params = strategyOBCAParams(dt=self.dt, N=self.N, n=self.n_x, d=self.n_u,
            n_obs=self.n_obs, n_ineq=self.n_ineq, d_ineq=self.d_ineq,
            G=G, g=g, Q=self.Q, R=self.R, R_d=self.R_d, d_min=self.d_min,
            u_l=np.array([self.steer_min,self.accel_min]), u_u=np.array([self.steer_max,self.accel_max]),
            du_l=np.array([self.dsteer_min,self.daccel_min]), du_u=np.array([self.dsteer_max,self.daccel_max]),
            optlevel=self.optlevel)
        self.obca_controller = NaiveOBCAParameterizedController(self.dynamics, obca_params)
        self.obca_controller.initialize(regen=False)

        safety_params = safetyParams(dt=self.dt,
            P_accel=self.P_accel, I_accel=self.I_accel, D_accel=self.D_accel,
            P_steer=self.P_steer, I_steer=self.I_steer, D_steer=self.D_steer,
            accel_max=self.accel_max, accel_min=self.accel_min,
            daccel_max=self.daccel_max, daccel_min=self.daccel_min,
            steer_max=self.steer_max, steer_min=self.steer_min,
            dsteer_max=self.dsteer_max, dsteer_min=self.dsteer_min)
        self.safety_controller = safetyController(safety_params)

        emergency_params = safetyParams(dt=self.dt,
            P_accel=self.P_accel, I_accel=self.I_accel, D_accel=self.D_accel,
            P_steer=self.P_steer, I_steer=self.I_steer, D_steer=self.D_steer,
            accel_max=self.accel_max, accel_min=self.accel_min,
            daccel_max=self.daccel_max, daccel_min=self.daccel_min,
            steer_max=self.steer_max, steer_min=self.steer_min,
            dsteer_max=self.dsteer_max, dsteer_min=self.dsteer_min)
        self.emergency_controller = emergencyController(emergency_params)

        self.obs = [[] for _ in range(self.n_obs)]
        for i in range(self.N+1):
            # Add lane constraints to end of obstacle list
            self.obs[-2].append({'A': np.array([0,-1]).reshape((-1,1)), 'b': np.array([-self.lanewidth/2])})
            self.obs[-1].append({'A': np.array([0,1]).reshape((-1,1)), 'b': np.array([-self.lanewidth/2])})

        self.state = np.zeros(self.n_x)
        self.last_state = np.zeros(self.n_x)
        self.tv_state_prediction = np.zeros((self.N+1,self.n_x))
        self.input = np.zeros(self.n_u)
        self.last_input = np.zeros(self.n_u)
        self.ev_state_prediction = None
        self.ev_input_prediction = None

        rospy.Subscriber('est_states', States, self.estimator_callback, queue_size=1)
        rospy.Subscriber('/target_vehicle/pred_states', Prediction, self.prediction_callback, queue_size=1)

        # Publisher for steering and motor control
        self.ecu_pub = rospy.Publisher('ecu', ECU, queue_size=1)
        # Publisher for mpc prediction
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

    def prediction_callback(self, msg):
        self.tv_state_prediction = np.vstack((msg.x, msg.y, msg.psi, msg.v)).T

    def spin(self):
        rospy.sleep(self.init_time)
        self.start_time = rospy.get_rostime().to_sec()

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

            EV_state = self.state
            TV_pred = self.tv_state_prediction[:self.N+1]

            EV_x, EV_y, EV_heading, EV_v = EV_state
            TV_x, TV_y, TV_heading, TV_v = TV_pred[0]

            X_ref = EV_x + np.arange(self.N+1)*self.dt*self.v_ref
            Z_ref = np.zeros((self.N+1, self.n_x))
            Z_ref[:,0] = X_ref

            self.obs[0] = get_car_poly(TV_pred, self.TV_W, self.TV_L)

            if self.ev_state_prediction is None:
                Z_ws = Z_ref
                U_ws = np.zeros((self.N, self.n_u))
            else:
                Z_ws = np.vstack((self.ev_state_prediction[1:],
                    self.dynamics.f_dt(self.ev_state_prediction[-1], self.ev_input_prediction[-1], type='numpy')))
                U_ws = np.vstack((self.ev_input_prediction[1:],
                    self.ev_input_prediction[-1]))

            obca_mpc_ebrake = False
            obca_mpc_safety = False
            status_ws = self.obca_controller.solve_ws(Z_ws, U_ws, self.obs)
            if status_ws['success']:
                rospy.loginfo('Warm start solved in %g s' % status_ws['solve_time'])
                Z_obca, U_obca, status_sol = self.obca_controller.solve(EV_state, self.last_input, Z_ref, self.obs)

            if not status_ws['success'] or not status_sol['success']:
                obca_mpc_safety = True
                rospy.loginfo('OBCA MPC not feasible, activating safety controller')

            if obca_mpc_safety:
                self.safety_controller.set_accel_ref(EV_v*np.cos(EV_heading), TV_v*np.cos(TV_heading))
                u_safe = self.safety_controller.solve(EV_state, TV_pred, self.last_input)

                z_next = self.dynamics.f_dt(EV_state, u_safe, type='numpy')
                collision = check_collision_poly(z_next, (self.EV_W, self.EV_L), TV_pred[1], (self.TV_W, self.TV_L))
                if collision:
                    obca_mpc_ebrake = True
                    u_safe = self.emergency_controller.solve(EV_state, TV_pred, self.last_input)

                U_pred = np.vstack((u_safe, np.zeros((self.N-1, self.n_u))))
                Z_pred = np.vstack((EV_state, np.zeros((self.N, self.n_x))))
                for i in range(self.N):
                    Z_pred[i+1] = self.dynamics.f_dt(Z_pred[i], U_pred[i], type='numpy')
            else:
                Z_pred, U_pred = Z_obca, U_obca

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
            pred_msg.df = U_pred[:,0]
            pred_msg.a = U_pred[:,1]
            self.pred_pub.publish(pred_msg)

            self.rate.sleep()

if __name__ == '__main__':
    naive_obca_node = naiveOBCAParameterizedControlNode()
    try:
        naive_obca_node.spin()
    except rospy.ROSInterruptException: pass
