import rospy
from bondpy import bondpy
import numpy as np

from barc.msg import ECU, States, Prediction

from mpclab_strategy_obca.control.OBCAController import NaiveOBCAController
from mpclab_strategy_obca.control.safety_controller import safetyController
from mpclab_strategy_obca.control.utils.types import strategyOBCAParams

from mpclab_strategy_obca.dynamics.dynamicsModels import bike_dynamics_rk4
from mpclab_strategy_obca.dynamics.utils.types import dynamicsKinBikeParams

from mpclab_strategy_obca.utils.utils import get_car_poly

class strategyOBCAControlNode(object):
    def __init__(self):
        # Read parameter values from ROS parameter server

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
        self.Q = np.diag(rospy.get_param('controller/obca/Q'))
        self.R = np.diag(rospy.get_param('controller/obca/R'))
        self.optlevel = rospy.get_param('controller/obca/optlevel')

        self.P_accel = rospy.get_param('controller/safety/P_accel')
        self.I_accel = rospy.get_param('controller/safety/I_accel')
        self.D_accel = rospy.get_param('controller/safety/D_accel')
        self.P_speed = rospy.get_param('controller/safety/P_speed')
        self.I_speed = rospy.get_param('controller/safety/I_speed')
        self.D_speed = rospy.get_param('controller/safety/D_speed')

        self.L_r = rospy.get_param('controller/dynamics/L_r')
        self.L_f = rospy.get_param('controller/dynamics/L_f')
        self.M = rospy.get_param('controller/dynamics/M')

        self.car_L = rospy.get_param('car/plot/L')
        self.car_W = rospy.get_param('car/plot/W')

        dyn_params = dynamicsKinBikeParams(dt=self.dt, L_r=self.L_r, L_f=self.L_f, M=self.M)
        dynamics = bike_dynamics_rk4(dyn_params)

        obca_params = strategyOBCAParams(dt=self.dt, N=self.N, n=self.n_x, d=self.n_u,
            n_obs=self.n_obs, n_ineq=self.n_ineq, d_ineq=self.d_ineq,
            G=G, g=g, Q=self.Q, R=self.R, d_min=self.d_min,
            u_l=np.array([self.steer_min,self.accel_min]), u_u=np.array([self.steer_max,self.accel_max]),
            du_l=np.array([self.dsteer_min,self.daccel_min]), du_u=np.array([self.dsteer_max,self.daccel_max]),
            optlevel=self.optlevel)
        controller = NaiveOBCAController(dynamics, obca_params)

        safety_params = safetyParams(dt=self.dt,
            P_accel=self.P_accel, I_accel=self.I_accel, D_accel=self.D_accel,
            P_speed=self.P_speed, I_speed=self.I_speed, D_speed=self.D_speed,
            accel_max=self.accel_max, accel_min=self.accel_min,
            daccel_max=self.daccel_max, daccel_min=self.daccel_min,
            speed_max=self.speed_max, speed_min=self.speed_min,
            dspeed_max=self.dspeed_max, dspeed_min=self.dspeed_min)
        safety_controller = safetyController(safety_params)

        self.obs = [[] for _ in range(self.n_obs)]
        for i in range(self.N+1):
            # Add lane constraints to end of obstacle list
            self.obs[-2].append({'A': np.array([0,-1]).reshape((-1,1)), 'b': -self.lanewidth/2})
            self.obs[-1].append({'A': np.array([0,1]).reshape((-1,1)), 'b': -self.lanewidth/2})

        self.state = np.zeros(self.n_x)
        self.last_state = np.zeros(self.n_x)
        self.tv_prediction = np.zeros(self.N+1,self.n_x)
        self.input = np.zeros(self.n_u)
        self.last_input = np.zeros(self.n_u)

        rospy.Subscriber('est_states', States, self.estimator_callback, queue_size=1)
        rospy.Subscriber('tv_prediction', Prediction, self.prediction_callback, queue_size=1)

        # Publisher for steering and motor control
        self.ecu_pub = rospy.Publisher('ecu', ECU, queue_size=1)
        # Publisher for data logger
        self.log_pub = rospy.Publisher('log_states', States, queue_size=1)

        # Create bond to shutdown data logger and arduino interface when controller stops
        bond_id = rospy.get_param('car/name')
        self.bond_log = bondpy.Bond('controller_logger', bond_id)
        self.bond_ard = bondpy.Bond('controller_arduino', bond_id)

        self.start_time = 0

    def estimator_callback(self, msg):
        self.state = np.array([msg.x, msg.y, msg.psi, np.sign(msg.v_x)*np.sqrt(msg.v_x**2+msg.v_y**2]))

    def prediction_callback(self, msg):
        self.tv_prediction = np.vstack((msg.x, msg.y, msg.psi, msg.v)).T

    def spin(self):
        rospy.sleep(self.init_time)
        self.start_time = rospy.get_rostime().to_sec()

        while not rospy.is_shutdown():
            t = rospy.get_rostime().to_sec()
            if t >= self.max_time:
                ecu_cmd.servo = 0.0
                ecu_cmd.motor = 0.0
                # Publish the final motor and steering commands
                self.ecu_pub.publish(ecu_cmd)

                self.bond_log.break_bond()
                self.bond_ard.break_bond()
                rospy.signal_shutdown('Max time of %g reached, controller shutting down...' % self.max_time)

        EV_state = self.state
        TV_pred = self.tv_prediction

        EV_x, EV_y, EV_heading, EV_v = EV_state

        x_ref = EV_x + np.arange(self.N+1)*self.dt*self.v_ref
        z_ref = np.zeros((self.N+1,self.n_x))
        z_ref[:,0] = x_ref

        tv_obs = get_car_poly(TV_pred, self.car_W, self.car_L)


if __name__ == '__main__':
    rospy.init_node('strategy_obca_control')
    loop_rate = 1.0/self.dt
    rate = rospy.Rate(loop_rate)

    strategy_obca_node = strategyOBCAControlNode()
    try:
        strategy_obca_node.spin()
    except rospy.ROSInterruptException: pass
