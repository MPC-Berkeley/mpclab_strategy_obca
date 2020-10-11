import rospy

from barc.msg import ECU, States

from mpclab_strategy_obca.control.OBCAController import NaiveOBCAController
from mpclab_strategy_obca.control.utils.types import strategyOBCAParams

class strategyOBCAControlNode(object):
    def __init__(self):
        # Read parameter values from ROS parameter server

        dt = rospy.get_param('controller/dt')
        N = rospy.get_param('controller/obca/N')

        obca_params = strategyOBCAParams(dt=dt, N=N)
        controller = NaiveOBCAController(obca_params)

        # controller.initialize(regen=True)

        loop_rate = 1.0/dt
        self.rate = rospy.Rate(loop_rate)

        # Publisher for steering and motor control
        self.ecu_pub = rospy.Publisher('ecu', ECU, queue_size=1)
        # Publisher for data logger
        self.log_pub = rospy.Publisher('log_states', States, queue_size=1)

        # Initialize state message for data logging
        # self.log_msg = States()

    def spin(self):
        while not rospy.is_shutdown():
            print('Hi')

if __name__ == '__main__':
    rospy.init_node('strategy_obca_control')
    strategy_obca_node = strategyOBCAControlNode()
    try:
        strategy_obca_node.spin()
    except rospy.ROSInterruptException: pass
