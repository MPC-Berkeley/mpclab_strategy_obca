import rospy


class strategyOBCAControlNode(object):
    def __init__(self):
        pass

    def spin(self):
        pass

if __name__ == '__main__':
    rp.init_node('strategy_obca_control')
    strategy_obca_node = strategyOBCAControlNode()
    try:
        strategy_obca_node.spin()
    except rospy.ROSInterruptException: pass
