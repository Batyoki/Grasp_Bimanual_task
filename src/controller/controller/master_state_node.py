import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class master_state_node(Node):

    def __init__(self):
        super().__init__('master_state_node')
        self.get_logger().info('Master State Node')


        self.get_state_subscription_L = self.create_subscription(
            JointState,
            '/masterL/joint_states',
            lambda m: self.get_state_callback(m,"L"),
            10
        )

        self.get_state_subscription_R = self.create_subscription(
            JointState,
            '/masterR/joint_states',
            lambda m: self.get_state_callback(m,"R"),
            10
        )

        self.send_state_publisher_L = self.create_publisher(
            JointState,
            'controller/master_state_L',
            10
        )

        self.send_state_publisher_R = self.create_publisher(
            JointState,
            'controller/master_state_R',
            10
        )
    def get_state_callback(self,msg:JointState,s:str):
        slave_state_msg = msg

        #temporary effort and velocity set to 0

        # slave_state_msg.effort=[0.01]*len(msg.name)
        # slave_state_msg.velocity=[0.00]*len(msg.name)
        # print(slave_state_msg)
        
        if s=="R":
            self.send_state_publisher_R.publish(slave_state_msg)
            return
        self.send_state_publisher_L.publish(slave_state_msg)
        return
    
   

def main(args=None):
    rclpy.init(args=args)
    node = master_state_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()

