import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class slave_state_node(Node):

    def __init__(self):
        super().__init__('slave_state_node')
        self.get_logger().info('Slave State Node')


        self.get_state_subscription_L = self.create_subscription(
            JointState,
            '/slaveL/joint_states',
            lambda m: self.get_state_callback(m,"L"),
            10
        )

        self.get_state_subscription_R = self.create_subscription(
            JointState,
            '/slaveR/joint_states',
            lambda m: self.get_state_callback(m,"R"),
            10
        )

        self.send_state_publisher_L = self.create_publisher(
            JointState,
            'controller/slave_state_L',
            10
        )

        self.send_state_publisher_R = self.create_publisher(
            JointState,
            'controller/slave_state_R',
            10
        )
    
    def get_state_callback(self,msg:JointState,s:str):
        master_state_msg = msg
        if len(master_state_msg.position)>0:
            master_state_msg.position.pop(3)
            # pass
        if len(master_state_msg.velocity)>0:
            master_state_msg.velocity.pop(3)
            # pass
        if len(master_state_msg.effort)>0:
            master_state_msg.effort.pop(3)
            # pass
        master_state_msg.name.pop(3)
        # print(msg.name)

        if(s=="R"):
            self.send_state_publisher_R.publish(master_state_msg)
            return
        
        self.send_state_publisher_R.publish(master_state_msg)

        return

def main(args=None):
    rclpy.init(args=args)
    node = slave_state_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
    
