import rclpy
from rclpy.node import Node

class software_software_episodes(Node):
    def __init__(self):
        super().__init__('software_software')
        self.get_logger().info("Software to Software node initialised")


def main(args=None):
    rclpy.init(args=args)
    node = software_software_episodes()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if(__name__=="__main__"):
    main()