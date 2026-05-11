import rclpy
from rclpy.node import Node


class hardware_software_episodes(Node):

    def __init__(self):
        super().__init__('hardware_software')
        self.get_logger().info('Hardware to software episode recorder initialised')



def main(args=None):

    rclpy.init(args=args)
    node = hardware_software_episodes()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()