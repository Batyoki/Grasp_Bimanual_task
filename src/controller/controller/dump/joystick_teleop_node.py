"""
joystick_teleop_node.py
-----------------------
Converts joystick input to joint velocity commands for the simulated master arm.

Typical joystick mapping (e.g., PS4 / Xbox controller):
  - Left stick: Shoulder + Elbow (joints 1-2)
  - Right stick: Wrist (joints 3-4)
  - D-pad or buttons: Waist (joint 0)
  - Triggers: Gripper (not used in this 5-DOF arm)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
import numpy as np


class JoystickTeleopNode(Node):
    """Maps joystick axes to master arm joint commands."""

    def __init__(self):
        super().__init__('joystick_teleop_node')
        
        # Joint names for rx150
        self.joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']
        self.n_joints = 5
        
        # Speed scaling factors for each joint (rad/s per axis unit)
        self.speed_scale = np.array([
            2.0,   # waist (joint 0)
            1.5,   # shoulder (joint 1)
            1.5,   # elbow (joint 2)
            1.0,   # wrist_angle (joint 3)
            1.0,   # wrist_rotate (joint 4)
        ])
        
        # Joystick axis mapping (for PS4/typical layout)
        # Axes: 0=left_x, 1=left_y, 2=right_x, 3=right_y, 4=L2, 5=R2
        self.axis_map = {
            'waist': 0,           # Left X
            'shoulder': 1,        # Left Y (inverted)
            'elbow': 3,           # Right Y (inverted)
            'wrist_angle': 2,     # Right X
            'wrist_rotate': 4,    # L2 (or use D-pad)
        }
        
        # Subscribe to joystick
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        
        # Publish desired joint velocities
        self.cmd_vel_pub = self.create_publisher(
            JointState,
            '/master_cmd_vel',
            10
        )
        
        self.get_logger().info('Joystick Teleop Node initialized')
        self.get_logger().info('Using axis mapping: {}'.format(self.axis_map))

    def joy_callback(self, msg: Joy):
        """
        Convert joystick input to joint velocity commands.
        """
        
        # Initialize command
        cmd_vel = np.zeros(self.n_joints)
        
        # Map axes to joint velocities
        try:
            if len(msg.axes) > 0:
                # Waist: left stick X
                if self.axis_map['waist'] < len(msg.axes):
                    cmd_vel[0] = msg.axes[self.axis_map['waist']] * self.speed_scale[0]
                
                # Shoulder: left stick Y (inverted)
                if self.axis_map['shoulder'] < len(msg.axes):
                    cmd_vel[1] = -msg.axes[self.axis_map['shoulder']] * self.speed_scale[1]
                
                # Elbow: right stick Y (inverted)
                if self.axis_map['elbow'] < len(msg.axes):
                    cmd_vel[2] = -msg.axes[self.axis_map['elbow']] * self.speed_scale[2]
                
                # Wrist angle: right stick X
                if self.axis_map['wrist_angle'] < len(msg.axes):
                    cmd_vel[3] = msg.axes[self.axis_map['wrist_angle']] * self.speed_scale[3]
                
                # Wrist rotate: triggers (L2/R2)
                if self.axis_map['wrist_rotate'] < len(msg.axes):
                    cmd_vel[4] = msg.axes[self.axis_map['wrist_rotate']] * self.speed_scale[4]
                
                # Alternative: use D-pad for wrist_rotate
                if len(msg.axes) > 6:  # D-pad as axes
                    if msg.axes[6] != 0:  # D-pad left/right
                        cmd_vel[4] = msg.axes[6] * self.speed_scale[4]
            
            # Clamp to reasonable velocities
            cmd_vel = np.clip(cmd_vel, -3.0, 3.0)
            
            # Publish command
            cmd_msg = JointState()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.name = self.joint_names
            cmd_msg.velocity = cmd_vel.tolist()
            
            self.cmd_vel_pub.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().warn(f'Error processing joystick: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = JoystickTeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
