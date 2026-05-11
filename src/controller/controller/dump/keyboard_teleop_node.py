"""
keyboard_teleop_node.py
-----------------------
Keyboard-based teleoperation for the simulated master arm (rx150, 5-DOF).

Key Mapping:
  Waist (joint 0):
    'a' / 'd'          - Rotate left / right
  
  Shoulder (joint 1):
    'w' / 's'          - Raise / lower
  
  Elbow (joint 2):
    'q' / 'e'          - Flex / extend
  
  Wrist Angle (joint 3):
    'r' / 'f'          - Pitch up / down
  
  Wrist Rotate (joint 4):
    'z' / 'x'          - Roll left / right
  
  Speed Control:
    'up' / 'down'      - Increase / decrease speed (10% increments)
    'space'            - Stop all joints (zero velocity)
    
  System:
    'h'                - Print help
    'ctrl+c'           - Exit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import sys
import threading
from pynput import keyboard


class KeyboardTeleopNode(Node):
    """Keyboard-based teleoperation for master arm."""

    def __init__(self):
        super().__init__('keyboard_teleop_node')
        
        # Joint configuration
        self.joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']
        self.n_joints = 5
        
        # Desired velocities for each joint (rad/s)
        self.cmd_vel = np.zeros(5)
        
        # Speed scaling (multiply by this factor)
        self.speed_scale = 1.0
        self.base_speed = 1.0  # rad/s per key press
        
        # Maximum velocities
        self.max_vel = np.array([2.0, 1.5, 1.5, 1.0, 1.0])
        
        # Publisher for commands
        self.cmd_pub = self.create_publisher(
            JointState,
            '/master_cmd_vel',
            10
        )
        
        # Timer to publish commands at regular rate
        self.timer = self.create_timer(0.05, self.publish_command)  # 20 Hz
        
        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        # Track which keys are currently pressed
        self.pressed_keys = set()
        
        self.print_help()
        self.get_logger().info('Keyboard Teleop Node initialized - Press keys to move arm')

    def on_press(self, key):
        """Handle key press."""
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.add(key.char.lower())
            else:
                self.pressed_keys.add(key)
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release."""
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.discard(key.char.lower())
            else:
                self.pressed_keys.discard(key)
        except AttributeError:
            pass
        
        # Check for special commands on release
        try:
            if key == keyboard.Key.esc:
                return False  # Stop listener
        except AttributeError:
            pass

    def update_velocities(self):
        """Update command velocities based on pressed keys."""
        self.cmd_vel = np.zeros(5)
        
        # Waist (joint 0): a/d
        if 'a' in self.pressed_keys:
            self.cmd_vel[0] = self.base_speed * self.speed_scale
        if 'd' in self.pressed_keys:
            self.cmd_vel[0] = -self.base_speed * self.speed_scale
        
        # Shoulder (joint 1): w/s
        if 'w' in self.pressed_keys:
            self.cmd_vel[1] = self.base_speed * self.speed_scale
        if 's' in self.pressed_keys:
            self.cmd_vel[1] = -self.base_speed * self.speed_scale
        
        # Elbow (joint 2): q/e
        if 'q' in self.pressed_keys:
            self.cmd_vel[2] = self.base_speed * self.speed_scale
        if 'e' in self.pressed_keys:
            self.cmd_vel[2] = -self.base_speed * self.speed_scale
        
        # Wrist Angle (joint 3): r/f
        if 'r' in self.pressed_keys:
            self.cmd_vel[3] = self.base_speed * self.speed_scale
        if 'f' in self.pressed_keys:
            self.cmd_vel[3] = -self.base_speed * self.speed_scale
        
        # Wrist Rotate (joint 4): z/x
        if 'z' in self.pressed_keys:
            self.cmd_vel[4] = self.base_speed * self.speed_scale
        if 'x' in self.pressed_keys:
            self.cmd_vel[4] = -self.base_speed * self.speed_scale
        
        # Speed control: up/down arrows
        if keyboard.Key.up in self.pressed_keys:
            self.speed_scale = min(2.0, self.speed_scale + 0.1)
            self.get_logger().info(f'Speed: {self.speed_scale:.1f}x')
        
        if keyboard.Key.down in self.pressed_keys:
            self.speed_scale = max(0.1, self.speed_scale - 0.1)
            self.get_logger().info(f'Speed: {self.speed_scale:.1f}x')
        
        # Stop: space bar
        if ' ' in self.pressed_keys:
            self.cmd_vel = np.zeros(5)
        
        # Help: h
        if 'h' in self.pressed_keys:
            self.print_help()
        
        # Clamp to max velocities
        for i in range(self.n_joints):
            self.cmd_vel[i] = np.clip(self.cmd_vel[i], -self.max_vel[i], self.max_vel[i])

    def publish_command(self):
        """Publish the current joint command."""
        self.update_velocities()
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.velocity = self.cmd_vel.tolist()
        
        self.cmd_pub.publish(msg)

    def print_help(self):
        """Print keyboard mapping help."""
        help_text = """
╔════════════════════════════════════════════════════════════╗
║        KEYBOARD TELEOPERATION - RX150 Master Arm           ║
╚════════════════════════════════════════════════════════════╝

┌─── JOINT CONTROL ──────────────────────────────────────────┐
│ WAIST (joint 0):                                           │
│   A / D              - Rotate left / right                 │
│                                                             │
│ SHOULDER (joint 1):                                        │
│   W / S              - Raise / lower                       │
│                                                             │
│ ELBOW (joint 2):                                           │
│   Q / E              - Flex / extend                       │
│                                                             │
│ WRIST ANGLE (joint 3):                                     │
│   R / F              - Pitch up / down                     │
│                                                             │
│ WRIST ROTATE (joint 4):                                    │
│   Z / X              - Roll left / right                   │
└─────────────────────────────────────────────────────────────┘

┌─── SPEED CONTROL ──────────────────────────────────────────┐
│ UP / DOWN arrows     - Increase / decrease speed (10%)     │
│ Current speed: {:.1f}x                              │
│ SPACE                - Stop all joints                     │
│ H                    - Print this help                     │
│ ESC or Ctrl+C        - Exit                                │
└─────────────────────────────────────────────────────────────┘

        Press keys to move the arm! (Multiple keys OK)
        """.format(self.speed_scale)
        
        self.get_logger().info(help_text)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = KeyboardTeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
