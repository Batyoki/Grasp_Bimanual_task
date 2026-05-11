"""
simulated_master_node.py
------------------------
Simulates the master arm (rx150, 5-DOF) joint states with synthetic
torque and velocity values for teleoperation testing.

This node:
1. Subscribes to joy input (joystick) for user teleoperation
2. Simulates joint dynamics with simple models
3. Publishes joint states with simulated torque and velocity
4. Includes safety limits and gravity compensation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Twist
import numpy as np
import time
from threading import Lock


class SimulatedMasterNode(Node):
    """Simulates rx150 (5-DOF) master arm with torque and velocity."""

    def __init__(self):
        super().__init__('simulated_master_node')
        
        # Arm parameters (rx150 5-DOF: waist, shoulder, elbow, wrist_angle, wrist_rotate)
        self.n_joints = 5
        self.joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']
        
        # Joint limits (radians)
        self.q_min = np.array([-np.pi, -1.8501, -1.7802, -1.7453, -np.pi])
        self.q_max = np.array([np.pi, 1.7453, 1.6580, 2.1468, np.pi])
        
        # Velocity limits (rad/s)
        self.dq_max = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        
        # Gravity compensation (approximate)
        self.gravity_taus = np.array([0.0, -0.981, -0.718, -0.187, 0.0])
        
        # Control parameters
        self.dt = 0.01  # 100 Hz
        
        # State variables
        self.q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.dq = np.zeros(5)
        self.tau_cmd = np.zeros(5)
        self.state_lock = Lock()
        
        # Simulation parameters
        self.M = np.diag([0.027, 0.042, 0.018, 0.005, 0.001])  # Inertia
        self.D = np.diag([0.5, 0.8, 0.5, 0.2, 0.1])  # Damping
        
        # User input (from joystick/teleop)
        self.joy_input = np.zeros(5)
        self.enable = False
        
        # Publishers
        self.joint_state_pub_L = self.create_publisher(
            JointState, '/masterL/joint_states', 10
        )
        self.joint_state_pub_R = self.create_publisher(
            JointState, '/masterR/joint_states', 10
        )
        
        # Subscribers
        self.joy_sub_L = self.create_subscription(
            Joy, '/joy_left', lambda msg: self.joy_callback(msg, 'L'), 10
        )
        self.joy_sub_R = self.create_subscription(
            Joy, '/joy_right', lambda msg: self.joy_callback(msg, 'R'), 10
        )
        self.enable_sub = self.create_subscription(
            Joy, '/enable', self.enable_callback, 10
        )
        
        # Timer for simulation loop
        self.create_timer(self.dt, self.simulation_step)
        
        self.get_logger().info(f'Simulated Master Node (rx150) initialized')

    def joy_callback(self, msg: Joy, arm: str):
        """
        Map joystick axes to desired joint velocities.
        
        Typical mapping for 5-DOF arm:
        - Axes 0-2: Shoulder/elbow (3 DOF)
        - Axes 3-4: Wrist (2 DOF)
        """
        with self.state_lock:
            if len(msg.axes) >= 5:
                # Scale joystick input to desired velocity (±2 rad/s max)
                self.joy_input = np.array(msg.axes[:5]) * 2.0
            
            # Buttons for enable/disable
            if len(msg.buttons) > 0:
                self.enable = bool(msg.buttons[0])

    def enable_callback(self, msg: Joy):
        """Simple enable/disable from joystick button."""
        if len(msg.buttons) > 0:
            self.enable = bool(msg.buttons[0])

    def simulation_step(self):
        """Execute one simulation step and publish state."""
        with self.state_lock:
            if not self.enable:
                # Passively relax to gravity-compensated position
                self.dq = 0.9 * self.dq - 0.1 * self.gravity_taus
            else:
                # Simple impedance control with user input
                # Desired velocity from joystick
                dq_desired = self.joy_input
                
                # Clamp desired velocity
                dq_desired = np.clip(dq_desired, -self.dq_max, self.dq_max)
                
                # Proportional damping to track desired velocity
                tau_prop = 10.0 * (dq_desired - self.dq)
                
                # Add gravity compensation
                self.tau_cmd = tau_prop + self.gravity_taus
            
            # Clamp torques to reasonable limits (±5 Nm)
            self.tau_cmd = np.clip(self.tau_cmd, -5.0, 5.0)
            
            # Simulate arm dynamics: M*ddq + D*dq = tau
            # => ddq = M^-1 * (tau - D*dq)
            ddq = np.linalg.solve(self.M, self.tau_cmd - self.D @ self.dq)
            
            # Numerical integration
            self.dq += ddq * self.dt
            self.q += self.dq * self.dt
            
            # Clamp positions to joint limits
            self.q = np.clip(self.q, self.q_min, self.q_max)
            
            # When hitting limits, zero out velocity in that direction
            for i in range(self.n_joints):
                if self.q[i] <= self.q_min[i] or self.q[i] >= self.q_max[i]:
                    self.dq[i] = 0.0
            
            # Create and publish joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.name = self.joint_names
            msg.position = self.q.tolist()
            msg.velocity = self.dq.tolist()
            # Effort is simulated torque
            msg.effort = self.tau_cmd.tolist()
            
            # Publish to both left and right (or just one based on arm param)
            self.joint_state_pub_L.publish(msg)
            self.joint_state_pub_R.publish(msg)
            
            self.get_logger().debug(
                f'q={self.q}, dq={self.dq}, tau={self.tau_cmd}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SimulatedMasterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
