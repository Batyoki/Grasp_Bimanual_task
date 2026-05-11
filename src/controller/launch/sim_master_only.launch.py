"""
launch/sim_master_only.launch.py
---------------------------------
Launches just the simulated master arm for testing without real hardware
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    
    ld = LaunchDescription()
    
    # =========================================================================
    # 1. SIMULATED MASTER ARM NODE
    # =========================================================================
    
    simulated_master_node = Node(
        package='controller',
        executable='simulated_master',
        name='simulated_master_node',
        output='screen',
        parameters=[
            {'update_rate': 100},
        ]
    )
    ld.add_action(simulated_master_node)
    
    # =========================================================================
    # 2. MASTER STATE NODE
    # =========================================================================
    
    master_state_node = Node(
        package='controller',
        executable='master_state_node',
        name='master_state_node',
        output='screen'
    )
    ld.add_action(master_state_node)
    
    # =========================================================================
    # 3. JOYSTICK TELEOP NODE (optional)
    # =========================================================================
    
    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy',
        output='screen',
        parameters=[
            {'axis_linear': {'x': 1}},
            {'axis_angular': {'yaw': 0}},
        ],
        remappings=[
            ('cmd_vel', '/master_cmd_vel'),
        ]
    )
    # Uncomment to enable joystick control:
    # ld.add_action(teleop_node)
    
    return ld
