"""
launch/sim_master_keyboard.launch.py
------------------------------------
Launches simulated master arm with keyboard teleoperation
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    
    ld = LaunchDescription()
    
    # Simulated Master Node
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
    
    # Master State Node (pads 5 DOF to 8)
    master_state_node = Node(
        package='controller',
        executable='master_state_node',
        name='master_state_node',
        output='screen'
    )
    ld.add_action(master_state_node)
    
    # Keyboard Teleoperation Node
    keyboard_teleop_node = Node(
        package='controller',
        executable='keyboard_teleop',
        name='keyboard_teleop_node',
        output='screen'
    )
    ld.add_action(keyboard_teleop_node)
    
    return ld
