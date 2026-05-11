"""
launch/sim_master_real_slave.launch.py
----------------------------------------
Launches the bilateral teleoperation system with:
  - Simulated master arm (rx150) on desktop/joystick
  - Real slave arm (vx300s) via InterbotiX control
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    
    # Parameters
    master_type = 'rx150'  # Simulated master
    slave_type = 'vx300s'   # Real slave (InterbotiX controlled)
    use_sim_master = True
    
    ld = LaunchDescription()
    
    # =========================================================================
    # 1. SIMULATED MASTER ARM NODE
    # =========================================================================
    # This publishes simulated joint states with torque/velocity values
    # for testing without hardware
    
    simulated_master_node = Node(
        package='controller',
        executable='simulated_master',
        name='simulated_master_node',
        output='screen',
        parameters=[
            {'update_rate': 100},  # 100 Hz simulation
        ]
    )
    ld.add_action(simulated_master_node)
    
    # =========================================================================
    # 2. MASTER STATE NODE
    # =========================================================================
    # Converts raw master joint states to controller format
    # Pads rx150 (5 DOF) to 8 elements for compatibility
    
    master_state_node = Node(
        package='controller',
        executable='master_state_node',
        name='master_state_node',
        output='screen'
    )
    ld.add_action(master_state_node)
    
    # =========================================================================
    # 3. REAL SLAVE ARM - InterbotiX Control
    # =========================================================================
    # Try to launch InterbotiX control for the real vx300s slave
    # This would typically include the joint controllers, hardware drivers, etc.
    
    try:
        interbotix_launch_file = os.path.join(
            FindPackageShare('interbotix_xsarm_control').find('interbotix_xsarm_control'),
            'launch',
            'xsarm_control.launch.py'
        )
        
        if os.path.exists(interbotix_launch_file):
            interbotix_control = IncludeLaunchDescription(
                interbotix_launch_file,
                launch_arguments={
                    'robot_model': 'vx300s',
                    'dof': '6',
                    'use_rviz': 'false',
                }.items()
            )
            ld.add_action(interbotix_control)
    except:
        # If InterbotiX packages not found, just note it
        print("InterbotiX packages not found - real slave control not available")
        pass
    
    # =========================================================================
    # 4. SLAVE STATE NODE
    # =========================================================================
    # Converts real slave joint states to controller format
    # Removes index 3 (forearm_roll) from vx300s and pads back to 8 elements
    
    slave_state_node = Node(
        package='controller',
        executable='slave_state_node',
        name='slave_state_node',
        output='screen'
    )
    ld.add_action(slave_state_node)
    
    # =========================================================================
    # 5. BILATERAL CONTROLLER NODE
    # =========================================================================
    # Main control loop that reads master/slave states and publishes commands
    
    controller_node = Node(
        package='controller',
        executable='controller',
        name='controller',
        output='screen',
        parameters=[
            {
                'control_mode': 'MISA',  # Master Impedance, Slave Admittance
                'loop_rate_hz': 100.0,
                'mpc_horizon': 10,
                'state_timeout_s': 0.5,
                'master_Mmd_diag': [0.027, 0.042, 0.018, 0.005, 0.001],
                'master_Dmd_diag': [0.5, 0.8, 0.5, 0.2, 0.1],
                'master_Kmd_diag': [0.0, 0.0, 0.0, 0.0, 0.0],
                'slave_Msd_diag': [0.8, 1.2, 0.8, 0.4, 0.25],
                'slave_Dsd_diag': [4.0, 6.0, 4.0, 2.5, 1.5],
                'slave_Ksd_diag': [8.0, 12.0, 8.0, 4.0, 2.5],
                'kof': 0.8,  # slave→master force coupling
                'ksf': 0.2,  # local master force coupling
                'kom': 0.8,  # slave→master position coupling
                'ksm': 0.2,  # local master position coupling
            }
        ]
    )
    ld.add_action(controller_node)
    
    # =========================================================================
    # 6. RVIZ VISUALIZATION (optional)
    # =========================================================================
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(
            FindPackageShare('controller').find('controller'),
            'config',
            'bilateral.rviz'
        )],
    )
    # Uncomment to enable visualization:
    # ld.add_action(rviz_node)
    
    return ld
