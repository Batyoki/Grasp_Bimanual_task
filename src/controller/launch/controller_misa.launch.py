from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    config_file = os.path.join(
        get_package_share_directory('controller'),  # your package name
        'config',
        'controllers.yaml'                          # your config file
    )

    controller_node = Node(
        package='controller',        # package name
        executable='controller',    # entry point (from setup.py)
        name='controller',          # 👈 MUST match YAML root key
        parameters=[config_file],   # 👈 load YAML
        output='screen'
    )

    return LaunchDescription([
        controller_node
    ])