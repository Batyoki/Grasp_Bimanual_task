from setuptools import find_packages, setup
import os

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            ['launch/sim_master_real_slave.launch.py', 'launch/sim_master_only.launch.py', 'launch/sim_master_keyboard.launch.py',
             'launch/controller_misa.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sp',
    maintainer_email='cs23bt072@iitdh.ac.in',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller = controller.controller:main',
            'master_state_node = controller.master_state_node:main',
            'slave_state_node = controller.slave_state_node:main',
            'simulated_master = controller.simulated_master_node:main',
            'joystick_teleop = controller.joystick_teleop_node:main',
            'keyboard_teleop = controller.keyboard_teleop_node:main',
        ],
    },
)
