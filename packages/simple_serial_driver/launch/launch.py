import os

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription

from launch.substitutions import Command
def generate_launch_description():
    robot_description = Command([
        "xacro",
        " ",
        f"{get_package_share_directory("simple_serial_driver")}/urdf/robot.urdf.xacro"
    ])

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": robot_description,
                "publish_frequency": 500.
            }]
        ),
        Node(
            package="simple_serial_driver",
            executable="simple_serial_driver_node",
            name="simple_serial_driver",
            parameters=[{
                "device_name": "/dev/ttyACM0",
                "baud_rate": 2000000,
                "timestamp_offset": 0.,
                "robot_name": "rm_robot"

            }]
        )
    ])

