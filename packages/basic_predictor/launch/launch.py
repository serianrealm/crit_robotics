from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="basic_predictor",
            executable="basic_predictor",
        )
    ])
# target_include_directories(${PROJECT_NAME} PU