from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    config = PathJoinSubstitution([
        FindPackageShare("image_pipeline"),
        "config",
        LaunchConfiguration("config", default="default.yaml")
    ])

    return LaunchDescription([
        Node(
            package="image_pipeline",
            executable="detector",
            name="detector",
            parameters=[config]
        ),

        Node(
            package="image_pipeline",
            executable="tracker",
            name="tracker",
            parameters=[config]
        )
    ])
