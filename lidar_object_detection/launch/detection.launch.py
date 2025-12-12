from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_object_detection',
            executable='lidar_detection_node',
            name='lidar_detection_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                # Add parameters here if needed
            ]
        )
    ])
