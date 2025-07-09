from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_path = get_package_share_directory('basic_mobile_robot')
    urdf_path = os.path.join(pkg_path, 'models/basic_mobile_bot_v1.urdf')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': open(urdf_path).read()}],
            output='screen'
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_path, 'rviz/urdf_config.rviz')],
            output='screen'
        ),
        Node(
            package= 'tf2_ros',
            executable= 'static_transform_publisher',
            arguments=['0', '0','0.1', '0', '0', 'base_link', 'lidar_link'],
            name = 'lidar_tf_publisher',
          
        )
    ])

