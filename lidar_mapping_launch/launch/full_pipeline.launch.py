import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    fast_lio_pkg = get_package_share_directory('fast_lio')
    fast_lio_launch = os.path.join(fast_lio_pkg, 'launch', 'mapping.launch.py')
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    from launch_ros.actions import Node

    # Static transform to link FAST-LIO body frame to Ouster sensor frame
    # Since FAST-LIO's body frame is usually at the IMU, we link it here.
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'body', 'os_sensor']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(fast_lio_launch),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'config_file': 'ouster64.yaml'
            }.items()
        ),
        static_tf_node
    ])
