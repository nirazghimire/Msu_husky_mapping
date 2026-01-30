from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Path to FAST-LIO Mapping pipeline
    mapping_pkg = get_package_share_directory('lidar_mapping_launch')
    mapping_launch = os.path.join(mapping_pkg, 'launch', 'full_pipeline.launch.py')

    # Path to Object Detection pipeline
    detection_pkg = get_package_share_directory('lidar_object_detection')
    detection_launch = os.path.join(detection_pkg, 'launch', 'detection.launch.py')

    # 1. Start Mapping (FAST-LIO) immediately
    # This sets up TF, Map, and RViz
    mapping_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mapping_launch),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 2. Start Detection (PointPillars) with a small delay
    # Delay ensures TF tree is stable before detector starts logic
    detection_node = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(detection_launch),
                launch_arguments={'use_sim_time': use_sim_time}.items()
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        mapping_node,
        detection_node
    ])
