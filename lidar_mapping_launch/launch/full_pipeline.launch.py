from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess

def generate_launch_description():
    
    return LaunchDescription([
    # Static transform for LIDAR
    Node(
    	package = 'tf2_ros',
    	executable = 'static_transform_publisher'
    	arguments = ['0.08', '0.0', '0.38', '0', '0', '0', 'base_link', 'os_sensor'] # this is measurement i took from my phone.
    	name = 'lidar_tf_publisher'
    ),
    
    # Static transform for IMU
    Node(
    	package = 'tf2_ros',
    	executable = 'static_transform_publisher',
    	arguments = ['0.05', '0', '0.1', '0', '0', '0','base_link','imu_link'], #this is the rough measurement.
    	name = 'imu_tf_publisher'
    ),
    
    
    Node(
	    package = 'lio_sam',
	    executable = 'lio_sam_node',
	    name = 'lio_sam',
	    output = 'screen',
	    parameters = [
	    	os.path.join(
	    		get_package_share_directory('lio_sam'),
	    		'config','params.yaml'),
	    ],
	    remappings = [
	    ('imu_raw', '/imu/data'),
	    ('points_raw','/ouster/points'),
	    ]
    ),
    
    #recording the data
    ExecuteProcess(
    	cmd = [
    		'ros2','bag','record',
    		'-o', '/home/thispc/ros2_ws/src/lidar_mapping_launch/launch/Recorded_data/',
    		'/ouster/points',
    		'/imu/data/',
    		'/tf','/tf_static',
    		'/lio_sam/mapping/odometry'
    		],
    	output = 'screen',
    	name = 'data_recorder'
    ),

    # Rviz Node to live visualization:
    Node(
        package = 'rviz2',
        executable='rviz2',
        name = 'rviz2',
        arguments = ['-d','/home/recorded_data.rviz'],
        output = 'screen',
    ),
    Node(
        package = 'smartmicro_radar_driver',
        executable = 'smartmicro_radar_node',
        name = 'radar_driver',
        output = 'screen',
        parameters = [
            os.path.join(get_package_share_directory('smartmicro_radar_driver'),
                         'config','radar_params.yaml')
		]
	),

    # this is for lidar
    # Add sensor fusion node (custom)
    Node(
        package = 'sensor_fusion',
        executable = 'radar_lidar_fusion_node',
        name = 'sensor_fusion',
        output = 'screen',
        parameters = []
	)

    ])
    
    
    
    # to save the map after mapping:
    #run this command:
# ros2 service call /lio_sam/save_map/srv/SaveMap "{resolution: 0.2, destination: '/home/ros2_ws/src/lidar_mapping_launch/launch/SLAM_Maps/'}"
