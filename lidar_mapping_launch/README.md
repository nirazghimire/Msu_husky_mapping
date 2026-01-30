# LiDAR Mapping Launch Package

## Overview
This package provides a streamlined launch interface for **FAST_LIO** (Fast LiDAR-Inertial Odometry) on the Husky A200 platform. It utilizes the Ouster OS1-64 LiDAR and its internal IMU to generate high-fidelity 3D point cloud maps in real-time.

## Architecture
- **LiDAR Source**: Ouster OS1-64 (`/ouster/points`)
- **IMU Source**: Ouster Internal IMU (`/ouster/imu`)
- **SLAM Backend**: FAST_LIO (LiDAR-Inertial Odometry)
- **Visualization**: RViz2 (Pre-configured)

## Prerequisites
- ROS2 Humble
- Ouster ROS2 Driver
- FAST_LIO_ROS2

## Usage

### 1. Start Sensors
Ensure your Ouster LiDAR is connected and running:
```bash
ros2 launch ouster_ros sensor.launch.xml sensor_hostname:=<your-sensor-ip> viz:=false
```

### 2. Launch Mapping Pipeline
Start the FAST_LIO mapping system:
```bash
ros2 launch lidar_mapping_launch full_pipeline.launch.py
```

This will automatically:
- Start the FAST_LIO mapping node
- Publish necessary static transforms (`body` -> `os_sensor`)
- Open RViz2 is configured for mapping visualization

### 3. Save Map
To save the generated map as a PCD file:
```bash
ros2 service call /map_save std_srvs/srv/Trigger
```
The map will be saved as `test.pcd` in the `FAST_LIO_ROS2` workspace directory (or the specific path defined in `config/ouster64.yaml`).

## Configuration
- **Scanning Parameters**: `FAST_LIO_ROS2/config/ouster64.yaml`
- **Launch Logic**: `launch/full_pipeline.launch.py`

## Troubleshooting
- **No Map**: Check `/ouster/points` and `/ouster/imu` topics are active.
- **Drift**: Verify extrinsic transformations in `ouster64.yaml` if sensor mounting changes.
