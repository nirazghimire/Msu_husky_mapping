# MSU Husky Mapping & Object Detection

This repository contains the ROS2 software stack for the MSU Husky robot, including LiDAR mapping and a custom **CPU-optimized** 3D object detection pipeline.

## Features

*   **LiDAR Mapping**: Integration with FAST_LIO and Ouster LiDAR for real-time visual-inertial mapping.
*   **CPU Lidar Object Detection**: A custom-patched `OpenPCDet` implementation that runs real-time 3D object detection (NuScenes PointPillars) **entirely on the CPU** without requiring CUDA/GPU.
*   **Production-Grade Visualization**: RViz2 configuration optimized for publication-quality screenshots with professional styling.

## LiDAR Mapping with FAST_LIO

Located in `src/FAST_LIO_ROS2/`, this package provides real-time visual-inertial odometry and mapping for Ouster LiDAR.

### Quick Start
```bash
# Launch Ouster sensor driver
ros2 launch ouster_ros sensor.launch.xml sensor_hostname:=<YOUR_SENSOR_IP> viz:=false

# In a new terminal, launch FAST_LIO mapping
ros2 launch lidar_mapping_launch full_pipeline.launch.py

# Visualize in RViz2
rviz2 -d install/fast_lio/share/fast_lio/rviz/fastlio.rviz
```

### Key Features
*   **Ouster Support**: Compatible with Ouster OS-1 and other LiDARs via PCL point struct registration.
*   **Real-Time Mapping**: Generates incremental point cloud maps with pose optimization.
*   **Publication-Ready Visualization**: Flat-color, dark-background RViz configuration with professional styling for research papers.
*   **Fixed Frame**: Uses `map` as global reference frame for consistent multi-robot coordinate systems.

### Recent Fixes (January 2026)
- **PCL Field Matching**: Fixed "Failed to find match for field 'ring'" warnings by aligning FAST_LIO's point struct definition (uint16_t) with Ouster SDK official definition.
- **RViz Configuration**: Updated `fastlio.rviz` with publication-quality settings:
  - Dark background (RGB: 10, 10, 10) for improved contrast
  - Flat gray point coloring (no rainbow artifacts)
  - Persistent point clouds (decay time = 0)
  - Optimized point size (0.04-0.05 m) for clarity in screenshots
  - Subtle grid overlay (RGB: 80, 80, 80)

### Saving the Map
To save the generated point cloud map (PCD file):
```bash
ros2 service call /map_save std_srvs/srv/Trigger
```
The map will be saved to `test.pcd` in the workspace directory (or the path specified in `config/ouster64.yaml`).

See [LIDAR_DETECTION_GUIDE.md](./LIDAR_DETECTION_GUIDE.md) for detailed mapping setup.

## CPU Object Detection Node

Located in `lidar_object_detection`, this node performs 3D detection on Ouster LiDAR point clouds.

### Key Capabilities
*   **Model**: PointPillars (pretrained on **NuScenes**).
*   **Classes**: 10 classes (Car, Truck, Bus, Pedestrian, Barrier, Traffic Cone, etc.).
*   **View**: 360° detection (unlike the previous front-view only model).
*   **Hardware**: optimized for CPU execution.
*   **Input Topic**: `/ouster/points`
*   **Output Topic**: `/detections` (Visualization Markers)

### How to Run

1.  **Launch the Detection Node**:
    ```bash
    ros2 launch lidar_object_detection detection.launch.py
    ```

2.  **Visualization**:
    Open RViz2:
    *   **Fixed Frame**: `os_sensor` (or `lidar_link`)
    *   **Add**: `MarkerArray` -> Topic: `/detections`
    *   **Add**: `PointCloud2` -> Topic: `/ouster/points` (Color Transformer: Intensity)

For detailed setup instructions, troubleshooting, and dependencies, please refer to **[LIDAR_DETECTION_GUIDE.md](./LIDAR_DETECTION_GUIDE.md)**.
