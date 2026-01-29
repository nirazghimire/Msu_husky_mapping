# MSU Husky Mapping & Object Detection

This repository contains the ROS2 software stack for the MSU Husky robot, including LiDAR mapping and a custom **CPU-optimized** 3D object detection pipeline.

## Features

*   **LiDAR Mapping**: Integration with `lio_sam` and Ouster LiDAR.
*   **CPU Lidar Object Detection**: A custom-patched `OpenPCDet` implementation that runs real-time 3D object detection (NuScenes PointPillars) **entirely on the CPU** without requiring CUDA/GPU.

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
    ros2 run lidar_object_detection lidar_detection_node
    ```

2.  **Visualization**:
    Open RViz2:
    *   **Fixed Frame**: `os_sensor` (or `lidar_link`)
    *   **Add**: `MarkerArray` -> Topic: `/detections`
    *   **Add**: `PointCloud2` -> Topic: `/ouster/points` (Color Transformer: Intensity)

For detailed setup instructions, troubleshooting, and dependencies, please refer to **[LIDAR_DETECTION_GUIDE.md](./LIDAR_DETECTION_GUIDE.md)**.
