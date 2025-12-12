# MSU Husky Mapping & Object Detection

This repository contains the ROS2 software stack for the MSU Husky robot, including LiDAR mapping and a custom **CPU-optimized** 3D object detection pipeline.

## Features

*   **LiDAR Mapping**: Integration with `lio_sam` and Ouster LiDAR.
*   **CPU Lidar Object Detection**: A custom-patched `OpenPCDet` implementation that runs real-time 3D object detection (PointPillars) **entirely on the CPU** without requiring CUDA/GPU.

## CPU Object Detection Node

Located in `lidar_object_detection`, this node performs 3D detection on Ouster LiDAR point clouds.

### Key Capabilities
*   **Model**: PointPillars (pretrained on KITTI).
*   **Hardware**: optimized for CPU execution.
*   **Input Topic**: `/ouster/points`
*   **Output Topic**: `/detections` (Visualization Markers)

### How to Run

1.  **Launch the Detection Node**:
    ```bash
    ros2 launch lidar_object_detection detection.launch.py
    ```

2.  **Visualization**:
    Open RViz2 and subscribe to `/detections` (MarkerArray).

### Troubleshooting
*   **Low FPS**: Expected on CPU (approx 1-3 FPS).
*   **False Positives**: The model is trained on outdoor street scenes; indoor performance may vary. Detections below 25% confidence are filtered out.
