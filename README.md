# MSU Husky: Autonomous Perception & Mapping Stack

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg) ![Platform](https://img.shields.io/badge/Platform-Clearpath%20Husky-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This repository hosts the specialized autonomy software for the **MSU Husky** research platform. It features a high-performance, **CPU-optimized** perception stack capable of real-time 360° 3D object detection and simultaneous visual-inertial mapping (SLAM) using a single Ouster LiDAR sensor.

Designed for robust operation in unstructured environments without requiring heavy GPU compute.

## Core Capabilities

### CPU-Optimized 3D Object Detection
- **Model**: Custom PointPillars architecture trained on NuScenes.
- **Hardware**: Runs entirely on standard i7/i9 CPUs (No GPU required).
- **Performance**: Real-time 360° detection of 10+ classes (Cars, Pedestrians, Trucks, etc.).
- **Integration**: Seamless ROS2 `MarkerArray` output for navigation stacks.

### Real-Time Mapping (SLAM)
- **Engine**: FAST_LIO (Fast LiDAR-Inertial Odometry).
- **Accuracy**: High-fidelity dense point cloud generation with drift correction.
- **Robustness**: Handles aggressive motion and unstructured terrain.

## Getting Started

We have prepared a comprehensive, step-by-step guide for new users to set up the environment, install dependencies, and run the full stack from scratch.

**[READ THE SETUP GUIDE HERE](./LIDAR_DETECTION_GUIDE.md)**

## Quick Commands

**Launch Object Detection:**
```bash
ros2 launch lidar_object_detection detection.launch.py
```

**Launch Mapping:**
```bash
ros2 launch lidar_mapping_launch full_pipeline.launch.py
```

**Save Map:**
```bash
ros2 service call /map_save std_srvs/srv/Trigger
```

## Repository Structure

- `lidar_object_detection/`: The core CPU detection node.
- `lidar_mapping_launch/`: Unified launch files for mapping pipelines.
- `FAST_LIO_ROS2/`: The underlying SLAM algorithm implementation.
- `OpenPCDet_backup/`: The inference engine patches and utilities.

---
*Developed by the MSU Autonomy Research Team | 2026*
