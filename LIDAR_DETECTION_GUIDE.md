# LiDAR Object Detection - Complete Setup Guide

This guide explains how to run real-time 3D object detection on the Ouster LiDAR using a CPU-only laptop. The system uses PointPillars (an AI model) to detect cars, pedestrians, and cyclists.

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Requirements](#software-requirements)
4. [Quick Start](#quick-start)
5. [How It Works](#how-it-works)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Known Limitations](#known-limitations)

---

## Overview

This package detects objects in 3D LiDAR point clouds in real-time. It works without a GPU (runs on CPU only).

**What it detects:**
- 🚗 Cars
- 🚶 Pedestrians
- 🚴 Cyclists

**Input:** Point cloud from Ouster LiDAR (`/ouster/points`)  
**Output:** 3D bounding boxes as ROS2 markers (`/detections`)

---

## Hardware Requirements

- **Ouster LiDAR** (tested with OS-1-64)
- **Laptop/PC** with at least 8GB RAM
- **Network connection** to the LiDAR (Ethernet)

---

## Software Requirements

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- PyTorch (CPU version)
- OpenPCDet (patched for CPU)

All dependencies should already be installed in the `venv_ros2` virtual environment.

---

## Quick Start

### Step 1: Open 3 Terminals

You need 3 terminal windows. In each terminal, first activate the environment:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### Step 2: Start the Ouster Driver (Terminal 1)

This connects to the physical LiDAR and publishes point clouds:

```bash
ros2 launch ouster_ros sensor.launch.xml sensor_hostname:=os-122512000887.local viz:=false
```

**Note:** Replace `os-122512000887.local` with your LiDAR's hostname if different.

Wait until you see: `Sensor configured successfully`

### Step 3: Start the Detection Node (Terminal 2)

This runs the AI model and publishes detections:

```bash
ros2 launch lidar_object_detection detection.launch.py
```

You should see:
```
✅ OpenPCDet model loaded on CPU
📊 Points: 50000 total, 26000 in detect range | X:[-7.0,3.0] ...
🔍 Raw predictions: 2 boxes
Published 2 detections (Threshold: 0.1)
```

### Step 4: Visualize in RViz2 (Terminal 3)

```bash
rviz2
```

In RViz2:
1. Set **Fixed Frame** to `os_lidar`
2. Click **Add** → **PointCloud2** → Topic: `/ouster/points`
3. Click **Add** → **MarkerArray** → Topic: `/detections`

You should now see the point cloud and bounding boxes!

---

## How It Works

### Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  Ouster LiDAR   │────▶│  lidar_detection_node │────▶│   RViz2     │
│  (Hardware)     │     │  (PointPillars AI)    │     │   (Display) │
└─────────────────┘     └──────────────────────┘     └─────────────┘
        │                         │
        ▼                         ▼
  /ouster/points            /detections
  (PointCloud2)             (MarkerArray)
```

### Key Files

| File | Purpose |
|------|---------|
| `src/lidar_object_detection/` | Main detection package |
| `src/lidar_object_detection/lidar_object_detection/detection_node.py` | Detection node code |
| `src/lidar_object_detection/config/pointpillar.yaml` | Model configuration |
| `src/OpenPCDet_backup/` | Patched OpenPCDet (CPU-only) |
| `src/OpenPCDet_backup/pretrained_models/pointpillar_7728.pth` | Pretrained model weights |

### Coordinate Transformation

The PointPillars model was trained on KITTI dataset (forward-facing car camera). It expects objects in front of the sensor (positive X direction). 

However, the Ouster LiDAR sees 360° around itself, with points in both positive and negative X.

To make detection work, the code:
1. Shifts all points by +35 meters on X axis (into KITTI range)
2. Runs AI inference
3. Shifts detection boxes back by -35 meters (to original positions)

This is handled automatically in `detection_node.py`.

---

## Configuration

### Detection Threshold

To reduce false positives, you can adjust the detection threshold.

In `detection_node.py`, find this line:
```python
DETECTION_THRESHOLD = 0.1  # Lowered from 0.25 for debugging
```

Change to a higher value (e.g., 0.25 or 0.3) for fewer but more confident detections.

### Point Cloud Range

The detection only considers points within this range:
```python
POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]
# [x_min, y_min, z_min, x_max, y_max, z_max]
```

After the X offset is applied, this becomes approximately:
- X: -35 to +34 meters (from sensor)
- Y: -40 to +40 meters
- Z: -3 to +1 meters (height)

---

## Troubleshooting

### Problem: No point cloud in RViz2

**Check:** Is the Ouster driver running?
```bash
ros2 topic list | grep ouster
```
Should show `/ouster/points`

**Check:** Is the Fixed Frame correct?
Set it to `os_lidar` in RViz2

### Problem: 0 detections / 0 raw predictions

**Check 1:** Are points being received?
Look for: `📊 Points: X total, Y in detect range`
- If X is 0, the detection node isn't receiving data
- If Y is very low, most points are outside detection range

**Check 2:** QoS compatibility
The detection node uses BEST_EFFORT QoS. If you see warnings about "incompatible QoS", there's a mismatch.

### Problem: Detection boxes appear in wrong location

**Check:** Make sure the X_OFFSET transformation is applied both:
1. Before voxelization (shift points)
2. After detection (shift boxes back)

### Problem: Model won't load / CUDA errors

The code is patched to run on CPU. If you see CUDA errors, check that:
- `src/OpenPCDet_backup/` is being used (not `src/OpenPCDet`)
- The `to_cpu=True` flag is set when loading model weights

---

## Known Limitations

### 1. Only Detects Cars, Pedestrians, Cyclists

The model was trained on KITTI (outdoor driving data). It will NOT detect:
- Furniture
- Animals
- Drones
- Indoor objects

### 2. False Positives Indoors

Indoor structures (walls, desks, shelves) may be detected as "Car" because they have similar shapes.

### 3. Slow on CPU

Detection runs at ~0.5 FPS on CPU. This is expected. For real-time performance, you would need a GPU.

### 4. Limited Detection Range

Objects beyond ~35 meters may not be detected well.

### 5. Height Cutoff

Objects taller than ~1 meter above the sensor may be partially cut off due to Z range limits.

---

## Testing with Recorded Data

If you don't have the physical LiDAR available, you can test with a recorded rosbag:

```bash
# Terminal 1: Detection node
ros2 launch lidar_object_detection detection.launch.py

# Terminal 2: Play rosbag
ros2 bag play ~/ros2_ws/rosbag2_2025_08_15-14_03_09 --clock
```

---

## Building After Code Changes

If you modify the Python code:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select lidar_object_detection --symlink-install
source install/setup.bash
```

---

## Contact & References

- **OpenPCDet:** https://github.com/open-mmlab/OpenPCDet
- **PointPillars Paper:** https://arxiv.org/abs/1812.05784
- **KITTI Dataset:** http://www.cvlibs.net/datasets/kitti/

---

*Last Updated: January 2026*
