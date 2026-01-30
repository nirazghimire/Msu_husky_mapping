# LiDAR Object Detection - Setup Guide

This guide explains how to run real-time **360° 3D object detection** on the Ouster LiDAR using a CPU-only laptop. The system uses a **NuScenes-trained PointPillars model** to detect 10 different classes of objects.

---

## 1. Overview

**Features:**
*   **360° Detection:** Detects objects all around the robot (unlike forward-only models).
*   **CPU Optimized:** Runs efficiently on a standard laptop CPU.
*   **10 Classes:** `car`, `truck`, `bus`, `trailer`, `construction_vehicle`, `pedestrian`, `motorcycle`, `bicycle`, `barrier`, `traffic_cone`.

**Input:** Point cloud from Ouster LiDAR (`/ouster/points`)  
**Output:** 3D bounding boxes as ROS2 markers (`/detections`)

---

## 2. Requirements

*   **OS:** Ubuntu 22.04 (ROS2 Humble)
*   **Hardware:** Ouster LiDAR (tested with OS-1) or Recorded PCAP data.
*   **Weights File:** `pp_multihead_nds5823_updated.pth` (Must be in `src/OpenPCDet_backup/pretrained_models/`)

---

## 3. Quick Start

### Step 1: Start the LiDAR Driver (or Replay)

**Option A: Live Sensor**
```bash
ros2 launch ouster_ros sensor.launch.xml sensor_hostname:=<YOUR_SENSOR_HOSTNAME> viz:=false
```

**Option B: Recorded Data (PCAP)**
```bash
ros2 launch ouster_ros replay_pcap.launch.xml metadata:=/path/to/data.json pcap_file:=/path/to/data.pcap viz:=false loop:=true
```

### Step 2: Start Object Detection

In a new terminal:
```bash
source install/setup.bash
ros2 launch lidar_object_detection detection.launch.py
```

You should see logs indicating the model loaded and is processing frames (e.g., `🔍 NuScenes Preds: 155 boxes`).

### Step 3: Visualization (RViz2)

1.  Run `rviz2`.
2.  **Global Options (Top Left):**
    *   **Fixed Frame:** Set to `os_sensor` (or `lidar_link`). *Do NOT use 'map'.*
3.  **Add Visualizations:**
    *   **PointCloud2:** Topic `/ouster/points` -> Color Transformer: `Intensity`.
    *   **MarkerArray:** Topic `/detections`.

---

## 4. Troubleshooting

### "Checkpoint file not found"
The model weights are not included in the git repo to save space. You must download `pp_multihead_nds5823_updated.pth` and place it in `src/OpenPCDet_backup/pretrained_models/`.

### "No executable found"
Run the build command again:
```bash
colcon build --packages-select lidar_object_detection
source install/setup.bash
```

### Blinking Markers
On CPU, inference can be slow (~2-3 FPS). We set the marker lifetime to 1.5 seconds to bridge the gap between frames. If they still blink, you may need to increase the lifetime in `detection_node.py` further.

### Wrong Coordinates / Boxes Underground
The model expects the LiDAR to be at `(0,0,0)`. If your `os_sensor` frame is defined differently in your URDF, you might see offsets. Ensure `Fixed Frame` is set to the sensor frame itself for debugging.

---

## 5. Technical Details

*   **Model Config:** `src/lidar_object_detection/config/nuscenes_pointpillar.yaml`
*   **Code:** `src/lidar_object_detection/lidar_object_detection/detection_node.py`
*   **Intensity:** Ouster intensity is normalized (divided by 255) to match NuScenes data distribution.
*   **NMS:** Uses a custom CPU-based Non-Maximum Suppression implementation to avoid CUDA dependencies.

---
*Last Updated: January 2026*
