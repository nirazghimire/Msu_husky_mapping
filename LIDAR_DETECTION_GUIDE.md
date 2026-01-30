# LiDAR Detection & Mapping - Complete Setup Guide

This comprehensive guide will walk you through setting up the **MSU Husky** software stack from scratch. By the end, you will be able to run **360° 3D Object Detection** and **Real-Time Mapping** on the Husky robot (or your local machine).

---

## 1. Prerequisites

Before you begin, ensure you have the following:

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish).
- **ROS Distribution**: ROS2 Humble Hawksbill.
- **Hardware**:
    - Laptop/PC (No GPU required, but AVX2 support recommended for performance).
    - Ouster LiDAR (OS-1-64 recommended) *OR* ensure you have recorded PCAP data.
- **Tools**: `git`, `python3-pip`, `wget`.

### Install ROS2 Humble (If not installed)
If you haven't installed ROS2 Humble yet, follow the [official guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) or run:
```bash
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools
```

---

## 2. Workspace Setup

### 1. Clone the Repository
Open a terminal and create your workspace:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/nirazghimire/Msu_husky_mapping.git .
```

### 2. Install Dependencies
Install the required system, Python, and ROS dependencies:

**System Requirements:**
```bash
sudo apt update
sudo apt install -y python3-pip ros-humble-desktop ros-humble-pcl-ros ros-humble-perception-pcl ros-humble-gtsam
```

**Python Requirements (for Object Detection):**
The object detection pipeline uses a custom `OpenPCDet` implementation.
```bash
# Core torch libraries (CPU version)
pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install project-specific requirements
cd ~/ros2_ws/src
pip3 install -r OpenPCDet_backup/requirements.txt
```

> **Note on spconv**: The `requirements.txt` should install the compatible `spconv` version. If you face issues, `pip3 install spconv-cu102` often works for CPU-only inference on compatible architectures.

---

## 3. Model Weights Setup (Critical)

The object detection model weights are **NOT** included in the repository due to size limits. You must download them manually.

1.  **Download** the weights file: `pp_multihead_nds5823_updated.pth`
    *   (Ask your team lead for the download link if not provided separately).
2.  **Move** the file to the correct directory:
    ```bash
    mv /path/to/downloaded/pp_multihead_nds5823_updated.pth ~/ros2_ws/src/OpenPCDet_backup/pretrained_models/
    ```

**Verification**:
Ensure the file exists:
```bash
ls ~/ros2_ws/src/OpenPCDet_backup/pretrained_models/
# Output should show: pp_multihead_nds5823_updated.pth
```

---

## 4. Building the Workspace

Build the entire software stack using `colcon`.

```bash
cd ~/ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

*This may take a few minutes.*

Once finished, source the workspace:
```bash
source install/setup.bash
# Tip: Add this to your ~/.bashrc to auto-source it
# echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## 5. Running the System

You can run the modules individually or together.

### A. Start the LiDAR Sensor
First, we need live data from the Ouster.

```bash
# Replace <SENSOR_IP> with your LiDAR's IP (e.g., 192.168.1.100)
ros2 launch ouster_ros sensor.launch.xml sensor_hostname:=<SENSOR_IP> viz:=false
```
*If using recorded data, launch your replay script or play the rosbag.*

### B. Run Object Detection (The "Eyes")
This node will detect objects (Cars, Pedestrians, etc.) in 3D space.

**In a new terminal:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch lidar_object_detection detection.launch.py
```
*Wait until you see logs say: `🔍 NuScenes Preds: ...`*

### C. Run Mapping (The "Memory")
This node builds a consistent map of the environment as you move.

**In a new terminal:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch lidar_mapping_launch full_pipeline.launch.py
```

---

## 6. Visualization (Seeing it all)

We have a pre-configured RViz setup to see everything at once.

1.  **Open RViz2:**
    ```bash
    rviz2
    ```
2.  **Load Config:**
    *   File -> Open Config -> Navigate to `src/FAST_LIO_ROS2/config/fastlio.rviz` (or manually configure as below).

**Manual Configuration Checklist:**
*   **Fixed Frame**: `os_sensor` (for live view) or `map` (if mapping).
*   **PointCloud2**: Topic `/ouster/points` -> Style: Points -> Color: Intensity.
*   **MarkerArray**: Topic `/detections` -> Show all boxes.
*   **PointCloud2 (Map)**: Topic `/cloud_registered` -> The built map.

---

## 7. Saving Data

### Saving the Map
To save the built map to a `.pcd` file:
```bash
ros2 service call /map_save std_srvs/srv/Trigger
```
The file `test.pcd` will be saved in your workspace root or the configured output directory.

---

## Troubleshooting

- **"ModuleNotFoundError: No module named 'pcdet'"**: You forgot to source the workspace (`source install/setup.bash`).
- **"Checkpoint not found"**: You skipped **Section 3**. The model weights are missing.
- **LiDAR Timestamps**: If detection boxes are "lagging" or not appearing, check time synchronization. Ensure PTP is active if using a real sensor.
