#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import numpy as np
import struct
import torch
import spconv.pytorch.utils as sputils

import sys
import os

# Add OpenPCDet to sys.path if not installed as a package
pcdet_path = '/home/thispc/ros2_ws/src/OpenPCDet_backup'
if pcdet_path not in sys.path:
    sys.path.append(pcdet_path)

from pcdet.models import build_network
from pcdet.config import cfg, cfg_from_yaml_file

def yaw_to_quaternion(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

def pointcloud2_to_array(cloud_msg):
    fmt = 'ffff'
    points = []
    for i in range(0, len(cloud_msg.data), cloud_msg.point_step):
        x, y, z, intensity = struct.unpack_from(fmt, cloud_msg.data, i)
        points.append([x, y, z, intensity])
    return np.array(points, dtype=np.float32)

class DummyDataset:
    def __init__(self, class_names):
        self.class_names = class_names
        self.point_cloud_range = np.array([0, -39.68, -3, 69.12, 39.68, 1])
        self.voxel_size = np.array([0.16, 0.16, 4])
        self.grid_size = np.array([432, 496, 1])
        self.depth_downsample_factor = None
        self.point_feature_encoder = type('FeatureEncoder', (object,), {'num_point_features': 4})() # Mock

class LidarDetectionNode(Node):
    def __init__(self):
        super().__init__('lidar_detection_node')

        cfg_file = '/home/thispc/ros2_ws/src/lidar_object_detection/config/pointpillar.yaml'
        ckpt_file = '/home/thispc/ros2_ws/src/OpenPCDet_backup/pretrained_models/pointpillar_7728.pth'

        cfg_from_yaml_file(cfg_file, cfg)
        self.device = torch.device('cpu')
        
        dataset = DummyDataset(cfg.CLASS_NAMES)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        self.model.load_params_from_file(filename=ckpt_file, logger=self.get_logger(), to_cpu=True)
        self.model.to(self.device).eval()

        # Voxel Generator
        self.voxel_generator = sputils.PointToVoxel(
            vsize_xyz=[0.16, 0.16, 4],
            coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
            num_point_features=4,
            max_num_voxels=40000,
            max_num_points_per_voxel=32,
            device=self.device
        )

        self.get_logger().info("✅ OpenPCDet model loaded on CPU")

        # Use SensorDataQoS (BEST_EFFORT) to match typical LiDAR/bag publisher policies
        self.subscription = self.create_subscription(
            PointCloud2, 
            '/ouster/points', 
            self.pointcloud_callback, 
            qos_profile=qos_profile_sensor_data
        )
        self.publisher = self.create_publisher(MarkerArray, '/detections', 10)

    def pointcloud_callback(self, msg):
        points = pointcloud2_to_array(msg)
        if points.shape[0] == 0:
            return

        # Voxelization
        points_tensor = torch.from_numpy(points).float().to(self.device)
        voxels, coords, num_points = self.voxel_generator(points_tensor)

        # Prepare input dict
        batch_idx = torch.zeros((coords.shape[0], 1), device=self.device, dtype=torch.int32)
        coords_batch = torch.cat((batch_idx, coords), dim=1)

        input_dict = {
            'voxels': voxels,
            'voxel_num_points': num_points,
            'voxel_coords': coords_batch,
            'batch_size': 1
        }

        with torch.no_grad():
            pred_dicts, _ = self.model.forward(input_dict)

        marker_array = MarkerArray()
        marker_id = 0

        for box, cls, score in zip(pred_dicts[0]['pred_boxes'],
                                   pred_dicts[0]['pred_labels'],
                                   pred_dicts[0]['pred_scores']):
            if score < 0.25:
                continue

            # Cube Marker
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = msg.header.stamp
            marker.ns = "lidar_boxes"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            x, y, z, dx, dy, dz, heading = box.cpu().numpy()
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)

            q = yaw_to_quaternion(heading)
            marker.pose.orientation.x = float(q[1])
            marker.pose.orientation.y = float(q[2])
            marker.pose.orientation.z = float(q[3])
            marker.pose.orientation.w = float(q[0])

            marker.scale.x = float(dx)
            marker.scale.y = float(dy)
            marker.scale.z = float(dz)

            if cls == 1:
                marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 0.0)
            elif cls == 2:
                marker.color.r, marker.color.g, marker.color.b = (0.0, 1.0, 0.0)
            elif cls == 3:
                marker.color.r, marker.color.g, marker.color.b = (0.0, 0.0, 1.0)
            marker.color.a = 0.5
            marker.lifetime = Duration(sec=1, nanosec=500_000_000) 
            marker_array.markers.append(marker)

            # Text Marker
            text_marker = Marker()
            text_marker.header.frame_id = msg.header.frame_id
            text_marker.header.stamp = msg.header.stamp
            text_marker.ns = "lidar_text"
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(x)
            text_marker.pose.position.y = float(y)
            text_marker.pose.position.z = float(z) + float(dz)/2 + 0.5 
            
            class_name = self.model.class_names[cls - 1]
            text_marker.text = f"{class_name}: {score:.2f}"
            
            text_marker.scale.z = 0.5 # Text height
            text_marker.color.r, text_marker.color.g, text_marker.color.b = (1.0, 1.0, 1.0)
            text_marker.color.a = 1.0
            text_marker.lifetime = Duration(sec=1, nanosec=500_000_000)
            marker_array.markers.append(text_marker)

        self.get_logger().info(f"Published {len(marker_array.markers)//2} detections (Threshold: 0.1)")
        self.publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
