from __future__ import annotations

from functools import partial
from typing import Dict, List

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from .pointcloud_utils import create_xyz_cloud, crop_points, extract_xyz_points, voxel_downsample
from .transform_utils import build_sensor_transform_map, transform_points


class CloudTransformMergeNode(Node):
    def __init__(self) -> None:
        super().__init__('cloud_transform_merge_node')

        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('voxel_size', 0.08)
        self.declare_parameter('merged_voxel_size', 0.08)
        self.declare_parameter('stale_cloud_timeout_sec', 1.5)
        self.declare_parameter('roi_min', [-12.0, -7.0, 0.0])
        self.declare_parameter('roi_max', [12.0, 7.0, 4.5])
        self.declare_parameter(
            'lidar_topics',
            [
                '/lidar_01/points',
                '/lidar_02/points',
                '/lidar_03/points',
                '/lidar_04/points',
                '/lidar_05/points',
                '/lidar_06/points',
            ],
        )

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.merged_voxel_size = float(self.get_parameter('merged_voxel_size').value)
        self.stale_cloud_timeout_sec = float(self.get_parameter('stale_cloud_timeout_sec').value)
        self.roi_min = np.asarray(self.get_parameter('roi_min').value, dtype=np.float32)
        self.roi_max = np.asarray(self.get_parameter('roi_max').value, dtype=np.float32)
        self.lidar_topics: List[str] = list(self.get_parameter('lidar_topics').value)

        self.sensor_transform_map = build_sensor_transform_map()
        self.latest_points_by_topic: Dict[str, np.ndarray] = {}
        self.latest_stamp_by_topic: Dict[str, float] = {}
        self.latest_msg_stamp = None

        self.merged_cloud_pub = self.create_publisher(PointCloud2, '/tracking/merged_cloud', 10)

        self.subscribers = [
            self.create_subscription(PointCloud2, topic, partial(self.cloud_callback, topic), 10)
            for topic in self.lidar_topics
        ]

        publish_period = 1.0 / max(self.publish_rate_hz, 0.1)
        self.timer = self.create_timer(publish_period, self.publish_merged_cloud)

        self.get_logger().info('cloud_transform_merge_node started.')
        self.get_logger().info(f'Target frame: {self.target_frame}')

    def cloud_callback(self, topic_name: str, cloud_msg: PointCloud2) -> None:
        frame_id = cloud_msg.header.frame_id
        transform_matrix = self.sensor_transform_map.get(frame_id)
        if transform_matrix is None:
            self.get_logger().warning(
                f'No static transform configured for frame "{frame_id}". '
                'Known frames: ' + ', '.join(sorted(self.sensor_transform_map.keys())),
                throttle_duration_sec=5.0,
            )
            return

        points_xyz = extract_xyz_points(cloud_msg)
        if points_xyz.size == 0:
            self.latest_points_by_topic[topic_name] = np.empty((0, 3), dtype=np.float32)
            self.latest_stamp_by_topic[topic_name] = self.msg_time_to_sec(cloud_msg)
            self.latest_msg_stamp = cloud_msg.header.stamp
            return

        world_points = transform_points(points_xyz, transform_matrix)
        cropped_points = crop_points(world_points, self.roi_min, self.roi_max)
        downsampled_points = voxel_downsample(cropped_points, self.voxel_size)

        self.latest_points_by_topic[topic_name] = downsampled_points
        self.latest_stamp_by_topic[topic_name] = self.msg_time_to_sec(cloud_msg)
        self.latest_msg_stamp = cloud_msg.header.stamp

    def publish_merged_cloud(self) -> None:
        if self.latest_msg_stamp is None:
            return

        if not self.latest_stamp_by_topic:
            return

        reference_time_sec = max(self.latest_stamp_by_topic.values())
        merged_parts = []

        for topic_name, points_xyz in self.latest_points_by_topic.items():
            msg_stamp_sec = self.latest_stamp_by_topic.get(topic_name)
            if msg_stamp_sec is None:
                continue
            if (reference_time_sec - msg_stamp_sec) > self.stale_cloud_timeout_sec:
                continue
            if points_xyz.size == 0:
                continue
            merged_parts.append(points_xyz)

        if not merged_parts:
            return

        merged_points = np.concatenate(merged_parts, axis=0)
        merged_points = voxel_downsample(merged_points, self.merged_voxel_size)
        merged_msg = create_xyz_cloud(self.target_frame, self.latest_msg_stamp, merged_points)
        self.merged_cloud_pub.publish(merged_msg)

    @staticmethod
    def msg_time_to_sec(cloud_msg: PointCloud2) -> float:
        stamp = cloud_msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CloudTransformMergeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()