from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .pointcloud_utils import create_xyz_cloud, extract_xyz_points, points_to_voxel_keys, voxel_keys_to_centers
from .tracking_types import ClusterInfo

VoxelKey = Tuple[int, int, int]


class ClusterDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('cluster_detector_node')

        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('voxel_size', 0.12)
        self.declare_parameter('min_cluster_voxels', 8)
        self.declare_parameter('max_cluster_voxels', 5000)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.min_cluster_voxels = int(self.get_parameter('min_cluster_voxels').value)
        self.max_cluster_voxels = int(self.get_parameter('max_cluster_voxels').value)

        self.background_keys: Set[VoxelKey] = set()
        self.last_points = np.empty((0, 3), dtype=np.float32)
        self.last_stamp = None

        self.dynamic_cloud_pub = self.create_publisher(PointCloud2, '/tracking/dynamic_cloud', 10)
        self.cluster_pose_pub = self.create_publisher(PoseArray, '/tracking/cluster_centroids', 10)
        self.cluster_marker_pub = self.create_publisher(MarkerArray, '/tracking/cluster_markers', 10)

        self.cloud_sub = self.create_subscription(PointCloud2, '/tracking/merged_cloud', self.cloud_callback, 10)

        self.capture_background_srv = self.create_service(Trigger, '/tracking/capture_background', self.capture_background)
        self.clear_background_srv = self.create_service(Trigger, '/tracking/clear_background', self.clear_background)

        self.neighbor_offsets = [
            offset for offset in product([-1, 0, 1], repeat=3)
            if offset != (0, 0, 0)
        ]

        self.get_logger().info('cluster_detector_node started.')
        self.get_logger().info('Call /tracking/capture_background while the hall is empty.')

    def cloud_callback(self, cloud_msg: PointCloud2) -> None:
        self.last_points = extract_xyz_points(cloud_msg)
        self.last_stamp = cloud_msg.header.stamp

        if not self.background_keys:
            self.publish_empty_outputs(cloud_msg.header.stamp)
            self.get_logger().warning(
                'Background not captured yet. Call /tracking/capture_background.',
                throttle_duration_sec=5.0,
            )
            return

        current_keys = points_to_voxel_keys(self.last_points, self.voxel_size)
        dynamic_keys = current_keys - self.background_keys
        dynamic_points = voxel_keys_to_centers(dynamic_keys, self.voxel_size)
        self.dynamic_cloud_pub.publish(create_xyz_cloud(self.target_frame, cloud_msg.header.stamp, dynamic_points))

        clusters = self.cluster_voxels(dynamic_keys)
        cluster_infos = self.build_cluster_infos(clusters)
        self.publish_cluster_outputs(cluster_infos, cloud_msg.header.stamp)

    def capture_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.last_points.size == 0 or self.last_stamp is None:
            response.success = False
            response.message = 'No merged cloud received yet.'
            return response

        self.background_keys = points_to_voxel_keys(self.last_points, self.voxel_size)
        response.success = True
        response.message = f'Background captured with {len(self.background_keys)} occupied voxels.'
        self.get_logger().info(response.message)
        return response

    def clear_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.background_keys.clear()
        response.success = True
        response.message = 'Background cleared.'
        self.get_logger().info(response.message)
        return response

    def cluster_voxels(self, voxel_keys: Set[VoxelKey]) -> List[List[VoxelKey]]:
        clusters: List[List[VoxelKey]] = []
        remaining = set(voxel_keys)

        while remaining:
            seed = remaining.pop()
            queue = [seed]
            cluster = [seed]

            while queue:
                key = queue.pop()
                for offset in self.neighbor_offsets:
                    neighbor = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        cluster.append(neighbor)

            if self.min_cluster_voxels <= len(cluster) <= self.max_cluster_voxels:
                clusters.append(cluster)

        return clusters

    def build_cluster_infos(self, clusters: Iterable[List[VoxelKey]]) -> List[ClusterInfo]:
        cluster_infos: List[ClusterInfo] = []

        for cluster_id, cluster in enumerate(clusters):
            cluster_points = voxel_keys_to_centers(cluster, self.voxel_size)
            centroid = np.mean(cluster_points, axis=0)
            min_bound = np.min(cluster_points, axis=0)
            max_bound = np.max(cluster_points, axis=0)
            cluster_infos.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    centroid=centroid.astype(np.float32),
                    min_bound=min_bound.astype(np.float32),
                    max_bound=max_bound.astype(np.float32),
                    voxel_count=len(cluster),
                )
            )

        return cluster_infos

    def publish_cluster_outputs(self, cluster_infos: List[ClusterInfo], stamp) -> None:
        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = stamp

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for cluster in cluster_infos:
            pose = Pose()
            pose.position.x = float(cluster.centroid[0])
            pose.position.y = float(cluster.centroid[1])
            pose.position.z = float(cluster.centroid[2])
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            bbox = Marker()
            bbox.header.frame_id = self.target_frame
            bbox.header.stamp = stamp
            bbox.ns = 'clusters'
            bbox.id = marker_id
            bbox.type = Marker.CUBE
            bbox.action = Marker.ADD
            bbox.pose.position.x = float((cluster.min_bound[0] + cluster.max_bound[0]) * 0.5)
            bbox.pose.position.y = float((cluster.min_bound[1] + cluster.max_bound[1]) * 0.5)
            bbox.pose.position.z = float((cluster.min_bound[2] + cluster.max_bound[2]) * 0.5)
            bbox.pose.orientation.w = 1.0
            bbox.scale.x = max(float(cluster.max_bound[0] - cluster.min_bound[0]), self.voxel_size)
            bbox.scale.y = max(float(cluster.max_bound[1] - cluster.min_bound[1]), self.voxel_size)
            bbox.scale.z = max(float(cluster.max_bound[2] - cluster.min_bound[2]), self.voxel_size)
            bbox.color.r = 0.0
            bbox.color.g = 1.0
            bbox.color.b = 0.0
            bbox.color.a = 0.45
            marker_array.markers.append(bbox)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = stamp
            text.ns = 'cluster_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(cluster.centroid[0])
            text.pose.position.y = float(cluster.centroid[1])
            text.pose.position.z = float(cluster.max_bound[2] + 0.25)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f'C{cluster.cluster_id} vox:{cluster.voxel_count}'
            marker_array.markers.append(text)
            marker_id += 1

        self.cluster_pose_pub.publish(pose_array)
        self.cluster_marker_pub.publish(marker_array)

    def publish_empty_outputs(self, stamp) -> None:
        self.dynamic_cloud_pub.publish(create_xyz_cloud(self.target_frame, stamp, np.empty((0, 3), dtype=np.float32)))

        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = stamp
        self.cluster_pose_pub.publish(pose_array)

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.cluster_marker_pub.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusterDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
