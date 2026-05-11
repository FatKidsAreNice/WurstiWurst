from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable, List, Set, Tuple

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
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('min_cluster_voxels', 80)
        self.declare_parameter('max_cluster_voxels', 5000)
        self.declare_parameter('min_cluster_size', [0.35, 0.15, 0.08])
        self.declare_parameter('max_cluster_size', [1.20, 0.90, 0.80])
        self.declare_parameter('min_cluster_height', 0.02)
        self.declare_parameter('max_cluster_height', 1.20)

        self.declare_parameter('background_capture_duration_sec', 5.0)
        self.declare_parameter('background_min_occupancy_ratio', 0.85)
        self.declare_parameter('background_expansion_voxels', 1)
        self.declare_parameter('background_expansion_axes', [1, 1, 0])
        self.declare_parameter('publish_only_clustered_dynamic', True)

        self.declare_parameter('allow_touched_clusters', True)
        self.declare_parameter('touched_min_cluster_voxels', 60)
        self.declare_parameter('touched_max_cluster_voxels', 25000)
        self.declare_parameter('touched_min_cluster_size', [0.20, 0.08, 0.05])
        self.declare_parameter('touched_max_cluster_size', [2.50, 2.00, 2.20])
        self.declare_parameter('publish_touched_markers', True)

        self.declare_parameter('allow_footprint_projection', True)
        self.declare_parameter('footprint_min_cluster_voxels', 50)
        self.declare_parameter('footprint_max_cluster_voxels', 12000)
        self.declare_parameter('footprint_min_size', [0.45, 0.45, 0.02])
        self.declare_parameter('footprint_max_size', [1.25, 1.25, 0.35])
        self.declare_parameter('footprint_min_height', 0.80)
        self.declare_parameter('footprint_max_height', 1.80)
        self.declare_parameter('projection_floor_z', 0.0)
        self.declare_parameter('projection_height', 1.30)
        self.declare_parameter('publish_footprint_markers', True)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.min_cluster_voxels = int(self.get_parameter('min_cluster_voxels').value)
        self.max_cluster_voxels = int(self.get_parameter('max_cluster_voxels').value)
        self.min_cluster_size = np.asarray(self.get_parameter('min_cluster_size').value, dtype=np.float32)
        self.max_cluster_size = np.asarray(self.get_parameter('max_cluster_size').value, dtype=np.float32)
        self.min_cluster_height = float(self.get_parameter('min_cluster_height').value)
        self.max_cluster_height = float(self.get_parameter('max_cluster_height').value)

        self.background_capture_duration_sec = float(self.get_parameter('background_capture_duration_sec').value)
        self.background_min_occupancy_ratio = float(self.get_parameter('background_min_occupancy_ratio').value)
        self.background_expansion_voxels = int(self.get_parameter('background_expansion_voxels').value)
        self.background_expansion_axes = [int(value) for value in self.get_parameter('background_expansion_axes').value]
        self.publish_only_clustered_dynamic = bool(self.get_parameter('publish_only_clustered_dynamic').value)

        self.allow_touched_clusters = bool(self.get_parameter('allow_touched_clusters').value)
        self.touched_min_cluster_voxels = int(self.get_parameter('touched_min_cluster_voxels').value)
        self.touched_max_cluster_voxels = int(self.get_parameter('touched_max_cluster_voxels').value)
        self.touched_min_cluster_size = np.asarray(self.get_parameter('touched_min_cluster_size').value, dtype=np.float32)
        self.touched_max_cluster_size = np.asarray(self.get_parameter('touched_max_cluster_size').value, dtype=np.float32)
        self.publish_touched_markers = bool(self.get_parameter('publish_touched_markers').value)

        self.allow_footprint_projection = bool(self.get_parameter('allow_footprint_projection').value)
        self.footprint_min_cluster_voxels = int(self.get_parameter('footprint_min_cluster_voxels').value)
        self.footprint_max_cluster_voxels = int(self.get_parameter('footprint_max_cluster_voxels').value)
        self.footprint_min_size = np.asarray(self.get_parameter('footprint_min_size').value, dtype=np.float32)
        self.footprint_max_size = np.asarray(self.get_parameter('footprint_max_size').value, dtype=np.float32)
        self.footprint_min_height = float(self.get_parameter('footprint_min_height').value)
        self.footprint_max_height = float(self.get_parameter('footprint_max_height').value)
        self.projection_floor_z = float(self.get_parameter('projection_floor_z').value)
        self.projection_height = float(self.get_parameter('projection_height').value)
        self.publish_footprint_markers = bool(self.get_parameter('publish_footprint_markers').value)

        self.background_keys: Set[VoxelKey] = set()
        self.last_points = np.empty((0, 3), dtype=np.float32)
        self.last_stamp = None

        self.is_capturing_background = False
        self.background_capture_start_sec = 0.0
        self.background_frame_count = 0
        self.background_voxel_counter: Counter[VoxelKey] = Counter()

        self.dynamic_cloud_pub = self.create_publisher(PointCloud2, '/tracking/dynamic_cloud', 10)
        self.cluster_pose_pub = self.create_publisher(PoseArray, '/tracking/cluster_centroids', 10)
        self.touched_cluster_pose_pub = self.create_publisher(PoseArray, '/tracking/touched_cluster_centroids', 10)
        self.cluster_marker_pub = self.create_publisher(MarkerArray, '/tracking/cluster_markers', 10)

        self.cloud_sub = self.create_subscription(PointCloud2, '/tracking/merged_cloud', self.cloud_callback, 10)

        self.capture_background_srv = self.create_service(Trigger, '/tracking/capture_background', self.capture_background)
        self.clear_background_srv = self.create_service(Trigger, '/tracking/clear_background', self.clear_background)

        self.neighbor_offsets = [
            offset for offset in product([-1, 0, 1], repeat=3)
            if offset != (0, 0, 0)
        ]

        self.background_expansion_offsets = self.build_background_expansion_offsets()

        self.get_logger().info('cluster_detector_node started.')
        self.get_logger().info('Call /tracking/capture_background while the hall is empty.')

    def cloud_callback(self, cloud_msg: PointCloud2) -> None:
        self.last_points = extract_xyz_points(cloud_msg)
        self.last_stamp = cloud_msg.header.stamp

        if self.is_capturing_background:
            self.collect_background_frame(cloud_msg)
            self.publish_empty_outputs(cloud_msg.header.stamp)
            return

        if not self.background_keys:
            self.publish_empty_outputs(cloud_msg.header.stamp)
            self.get_logger().warning(
                'Background not captured yet. Call /tracking/capture_background.',
                throttle_duration_sec=5.0,
            )
            return

        current_keys = points_to_voxel_keys(self.last_points, self.voxel_size)
        dynamic_keys = current_keys - self.background_keys
        all_clusters = self.cluster_voxels(dynamic_keys)
        accepted_cluster_infos = self.build_accepted_cluster_infos(all_clusters)

        if self.publish_only_clustered_dynamic:
            accepted_keys: Set[VoxelKey] = set()
            for cluster in all_clusters:
                cluster_info = self.build_cluster_info_from_keys(cluster, cluster_id=0, cluster_type='candidate')
                if self.classify_cluster(cluster_info) is not None:
                    accepted_keys.update(cluster)
            dynamic_points = voxel_keys_to_centers(accepted_keys, self.voxel_size)
        else:
            dynamic_points = voxel_keys_to_centers(dynamic_keys, self.voxel_size)

        self.dynamic_cloud_pub.publish(create_xyz_cloud(self.target_frame, cloud_msg.header.stamp, dynamic_points))
        self.publish_cluster_outputs(accepted_cluster_infos, cloud_msg.header.stamp)

    def capture_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.last_points.size == 0 or self.last_stamp is None:
            response.success = False
            response.message = 'No merged cloud received yet.'
            return response

        self.is_capturing_background = True
        self.background_capture_start_sec = self.current_time_sec()
        self.background_frame_count = 0
        self.background_voxel_counter.clear()
        self.background_keys.clear()

        response.success = True
        response.message = (
            f'Background capture started for {self.background_capture_duration_sec:.2f} seconds. '
            'Keep the scene empty and still.'
        )
        self.get_logger().info(response.message)
        return response

    def clear_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.background_keys.clear()
        self.is_capturing_background = False
        self.background_voxel_counter.clear()
        self.background_frame_count = 0
        response.success = True
        response.message = 'Background cleared.'
        self.get_logger().info(response.message)
        return response

    def collect_background_frame(self, cloud_msg: PointCloud2) -> None:
        elapsed_sec = self.current_time_sec() - self.background_capture_start_sec
        current_keys = points_to_voxel_keys(self.last_points, self.voxel_size)
        self.background_voxel_counter.update(current_keys)
        self.background_frame_count += 1

        self.get_logger().info(
            f'Capturing background: {elapsed_sec:.2f}s / {self.background_capture_duration_sec:.2f}s, '
            f'frames={self.background_frame_count}, current_voxels={len(current_keys)}.',
            throttle_duration_sec=1.0,
        )

        if elapsed_sec < self.background_capture_duration_sec:
            return

        self.finish_background_capture()

    def finish_background_capture(self) -> None:
        if self.background_frame_count <= 0:
            self.background_keys.clear()
            self.is_capturing_background = False
            self.get_logger().warning('Background capture finished without frames.')
            return

        ratio = min(max(self.background_min_occupancy_ratio, 0.0), 1.0)
        min_count = max(1, int(np.ceil(float(self.background_frame_count) * ratio)))
        stable_keys = {
            key for key, count in self.background_voxel_counter.items()
            if count >= min_count
        }
        expanded_keys = self.expand_background_keys(stable_keys)

        self.background_keys = expanded_keys
        self.is_capturing_background = False
        self.background_voxel_counter.clear()

        self.get_logger().info(
            f'Background captured from {self.background_frame_count} frames. '
            f'Stable voxels={len(stable_keys)}, expanded voxels={len(self.background_keys)}, '
            f'min_count={min_count}.'
        )

    def build_background_expansion_offsets(self) -> List[VoxelKey]:
        radius = max(self.background_expansion_voxels, 0)
        axes = self.background_expansion_axes
        if len(axes) != 3:
            axes = [1, 1, 1]

        ranges = []
        for axis_index in range(3):
            if axes[axis_index]:
                ranges.append(range(-radius, radius + 1))
            else:
                ranges.append(range(0, 1))

        return [tuple(int(value) for value in offset) for offset in product(*ranges)]

    def expand_background_keys(self, stable_keys: Set[VoxelKey]) -> Set[VoxelKey]:
        if self.background_expansion_voxels <= 0:
            return set(stable_keys)

        expanded: Set[VoxelKey] = set()
        for key in stable_keys:
            for offset in self.background_expansion_offsets:
                expanded.add((key[0] + offset[0], key[1] + offset[1], key[2] + offset[2]))
        return expanded

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

            clusters.append(cluster)

        return clusters

    def build_accepted_cluster_infos(self, clusters: Iterable[List[VoxelKey]]) -> List[ClusterInfo]:
        accepted_cluster_infos: List[ClusterInfo] = []

        for cluster in clusters:
            cluster_info = self.build_cluster_info_from_keys(
                cluster,
                cluster_id=len(accepted_cluster_infos),
                cluster_type='candidate',
            )
            cluster_type = self.classify_cluster(cluster_info)
            if cluster_type is None:
                continue

            if cluster_type == 'footprint':
                cluster_info = self.project_footprint_cluster(cluster_info)

            cluster_info.cluster_id = len(accepted_cluster_infos)
            cluster_info.cluster_type = cluster_type
            accepted_cluster_infos.append(cluster_info)

        return accepted_cluster_infos

    def build_cluster_info_from_keys(self, cluster: List[VoxelKey], cluster_id: int, cluster_type: str) -> ClusterInfo:
        cluster_points = voxel_keys_to_centers(cluster, self.voxel_size)
        centroid = np.mean(cluster_points, axis=0)
        min_bound = np.min(cluster_points, axis=0)
        max_bound = np.max(cluster_points, axis=0)
        return ClusterInfo(
            cluster_id=cluster_id,
            centroid=centroid.astype(np.float32),
            min_bound=min_bound.astype(np.float32),
            max_bound=max_bound.astype(np.float32),
            voxel_count=len(cluster),
            cluster_type=cluster_type,
        )

    def classify_cluster(self, cluster: ClusterInfo) -> str | None:
        if self.is_normal_cluster(cluster):
            return 'normal'

        if self.allow_footprint_projection and self.is_footprint_cluster(cluster):
            return 'footprint'

        if self.allow_touched_clusters and self.is_touched_cluster(cluster):
            return 'touched'

        return None

    def is_normal_cluster(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        if cluster.voxel_count < self.min_cluster_voxels or cluster.voxel_count > self.max_cluster_voxels:
            return False
        if not np.all(size >= self.min_cluster_size):
            return False
        if not np.all(size <= self.max_cluster_size):
            return False
        if float(cluster.min_bound[2]) < self.min_cluster_height:
            return False
        if float(cluster.max_bound[2]) > self.max_cluster_height:
            return False
        return True


    def is_footprint_cluster(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        centroid_z = float(cluster.centroid[2])

        if cluster.voxel_count < self.footprint_min_cluster_voxels:
            return False
        if cluster.voxel_count > self.footprint_max_cluster_voxels:
            return False
        if not np.all(size >= self.footprint_min_size):
            return False
        if not np.all(size <= self.footprint_max_size):
            return False
        if centroid_z < self.footprint_min_height:
            return False
        if centroid_z > self.footprint_max_height:
            return False
        return True

    def project_footprint_cluster(self, cluster: ClusterInfo) -> ClusterInfo:
        projected_min = cluster.min_bound.copy()
        projected_max = cluster.max_bound.copy()
        floor_z = self.projection_floor_z
        top_z = self.projection_floor_z + max(self.projection_height, self.voxel_size)

        projected_min[2] = floor_z
        projected_max[2] = top_z

        projected_centroid = np.array(
            [
                float((projected_min[0] + projected_max[0]) * 0.5),
                float((projected_min[1] + projected_max[1]) * 0.5),
                float((projected_min[2] + projected_max[2]) * 0.5),
            ],
            dtype=np.float32,
        )

        return ClusterInfo(
            cluster_id=cluster.cluster_id,
            centroid=projected_centroid,
            min_bound=projected_min.astype(np.float32),
            max_bound=projected_max.astype(np.float32),
            voxel_count=cluster.voxel_count,
            cluster_type=cluster.cluster_type,
        )

    def is_touched_cluster(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        if cluster.voxel_count < self.touched_min_cluster_voxels:
            return False
        if cluster.voxel_count > self.touched_max_cluster_voxels:
            return False
        if not np.all(size >= self.touched_min_cluster_size):
            return False
        if not np.all(size <= self.touched_max_cluster_size):
            return False
        if float(cluster.min_bound[2]) < self.min_cluster_height:
            return False
        if float(cluster.max_bound[2]) > self.touched_max_cluster_size[2]:
            return False
        return True

    def publish_cluster_outputs(self, cluster_infos: List[ClusterInfo], stamp) -> None:
        normal_clusters = [cluster for cluster in cluster_infos if cluster.cluster_type in ('normal', 'footprint')]
        touched_clusters = [cluster for cluster in cluster_infos if cluster.cluster_type == 'touched']

        self.cluster_pose_pub.publish(self.build_pose_array(normal_clusters, stamp))
        self.touched_cluster_pose_pub.publish(self.build_pose_array(touched_clusters, stamp))
        self.cluster_marker_pub.publish(self.build_marker_array(cluster_infos, stamp))

    def build_pose_array(self, cluster_infos: List[ClusterInfo], stamp) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = stamp

        for cluster in cluster_infos:
            pose = Pose()
            pose.position.x = float(cluster.centroid[0])
            pose.position.y = float(cluster.centroid[1])
            pose.position.z = float(cluster.centroid[2])
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        return pose_array

    def build_marker_array(self, cluster_infos: List[ClusterInfo], stamp) -> MarkerArray:
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for cluster in cluster_infos:
            if cluster.cluster_type == 'touched' and not self.publish_touched_markers:
                continue
            if cluster.cluster_type == 'footprint' and not self.publish_footprint_markers:
                continue

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

            if cluster.cluster_type == 'normal':
                bbox.color.r = 0.0
                bbox.color.g = 1.0
                bbox.color.b = 0.0
            elif cluster.cluster_type == 'footprint':
                bbox.color.r = 0.0
                bbox.color.g = 0.65
                bbox.color.b = 1.0
            else:
                bbox.color.r = 1.0
                bbox.color.g = 0.55
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
            text.text = f'C{cluster.cluster_id} {cluster.cluster_type} vox:{cluster.voxel_count}'
            marker_array.markers.append(text)
            marker_id += 1

        return marker_array

    def publish_empty_outputs(self, stamp) -> None:
        self.dynamic_cloud_pub.publish(create_xyz_cloud(self.target_frame, stamp, np.empty((0, 3), dtype=np.float32)))
        self.cluster_pose_pub.publish(self.build_pose_array([], stamp))
        self.touched_cluster_pose_pub.publish(self.build_pose_array([], stamp))

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.cluster_marker_pub.publish(marker_array)

    def current_time_sec(self) -> float:
        now = self.get_clock().now().to_msg()
        return float(now.sec) + float(now.nanosec) * 1e-9


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
