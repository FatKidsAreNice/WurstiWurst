from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .event_utils import build_track_states_payload, make_string_message, parse_string_message
from .tracking_types import Track


class TrackManagerNode(Node):
    def __init__(self) -> None:
        super().__init__('track_manager_node')

        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('max_match_distance', 1.2)
        self.declare_parameter('max_missed_updates', 8)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.max_match_distance = float(self.get_parameter('max_match_distance').value)
        self.max_missed_updates = int(self.get_parameter('max_missed_updates').value)

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.last_stamp_sec = 0.0
        self.last_stamp_msg = None

        self.centroid_sub = self.create_subscription(
            PoseArray,
            '/tracking/cluster_centroids',
            self.centroid_callback,
            10,
        )
        self.assignment_sub = self.create_subscription(
            String,
            '/tracking/id_assignments',
            self.assignment_callback,
            10,
        )
        self.remove_track_sub = self.create_subscription(
            String,
            '/tracking/remove_track_events',
            self.remove_track_callback,
            10,
        )

        self.track_marker_pub = self.create_publisher(MarkerArray, '/tracking/track_markers', 10)
        self.track_state_pub = self.create_publisher(String, '/tracking/track_states', 10)

        self.clear_tracks_srv = self.create_service(
            Trigger,
            '/tracking/clear_tracks',
            self.clear_tracks,
        )

        self.get_logger().info('track_manager_node started.')

    def centroid_callback(self, pose_array: PoseArray) -> None:
        stamp_sec = float(pose_array.header.stamp.sec) + float(pose_array.header.stamp.nanosec) * 1e-9
        detections = [
            np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
            for pose in pose_array.poses
        ]

        self.last_stamp_sec = stamp_sec
        self.last_stamp_msg = pose_array.header.stamp

        self.update_tracks(detections, stamp_sec)
        self.publish_track_markers(pose_array.header.stamp)
        self.publish_track_states(stamp_sec)

    def update_tracks(self, detections: List[np.ndarray], stamp_sec: float) -> None:
        if not self.tracks:
            for detection in detections:
                self.create_track(detection, stamp_sec)
            return

        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))
        candidate_matches: List[Tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(track.centroid - detection))
                if distance <= self.max_match_distance:
                    candidate_matches.append((distance, track_id, detection_index))

        candidate_matches.sort(key=lambda item: item[0])

        for _, track_id, detection_index in candidate_matches:
            if track_id not in unmatched_track_ids:
                continue
            if detection_index not in unmatched_detection_indices:
                continue

            self.update_track(track_id, detections[detection_index], stamp_sec)
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed_updates += 1
            if track.missed_updates > self.max_missed_updates:
                del self.tracks[track_id]

        for detection_index in sorted(unmatched_detection_indices):
            self.create_track(detections[detection_index], stamp_sec)

    def create_track(self, detection: np.ndarray, stamp_sec: float) -> None:
        track = Track(
            track_id=self.next_track_id,
            centroid=detection.copy(),
            velocity=np.zeros(3, dtype=np.float32),
            age=1,
            missed_updates=0,
            last_stamp_sec=stamp_sec,
            barcode_id='',
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    def update_track(self, track_id: int, detection: np.ndarray, stamp_sec: float) -> None:
        track = self.tracks[track_id]
        dt = max(stamp_sec - track.last_stamp_sec, 1e-3)
        track.velocity = (detection - track.centroid) / dt
        track.centroid = detection.copy()
        track.age += 1
        track.missed_updates = 0
        track.last_stamp_sec = stamp_sec

    def assignment_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        track_id = int(payload.get('track_id', 0))
        barcode_id = str(payload.get('barcode_id', ''))

        if track_id <= 0 or not barcode_id:
            return

        track = self.tracks.get(track_id)
        if track is None:
            self.get_logger().warning(
                f'Received assignment for unknown track_id={track_id}.',
                throttle_duration_sec=5.0,
            )
            return

        if track.barcode_id == barcode_id:
            return

        track.barcode_id = barcode_id
        self.get_logger().info(f'Assigned barcode "{barcode_id}" to track T{track_id}.')
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())

    def remove_track_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        track_id = int(payload.get('track_id', 0))
        barcode_id = str(payload.get('barcode_id', ''))

        if track_id <= 0:
            return

        track = self.tracks.get(track_id)
        if track is None:
            self.get_logger().warning(
                f'Received remove request for unknown track_id={track_id}.',
                throttle_duration_sec=5.0,
            )
            return

        if barcode_id and track.barcode_id and track.barcode_id != barcode_id:
            self.get_logger().warning(
                f'Remove request barcode mismatch for T{track_id}: '
                f'track="{track.barcode_id}" request="{barcode_id}".',
                throttle_duration_sec=5.0,
            )
            return

        del self.tracks[track_id]
        self.get_logger().info(f'Removed track T{track_id} with barcode "{barcode_id}".')
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())

    def clear_tracks(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.tracks.clear()
        self.next_track_id = 1
        response.success = True
        response.message = 'All tracks cleared.'
        self.get_logger().info(response.message)
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())
        return response

    def publish_track_states(self, stamp_sec: float) -> None:
        payload = build_track_states_payload(self.tracks, stamp_sec)
        self.track_state_pub.publish(make_string_message(payload))

    def publish_track_markers(self, stamp) -> None:
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            sphere = Marker()
            sphere.header.frame_id = self.target_frame
            sphere.header.stamp = stamp
            sphere.ns = 'tracks'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(track.centroid[0])
            sphere.pose.position.y = float(track.centroid[1])
            sphere.pose.position.z = float(track.centroid[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            marker_array.markers.append(sphere)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = stamp
            text.ns = 'track_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(track.centroid[0])
            text.pose.position.y = float(track.centroid[1])
            text.pose.position.z = float(track.centroid[2] + 0.35)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.28
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 0.0
            text.color.a = 1.0

            speed = float(np.linalg.norm(track.velocity))
            barcode = track.barcode_id if track.barcode_id else '-'
            text.text = f'T{track.track_id} id:{barcode} age:{track.age} v:{speed:.2f}m/s'
            marker_array.markers.append(text)
            marker_id += 1

        self.track_marker_pub.publish(marker_array)

    def current_stamp_sec(self) -> float:
        if self.last_stamp_sec > 0.0:
            return self.last_stamp_sec

        now_msg = self.get_clock().now().to_msg()
        return float(now_msg.sec) + float(now_msg.nanosec) * 1e-9

    def current_stamp_msg(self):
        if self.last_stamp_msg is not None:
            return self.last_stamp_msg
        return self.get_clock().now().to_msg()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()