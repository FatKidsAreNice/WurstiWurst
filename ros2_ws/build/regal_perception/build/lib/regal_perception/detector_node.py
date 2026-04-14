import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
import time

class WorldFusionDetector(Node):
    def __init__(self):
        super().__init__('world_fusion_detector')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, 'detected_shelves_blue', 10)

        self.latest_detections = {} 
        self.active_ids = set()

        self.subs = [self.create_subscription(PointCloud2, f'/lidar_{i:02d}/points', 
                     lambda msg, i=i: self.pc_callback(msg, i), 10) for i in range(1, 7)]
        
        self.timer = self.create_timer(0.1, self.publish_combined_markers)
        self.get_logger().info('PRECISION-VERSION: Doppelbox-Fix & Wand-Filter aktiv.')

    def get_transformation_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            t = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            r = R.from_quat(q).as_matrix()
            m = np.identity(4)
            m[0:3, 0:3], m[0:3, 3] = r, t
            return m
        except: return None

    def pc_callback(self, msg, lidar_id):
        mat = self.get_transformation_matrix('world', msg.header.frame_id)
        if mat is None: return

        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts_list = [[float(p[0]), float(p[1]), float(p[2])] for p in gen]
        pts = np.array(pts_list, dtype=np.float32)
        if pts.size == 0: return
        
        pts_world = (np.hstack((pts, np.ones((pts.shape[0], 1)))) @ mat.T)[:, :3]

        # 1.8m Höhenschnitt
        mask = (pts_world[:, 2] > 1.65) & (pts_world[:, 2] < 1.95)
        relevant = pts_world[mask]
        
        current_frame_shelves = []
        if len(relevant) >= 5:
            # eps auf 0.4 erhöht, um Fragmente einer Box zu EINEM Cluster zu verbinden
            clustering = DBSCAN(eps=0.4, min_samples=5).fit(relevant[:, :2])
            for label in set(clustering.labels_):
                if label == -1: continue
                cp = relevant[clustering.labels_ == label]
                min_p, max_p = np.min(cp, axis=0), np.max(cp, axis=0)
                dims = max_p - min_p
                
                max_s = max(dims[0], dims[1])
                min_s = min(dims[0], dims[1])

                # VERBESSERTER FILTER:
                # 1. Länge zwischen 0.5m und 1.1m
                # 2. min_s > 0.06 (etwas strenger gegen Wände als 0.04)
                if (0.50 < max_s < 1.1) and (min_s > 0.06):
                    current_frame_shelves.append([(min_p[0]+max_p[0])/2, (min_p[1]+max_p[1])/2])
        
        if current_frame_shelves:
            self.latest_detections[lidar_id] = (current_frame_shelves, time.time())

    def publish_combined_markers(self):
        current_time = time.time()
        all_centers = []
        
        for lid_id in list(self.latest_detections.keys()):
            coords, timestamp = self.latest_detections[lid_id]
            if current_time - timestamp < 0.8:
                all_centers.extend(coords)
        
        if not all_centers: return

        # RÄUMLICHE FUSION: Radius auf 0.7m erhöht! 
        # Das verschmilzt "Körper an Körper" Boxen zu einer einzigen.
        final_shelves = []
        for c in all_centers:
            if not any(np.linalg.norm(np.array(c) - np.array(f)) < 0.7 for f in final_shelves):
                final_shelves.append(c)

        marker_array = MarkerArray()
        new_active_ids = set()

        for i, (x, y) in enumerate(final_shelves):
            # Stabiles 20cm Raster für die ID
            grid_x, grid_y = round(x * 5) / 5, round(y * 5) / 5
            m_id = abs(hash((grid_x, grid_y))) % 10000
            
            # Position leicht glätten (Snap auf 2.5cm statt 5cm für mehr Präzision)
            snapped_x, snapped_y = round(x * 40) / 40, round(y * 40) / 40
            
            marker_array.markers.append(self.create_marker([snapped_x, snapped_y, 0.9], m_id))
            new_active_ids.add(m_id)

        # Cleanup
        for old_id in self.active_ids - new_active_ids:
            del_marker = Marker()
            del_marker.header.frame_id = "world"
            del_marker.id = int(old_id)
            del_marker.action = Marker.DELETE
            marker_array.markers.append(del_marker)

        self.active_ids = new_active_ids
        self.marker_pub.publish(marker_array)

    def create_marker(self, center, m_id):
        m = Marker()
        m.header.frame_id, m.header.stamp = "world", self.get_clock().now().to_msg()
        m.id, m.type, m.action = int(m_id), Marker.CUBE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = center
        m.scale.x, m.scale.y, m.scale.z = 0.8, 0.8, 1.8
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.4, 1.0, 0.8
        m.lifetime = rclpy.duration.Duration(seconds=1.2).to_msg()
        m.frame_locked = True
        return m

def main():
    rclpy.init()
    node = WorldFusionDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
