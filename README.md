# WurstiWurst


nano ~/.bashrc

hinzufügen:

# Grafik & Tastatur Fixes (wichtig für VNC)
export QT_X11_NO_MITSHM=1
export XKB_DEFAULT_RULES=base

# ROS 2 Jazzy Core
source /opt/ros/jazzy/setup.bash

# Dein Workspace (Vorsicht: siehe Hinweis unten!)
if [ -f ~/ros2_ws/install/setup.bash ]; then source ~/ros2_ws/install/setup.bas>

# Gazebo / Ignition Settings
export GZ_IP=127.0.0.1
export GZ_PARTITION=""

# CUDA 13.2 - Volle Pfade
export PATH=/usr/local/cuda-13.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRA>
source /opt/ros/jazzy/setup.bash


Wie starten?
Simulation (ACHTUNG MUSS LEER SEIN vom Inhalt):
Terminal 1: LIBGL_ALWAYS_SOFTWARE=1 gz sim -s -v 4 ~/ros2_ws/worlds/kuehlhaus.sdf
Terminal 2: LIBGL_ALWAYS_SOFTWARE=1 gz sim -g

Bridge:
Terminal 3:
source /opt/ros/jazzy/setup.bash

ros2 run ros_gz_bridge parameter_bridge \
/world/kuehlhaus_world/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
/lidar_01/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
/lidar_02/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
/lidar_03/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
/lidar_04/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
/lidar_05/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
/lidar_06/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
--ros-args \
-r /world/kuehlhaus_world/clock:=/clock \
-r /lidar_01/points/points:=/lidar_01/points \
-r /lidar_02/points/points:=/lidar_02/points \
-r /lidar_03/points/points:=/lidar_03/points \
-r /lidar_04/points/points:=/lidar_04/points \
-r /lidar_05/points/points:=/lidar_05/points \
-r /lidar_06/points/points:=/lidar_06/points

Tracking-Pipeline:
Terminal 4:
source ~/ros2_ws/install/setup.bash
ros2 launch coldstore_tracking tracking_pipeline.launch.py

Hintergrund entfernen:
Terminal 5:
ros2 service call /tracking/capture_background std_srvs/srv/Trigger

RVIZ Anzeigen lassen, add MergedCloud, DynamicCloud, ClusterMakers und TrackMarkers, Frame: world
Terminal 6:
ros2 run rviz2 rviz2
