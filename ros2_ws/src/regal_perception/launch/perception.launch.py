from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    
    # Wir starten den Node 6 mal, jeweils für einen anderen Lidar
    for i in range(1, 7):
        topic = f'/lidar_{i:02d}/points'
        node = Node(
            package='regal_perception',
            executable='detector_node',
            name=f'detector_lidar_{i:02d}',
            parameters=[{'lidar_topic': topic}]
        )
        ld.add_action(node)
        
    return ld
