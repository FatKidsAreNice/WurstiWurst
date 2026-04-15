from setuptools import find_packages, setup

package_name = 'coldstore_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/tracking_pipeline.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenAI',
    maintainer_email='noreply@example.com',
    description='Minimal cold store multi-lidar tracking pipeline for ROS 2 Jazzy',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cloud_transform_merge_node = coldstore_tracking.cloud_transform_merge_node:main',
            'cluster_detector_node = coldstore_tracking.cluster_detector_node:main',
            'track_manager_node = coldstore_tracking.track_manager_node:main',
        ],
    },
)
