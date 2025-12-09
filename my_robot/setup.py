import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'my_robot' 

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'configs'),
         glob('my_robot/configs/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
         glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'urdf'),   glob('my_robot/urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('my_robot/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='',
    description='Gói ROS 2 xuất bản RGBD + IMU + LiDAR + (tuỳ chọn) YOLO và launch RTAB-Map.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'realsense_yolo_node  = my_robot.nodes.realsense_yolo_node:main',
            'imu_to_tf = my_robot.nodes.imu_to_tf:main',
            'rplidar_node = my_robot.nodes.rplidar_node:main',
        ],
    },
)
