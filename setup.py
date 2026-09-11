import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rgbd_recorder_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools', 'PySide6'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='RGB-D visualiser and dataset recorder for ROS 2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgbd_recorder = rgbd_recorder_ros.rgbd_recorder_node:main',
        ],
    },
)
