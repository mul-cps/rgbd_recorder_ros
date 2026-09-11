import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rgbd_recorder_ros'),
        'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='rgbd_recorder_ros',
            executable='rgbd_recorder',
            name='rgbd_recorder',
            output='screen',
            parameters=[config],
        ),
    ])
