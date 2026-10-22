# rgbd_recorder_ros

ROS 2 node that subscribes to colour/depth images, camera intrinsics and TF
extrinsics, visualises both streams side-by-side and records them as a
standard RGB-D dataset. Recording is toggled with the **SPACE** key (or the
`~/set_recording` SetBool service for headless use).

## Usage

```bash
ros2 run rgbd_recorder_ros rgbd_recorder
```

# Topics

- `/camera/color/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`)
- `/camera/depth/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`)
- `/camera/color/camera_info` (type: `sensor_msgs/msg/CameraInfo`)
- `/camera/depth/camera_info` (type: `sensor_msgs/msg/CameraInfo`)
- `/camera/depth_to_color` (type: `orbbec_camera_msgs/msg/Extrinsics`, if available)
