# rgbd_recorder_ros

ROS 2 node that subscribes to colour/depth images, camera intrinsics and TF
extrinsics, visualises both streams side-by-side and records them as a
standard RGB-D dataset. Recording is toggled with the **SPACE** key (or the
`~/set_recording` SetBool service for headless use).

## Installation

On older ROS 2 distributions, you have to install `PySide6` manually via pip:
```bash
pip install PySide6
```


## Usage

Launch the node with default parameters:
```bash
ros2 run rgbd_recorder_ros rgbd_recorder
```

Launch the node with custom output folder and maximum visualisation depth:
```bash
ros2 run rgbd_recorder_ros rgbd_recorder --ros-args \
  -p output_dir:=~/rgbd_dataset/ \
  -p depth_vis_max_mm:=2000.0
```


## Topics

- `/camera/color/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`)
- `/camera/depth/image_raw/compressed` (type: `sensor_msgs/msg/CompressedImage`)
- `/camera/color/camera_info` (type: `sensor_msgs/msg/CameraInfo`)
- `/camera/depth/camera_info` (type: `sensor_msgs/msg/CameraInfo`)
- `/camera/depth_to_color` (type: `orbbec_camera_msgs/msg/Extrinsics`, if available)
