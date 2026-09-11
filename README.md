# rgbd_recorder_ros

ROS 2 node that subscribes to colour/depth images, camera intrinsics and TF
extrinsics, visualises both streams side-by-side and records them as a
standard RGB-D dataset. Recording is toggled with the **SPACE** key (or the
`~/set_recording` SetBool service for headless use).

## Usage

```bash
ros2 run rgbd_recorder_ros rgbd_recorder
```
