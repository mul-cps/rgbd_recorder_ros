#!/usr/bin/env python3
"""
RGB-D recorder node with Qt6 (PySide6) GUI — compressed transport version.

- Subscribes to sensor_msgs/CompressedImage for colour and depth.
- Validates payload formats on the first frame:
    * depth MUST be PNG (fatal error + exit otherwise)
    * colour may be JPG or PNG, exported in its native format
- Export writes the raw compressed payload byte-for-byte to disk
  (no decode/re-encode); decoding happens only for the live preview.
- Also records intrinsics (CameraInfo), depth->colour extrinsics
  (orbbec_camera_msgs/Extrinsics, if available) and TF camera poses.
- SPACE toggles recording, Q/ESC quits. Output dirs created automatically.

Output layout:
    <output_dir>/<session_stamp>/
        rgb/<timestamp>.jpg|png      # exact bytes from the compressed topic
        depth/<timestamp>.png        # exact bytes from the compressed topic
        intrinsics.json              # K, D, resolution (colour + depth)
        extrinsics.json              # depth->color from the driver
        poses.txt                    # TUM: t tx ty tz qx qy qz qw
        associations.txt             # t_rgb rgb/... t_depth depth/...
"""

import json
import os
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_srvs.srv import SetBool

import tf2_ros

# Orbbec extrinsics topic is optional — degrade gracefully if msgs missing.
try:
    from orbbec_camera_msgs.msg import Extrinsics as OrbbecExtrinsics
    HAVE_ORBBEC_MSGS = True
except ImportError:
    HAVE_ORBBEC_MSGS = False

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QStatusBar, QVBoxLayout, QWidget)


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def detect_format(fmt: str, data: bytes) -> str:
    """Identify the actual payload format from the format string and,
    as a fallback, the file magic bytes."""
    fmt = fmt.lower()
    if 'jpeg' in fmt or 'jpg' in fmt:
        return 'jpg'
    if 'png' in fmt:
        return 'png'
    # Fall back to magic numbers if the format string is unhelpful
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:2] == b'\xff\xd8':
        return 'jpg'
    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
#  Signal bridge: ROS callback thread -> Qt GUI thread
# ─────────────────────────────────────────────────────────────────────────────
class RosQtBridge(QObject):
    # colour (BGR uint8), depth (uint16 mm), recording flag, frame count
    frame_ready = Signal(object, object, bool, int)
    log_message = Signal(str)
    fatal_error = Signal(str)


# ─────────────────────────────────────────────────────────────────────────────
#  ROS node
# ─────────────────────────────────────────────────────────────────────────────
class RGBDRecorder(Node):
    def __init__(self, bridge: RosQtBridge):
        super().__init__('rgbd_recorder')
        self.bridge_qt = bridge

        # ---------------- Parameters ----------------
        self.declare_parameter('color_topic',
                               '/camera/color/image_raw/compressed')
        self.declare_parameter('depth_topic',
                               '/camera/depth/image_raw/compressed')
        self.declare_parameter('color_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('extrinsics_topic', '/camera/depth_to_color')
        self.declare_parameter('base_frame', 'map')
        self.declare_parameter('camera_frame', '')   # empty -> header frame_id
        self.declare_parameter('output_dir', '~/rgbd_dataset')
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('depth_vis_max_mm', 5000.0)

        gp = self.get_parameter
        self.base_frame = gp('base_frame').value
        self.camera_frame = gp('camera_frame').value
        # Expand ~ and env vars, create the root folder right away
        self.output_root = os.path.expandvars(
            os.path.expanduser(gp('output_dir').value))
        os.makedirs(self.output_root, exist_ok=True)

        # ---------------- State ----------------
        self._lock = threading.Lock()
        self.recording = False
        self.session_dir = None
        self.poses_file = None
        self.assoc_file = None
        self.frame_count = 0
        self.color_info = None
        self.depth_info = None
        self.depth_to_color = None      # orbbec extrinsics msg
        self.metadata_written = False

        # Format validation (done once, on the first synchronized frame)
        self.formats_validated = False
        self.color_ext = None
        self.depth_ext = None

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- Subscribers ----------------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.color_sub = Subscriber(self, CompressedImage,
                                    gp('color_topic').value, qos_profile=qos)
        self.depth_sub = Subscriber(self, CompressedImage,
                                    gp('depth_topic').value, qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10, slop=gp('sync_slop').value)
        self.sync.registerCallback(self.rgbd_callback)

        self.create_subscription(CameraInfo, gp('color_info_topic').value,
                                 self.color_info_cb, qos)
        self.create_subscription(CameraInfo, gp('depth_info_topic').value,
                                 self.depth_info_cb, qos)

        if HAVE_ORBBEC_MSGS:
            self.create_subscription(
                OrbbecExtrinsics, gp('extrinsics_topic').value,
                self.extrinsics_cb, qos)
        else:
            self.get_logger().warning(
                'orbbec_camera_msgs not available — depth->color extrinsics '
                'topic will not be recorded (TF poses still are).')

        self.create_service(SetBool, '~/set_recording', self.set_recording_cb)

        self.get_logger().info(
            f'RGB-D recorder ready (compressed transport). '
            f'Output root: {self.output_root}')

    # ------------------------------------------------------------------ #
    #  Info / extrinsics callbacks                                       #
    # ------------------------------------------------------------------ #
    def color_info_cb(self, msg):
        self.color_info = msg

    def depth_info_cb(self, msg):
        self.depth_info = msg

    def extrinsics_cb(self, msg):
        self.depth_to_color = msg

    @staticmethod
    def camera_info_to_dict(info: CameraInfo) -> dict:
        return {
            'width': info.width, 'height': info.height,
            'distortion_model': info.distortion_model,
            'K': list(info.k), 'D': list(info.d),
            'fx': info.k[0], 'fy': info.k[4],
            'cx': info.k[2], 'cy': info.k[5],
        }

    # ------------------------------------------------------------------ #
    #  Format validation (first frame only)                              #
    # ------------------------------------------------------------------ #
    def _validate_formats(self, color_msg, depth_msg):
        """Check payload formats on the first frame. Depth MUST be PNG
        (JPEG is lossy and cannot represent 16-bit depth). Colour may be
        JPG or PNG and is exported in its native format."""
        color_fmt = detect_format(color_msg.format, bytes(color_msg.data[:8]))
        depth_fmt = detect_format(depth_msg.format, bytes(depth_msg.data[:8]))

        if depth_fmt != 'png':
            msg = (f'FATAL: depth topic delivers "{depth_fmt}" '
                   f'(format string: "{depth_msg.format}"), but PNG is '
                   f'required. JPEG or other lossy formats destroy 16-bit '
                   f'depth data. Reconfigure the image transport '
                   f'(png_level) or use the raw depth topic. Exiting.')
            self.get_logger().fatal(msg)
            self.bridge_qt.fatal_error.emit(msg)
            return False

        if color_fmt not in ('jpg', 'png'):
            msg = (f'FATAL: colour topic delivers unsupported format '
                   f'"{color_fmt}" (format string: "{color_msg.format}"). '
                   f'Expected JPG or PNG. Exiting.')
            self.get_logger().fatal(msg)
            self.bridge_qt.fatal_error.emit(msg)
            return False

        self.color_ext = color_fmt          # native format, jpg or png
        self.depth_ext = depth_fmt          # always png
        self.formats_validated = True
        self.get_logger().info(
            f'Formats validated: colour={color_fmt}, depth={depth_fmt}')
        return True

    # ------------------------------------------------------------------ #
    #  Recording control (callable from GUI thread and service)          #
    # ------------------------------------------------------------------ #
    def set_recording_cb(self, request, response):
        self.toggle_recording(force=request.data)
        response.success = True
        response.message = f'recording={self.recording}'
        return response

    def toggle_recording(self, force=None):
        with self._lock:
            target = (not self.recording) if force is None else force
            if target and not self.recording:
                self._start_recording()
            elif not target and self.recording:
                self._stop_recording()

    def _start_recording(self):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.output_root, stamp)
        # Create the whole tree, including any missing parents
        os.makedirs(os.path.join(self.session_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, 'depth'), exist_ok=True)

        self.poses_file = open(
            os.path.join(self.session_dir, 'poses.txt'), 'w')
        self.poses_file.write('# timestamp tx ty tz qx qy qz qw '
                              f'({self.base_frame} -> camera)\n')
        self.assoc_file = open(
            os.path.join(self.session_dir, 'associations.txt'), 'w')

        self.frame_count = 0
        self.metadata_written = False
        self.recording = True
        msg = f'Recording started -> {self.session_dir}'
        self.get_logger().info(msg)
        self.bridge_qt.log_message.emit(msg)

    def _stop_recording(self):
        self.recording = False
        for f in (self.poses_file, self.assoc_file):
            if f:
                f.close()
        self.poses_file = self.assoc_file = None
        msg = (f'Recording stopped. {self.frame_count} frames '
               f'saved to {self.session_dir}')
        self.get_logger().info(msg)
        self.bridge_qt.log_message.emit(msg)

    def stop_if_recording(self):
        with self._lock:
            if self.recording:
                self._stop_recording()

    def write_metadata(self, color_fmt: str, depth_fmt: str):
        intr = {}
        if self.color_info:
            intr['color'] = self.camera_info_to_dict(self.color_info)
        if self.depth_info:
            intr['depth'] = self.camera_info_to_dict(self.depth_info)
        intr['color_compressed_format'] = color_fmt
        intr['depth_compressed_format'] = depth_fmt
        intr['depth_unit'] = 'millimetres (16UC1)'
        with open(os.path.join(self.session_dir, 'intrinsics.json'), 'w') as f:
            json.dump(intr, f, indent=2)

        if self.depth_to_color is not None:
            e = self.depth_to_color
            extr = {
                'depth_to_color': {
                    'rotation': list(e.rotation),        # 3x3 row-major
                    'translation': list(e.translation),  # metres
                }
            }
            with open(os.path.join(self.session_dir,
                                   'extrinsics.json'), 'w') as f:
                json.dump(extr, f, indent=2)

        self.metadata_written = True

    def lookup_pose(self, camera_frame, stamp):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, camera_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=0.05))
            t, q = tf.transform.translation, tf.transform.rotation
            return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warning(f'TF lookup failed: {e}',
                                   throttle_duration_sec=2.0)
            return None

    # ------------------------------------------------------------------ #
    #  Main synchronized callback (ROS executor thread)                  #
    # ------------------------------------------------------------------ #
    def rgbd_callback(self, color_msg: CompressedImage,
                      depth_msg: CompressedImage):
        # ---- One-time format validation ----
        if not self.formats_validated:
            if not self._validate_formats(color_msg, depth_msg):
                return          # fatal_error signal already emitted

        # ---- Export: raw bytes, no conversion ----
        with self._lock:
            if self.recording:
                self._save_frame(color_msg, depth_msg)
            rec, count = self.recording, self.frame_count

        # ---- Preview: decode only for display ----
        color = cv2.imdecode(
            np.frombuffer(color_msg.data, np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(
            np.frombuffer(depth_msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            self.get_logger().warning('Failed to decode compressed image '
                                   'for preview', throttle_duration_sec=2.0)
            return
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)

        # Hand off to the GUI thread (queued signal, thread-safe)
        self.bridge_qt.frame_ready.emit(color, depth, rec, count)

    def _save_frame(self, color_msg, depth_msg):
        if not self.metadata_written and (self.color_info or self.depth_info):
            self.write_metadata(color_msg.format, depth_msg.format)

        t_color = stamp_to_sec(color_msg.header.stamp)
        t_depth = stamp_to_sec(depth_msg.header.stamp)
        rgb_name = f'rgb/{t_color:.6f}.{self.color_ext}'
        depth_name = f'depth/{t_depth:.6f}.{self.depth_ext}'

        # Write the compressed payload byte-for-byte — no decode/re-encode.
        with open(os.path.join(self.session_dir, rgb_name), 'wb') as f:
            f.write(bytes(color_msg.data))
        with open(os.path.join(self.session_dir, depth_name), 'wb') as f:
            f.write(bytes(depth_msg.data))

        self.assoc_file.write(
            f'{t_color:.6f} {rgb_name} {t_depth:.6f} {depth_name}\n')

        cam_frame = self.camera_frame or color_msg.header.frame_id
        pose = self.lookup_pose(cam_frame, color_msg.header.stamp)
        if pose:
            (tx, ty, tz), (qx, qy, qz, qw) = pose
            self.poses_file.write(
                f'{t_color:.6f} {tx:.6f} {ty:.6f} {tz:.6f} '
                f'{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n')

        self.frame_count += 1


# ─────────────────────────────────────────────────────────────────────────────
#  Qt6 main window
# ─────────────────────────────────────────────────────────────────────────────
class RecorderWindow(QMainWindow):
    def __init__(self, node: RGBDRecorder, bridge: RosQtBridge):
        super().__init__()
        self.node = node
        self.recording = False
        self.blink_on = False

        self.setWindowTitle('RGB-D Recorder — waiting for images…')
        self.resize(1280, 540)

        # --- image view ---
        self.image_label = QLabel('Waiting for colour + depth images…')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            'background-color: #202020; color: #aaaaaa; font-size: 16px;')
        self.image_label.setMinimumSize(640, 240)

        # --- header bar: REC indicator + info ---
        self.rec_dot = QLabel('  ')
        self.rec_dot.setFixedSize(22, 22)
        self._set_dot(False)

        self.rec_text = QLabel('IDLE — press SPACE to record')
        f = QFont()
        f.setPointSize(12)
        f.setBold(True)
        self.rec_text.setFont(f)

        header = QHBoxLayout()
        header.addWidget(self.rec_dot)
        header.addWidget(self.rec_text)
        header.addStretch()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(header)
        layout.addWidget(self.image_label, stretch=1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            'SPACE: start/stop recording   |   Q / ESC: quit')

        # --- signals from ROS thread (queued connections) ---
        bridge.frame_ready.connect(self.on_frame)
        bridge.log_message.connect(self.status.showMessage)
        bridge.fatal_error.connect(self.on_fatal_error)

        # --- blink timer for REC dot ---
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._blink)
        self.blink_timer.start(500)

        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ #
    def _set_dot(self, on: bool):
        color = '#ff2222' if on else '#404040'
        self.rec_dot.setStyleSheet(
            f'background-color: {color}; border-radius: 11px;')

    def _blink(self):
        if self.recording:
            self.blink_on = not self.blink_on
            self._set_dot(self.blink_on)
        else:
            self._set_dot(False)

    # ------------------------------------------------------------------ #
    #  Fatal error slot — runs in the GUI thread                         #
    # ------------------------------------------------------------------ #
    def on_fatal_error(self, msg):
        self.node.stop_if_recording()
        QMessageBox.critical(self, 'RGB-D Recorder — fatal error', msg)
        QApplication.exit(1)

    # ------------------------------------------------------------------ #
    #  Frame slot — runs in the GUI thread                               #
    # ------------------------------------------------------------------ #
    def on_frame(self, color, depth_mm, recording, frame_count):
        self.recording = recording

        # depth -> colormap
        depth_vis = cv2.convertScaleAbs(
            depth_mm, alpha=255.0 / self.node.get_parameter(
                'depth_vis_max_mm').value)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        if depth_vis.shape[:2] != color.shape[:2]:
            depth_vis = cv2.resize(depth_vis,
                                   (color.shape[1], color.shape[0]))

        canvas = np.ascontiguousarray(np.hstack([color, depth_vis]))
        h, w, _ = canvas.shape
        qimg = QImage(canvas.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

        # recording indication
        if recording:
            self.rec_text.setText(f'● RECORDING — {frame_count} frames')
            self.rec_text.setStyleSheet('color: #ff2222;')
            self.image_label.setStyleSheet(
                'background-color: #202020; border: 3px solid #ff2222;')
            self.setWindowTitle(
                f'RGB-D Recorder — ● REC ({frame_count} frames)')
        else:
            self.rec_text.setText('IDLE — press SPACE to record')
            self.rec_text.setStyleSheet('color: #cccccc;')
            self.image_label.setStyleSheet(
                'background-color: #202020; border: none;')
            self.setWindowTitle('RGB-D Recorder — idle')

    # ------------------------------------------------------------------ #
    #  Keyboard                                                          #
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self.node.toggle_recording()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.node.stop_if_recording()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)

    app = QApplication(sys.argv)
    bridge = RosQtBridge()
    node = RGBDRecorder(bridge)

    # Spin ROS in a background thread; Qt owns the main thread.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    window = RecorderWindow(node, bridge)
    window.show()

    exit_code = app.exec()

    node.stop_if_recording()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    ros_thread.join(timeout=2.0)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
