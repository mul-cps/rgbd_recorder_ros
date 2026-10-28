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
- Reception watchdog: shows a GUI warning banner if no images arrive
  within `watchdog_timeout` seconds — from startup and during operation.
- SPACE toggles sequence recording; C saves one synchronized RGB-D pair; Q/ESC quits.

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
import queue
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
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
    data_timeout = Signal(bool, float)   # (timed_out, secs since last image)


# ─────────────────────────────────────────────────────────────────────────────
#  ROS node
# ─────────────────────────────────────────────────────────────────────────────
class RGBDRecorder(Node):
    def __init__(self, bridge: RosQtBridge):
        super().__init__('rgbd_recorder')
        self.bridge_qt = bridge

        # ---------------- Parameters ----------------
        self.base_frame = self.declare_parameter('base_frame', 'map').value
        self.camera_frame = self.declare_parameter('camera_frame', '').value   # empty -> header frame_id
        self.output_dir = self.declare_parameter('output_dir', '~/rgbd_dataset').value
        self.sync_slop = self.declare_parameter('sync_slop', 0.01).value
        self.depth_vis_max_mm = self.declare_parameter('depth_vis_max_mm', 5000.0).value
        self.watchdog_timeout = self.declare_parameter('watchdog_timeout', 5.0).value

        # gp = self.get_parameter
        # Expand ~ and env vars, create the root folder right away
        self.output_root = os.path.expandvars(os.path.expanduser(self.output_dir))
        os.makedirs(self.output_root, exist_ok=True)

        # One session per process invocation, shared by sequences and captures.
        session_stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.session_dir = os.path.join(self.output_root, session_stamp)
        os.makedirs(os.path.join(self.session_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, 'depth'), exist_ok=True)

        # ---------------- State ----------------
        self._lock = threading.Lock()
        self._metadata_file_lock = threading.Lock()
        self.recording = False
        self._single_capture_requested = False
        self.poses_file = None
        self.assoc_file = None
        self.frame_count = 0
        self.color_info = None
        self.depth_info = None
        self.depth_to_color = None      # orbbec extrinsics msg
        self.metadata_written = False

        # Recording I/O is performed by a dedicated worker thread.
        # The bounded queue prevents unbounded memory growth if the
        # storage device cannot sustain the incoming data rate.
        self._write_queue = None
        self._writer_thread = None
        self._write_queue_size = 100
        self._dropped_frames = 0

        # Format validation (done once, on the first synchronized frame)
        self.formats_validated = False
        self.color_ext = None
        self.depth_ext = None

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- Subscribers ----------------
        self.color_topic = '/camera/color/image_raw/compressed'
        self.depth_topic = '/camera/depth/image_raw/compressed'
        self.color_info_topic = '/camera/color/camera_info'
        self.depth_info_topic = '/camera/depth/camera_info'
        self.extrinsics_topic = '/camera/depth_to_color'

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # message_filters is NOT thread-safe: force colour/depth callbacks
        # to never run concurrently, even under MultiThreadedExecutor.
        self.sync_group = MutuallyExclusiveCallbackGroup()

        # Reception diagnostics
        self.n_color = 0
        self.n_depth = 0
        self.n_sync = 0
        self._last_color_stamp = 0.0
        self._last_depth_stamp = 0.0

        self.color_sub = Subscriber(
            self, CompressedImage, self.color_topic,
            qos_profile=qos, callback_group=self.sync_group)
        self.depth_sub = Subscriber(
            self, CompressedImage, self.depth_topic,
            qos_profile=qos, callback_group=self.sync_group)

        # Count raw arrivals without disturbing the synchronizer
        self.color_sub.registerCallback(self._count_color)
        self.depth_sub.registerCallback(self._count_depth)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10, slop=self.sync_slop)
        self.sync.registerCallback(self.rgbd_callback)

        self.create_subscription(CameraInfo, self.color_info_topic,
                                 self.color_info_cb, qos)
        self.create_subscription(CameraInfo, self.depth_info_topic,
                                 self.depth_info_cb, qos)

        if HAVE_ORBBEC_MSGS:
            self.create_subscription(
                OrbbecExtrinsics, self.extrinsics_topic,
                self.extrinsics_cb, qos)
        else:
            self.get_logger().warning(
                'orbbec_camera_msgs not available — depth->color extrinsics '
                'topic will not be recorded (TF poses still are).')

        self.create_service(SetBool, '~/set_recording', self.set_recording_cb)

        # Periodic diagnostics: tells you exactly where messages stop
        self.diag_timer = self.create_timer(3.0, self._diag)

        # ---------------- Reception watchdog ----------------
        # Initialise to "now" so the watchdog also triggers if NOTHING
        # ever arrives after startup.
        self._last_rx_time_color = self.get_clock().now()
        self._last_rx_time_depth = self.get_clock().now()
        self._timeout_active = False
        # Check at 1 Hz — a fired warning is refreshed every second with
        # the updated outage duration.
        self.watchdog_timer = self.create_timer(1.0, self._watchdog_check)

        self.get_logger().info(
            f'RGB-D recorder ready (compressed transport). '
            f'Output root: {self.output_root}')

    # ------------------------------------------------------------------ #
    #  Reception diagnostics + watchdog                                  #
    # ------------------------------------------------------------------ #
    def _count_color(self, msg):
        self.n_color += 1
        self._last_color_stamp = stamp_to_sec(msg.header.stamp)
        self._last_rx_time_color = self.get_clock().now()

    def _count_depth(self, msg):
        self.n_depth += 1
        self._last_depth_stamp = stamp_to_sec(msg.header.stamp)
        self._last_rx_time_depth = self.get_clock().now()

    def _diag(self):
        dt = None
        if self.n_color and self.n_depth:
            dt = abs(self._last_color_stamp - self._last_depth_stamp)
        self.get_logger().debug(
            f'rx color={self.n_color} depth={self.n_depth} '
            f'synced={self.n_sync}'
            + (f'  |stamp diff|={dt:.3f}s' if dt is not None else ''))

    def _watchdog_check(self):
        now = self.get_clock().now()
        elapsed_color = (now - self._last_rx_time_color).nanoseconds * 1e-9
        elapsed_depth = (now - self._last_rx_time_depth).nanoseconds * 1e-9

        if (elapsed_color >= self.watchdog_timeout) or (elapsed_depth >= self.watchdog_timeout):
            # (Re-)emit every second so the GUI can show a live counter
            # and so the log reflects an ongoing outage.
            self._timeout_active = True
            self.get_logger().warning(
                f'No images received for {self.watchdog_timeout:.0f} s '
                f'(last color {elapsed_color:.0f} s ago, last depth {elapsed_depth:.0f} s ago). '
                f'Camera disconnected or topics not publishing?',
                throttle_duration_sec=5.0)
            self.bridge_qt.data_timeout.emit(True, self.watchdog_timeout)
        elif self._timeout_active:
            # Data resumed — clear the warning exactly once.
            self._timeout_active = False
            self.get_logger().info('Image reception resumed.')
            self.bridge_qt.data_timeout.emit(False, 0.0)

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

    def request_single_capture(self):
        """Save the next synchronized RGB-D pair as an individual capture."""
        with self._lock:
            self._single_capture_requested = True
        msg = 'Single RGB-D capture requested; waiting for next synchronized pair.'
        self.get_logger().info(msg)
        self.bridge_qt.log_message.emit(msg)

    def _save_single_frame(self, color_msg, depth_msg):
        """Save one RGB-D pair in the current session directory."""
        t_color = stamp_to_sec(color_msg.header.stamp)
        t_depth = stamp_to_sec(depth_msg.header.stamp)
        rgb_file = f'{t_color:.6f}.{self.color_ext}'
        depth_file = f'{t_depth:.6f}.{self.depth_ext}'
        rgb_name = f'rgb/{rgb_file}'
        depth_name = f'depth/{depth_file}'

        with open(os.path.join(self.session_dir, rgb_name), 'wb') as f:
            f.write(bytes(color_msg.data))
        with open(os.path.join(self.session_dir, depth_name), 'wb') as f:
            f.write(bytes(depth_msg.data))

        # Camera calibration is shared by all data in this session.
        intr = {}
        if self.color_info:
            intr['color'] = self.camera_info_to_dict(self.color_info)
        if self.depth_info:
            intr['depth'] = self.camera_info_to_dict(self.depth_info)
        intr['color_compressed_format'] = self.color_ext
        intr['depth_compressed_format'] = self.depth_ext
        intr['depth_unit'] = 'millimetres (16UC1)'
        with open(os.path.join(self.session_dir, 'intrinsics.json'), 'w') as f:
            json.dump(intr, f, indent=2)

        if self.depth_to_color is not None:
            e = self.depth_to_color
            extr = {'depth_to_color': {
                'rotation': list(e.rotation),
                'translation': list(e.translation),
            }}
            with open(os.path.join(self.session_dir, 'extrinsics.json'), 'w') as f:
                json.dump(extr, f, indent=2)

        cam_frame = self.camera_frame or color_msg.header.frame_id
        pose = self.lookup_pose(cam_frame, color_msg.header.stamp)

        # Append captures to the same metadata files used for sequences.
        # Serialize these writes with the sequence writer to avoid concurrent
        # access and flush sequence writes before appending from this thread.
        poses_path = os.path.join(self.session_dir, 'poses.txt')
        associations_path = os.path.join(self.session_dir, 'associations.txt')
        with self._metadata_file_lock:
            if self.assoc_file is not None:
                self.assoc_file.flush()
            if self.poses_file is not None:
                self.poses_file.flush()

            with open(associations_path, 'a') as f:
                f.write(f'{t_color:.6f} {rgb_name} {t_depth:.6f} {depth_name}\n')

            poses_needs_header = (
                not os.path.exists(poses_path) or os.path.getsize(poses_path) == 0)
            with open(poses_path, 'a') as f:
                if poses_needs_header:
                    f.write('# timestamp tx ty tz qx qy qz qw '
                            f'({self.base_frame} -> camera)\n')
                if pose:
                    (tx, ty, tz), (qx, qy, qz, qw) = pose
                    f.write(f'{t_color:.6f} {tx:.6f} {ty:.6f} {tz:.6f} '
                            f'{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n')

        msg = f'Single RGB-D pair saved -> {self.session_dir}'
        self.get_logger().info(msg)
        self.bridge_qt.log_message.emit(msg)

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
        # Reuse the session created at node startup. Append so starting a
        # second sequence does not erase the first sequence's metadata.
        poses_path = os.path.join(self.session_dir, 'poses.txt')
        associations_path = os.path.join(self.session_dir, 'associations.txt')
        poses_needs_header = (
            not os.path.exists(poses_path) or os.path.getsize(poses_path) == 0)
        self.poses_file = open(poses_path, 'a')
        if poses_needs_header:
            self.poses_file.write('# timestamp tx ty tz qx qy qz qw '
                                  f'({self.base_frame} -> camera)\\n')
        self.assoc_file = open(associations_path, 'a')

        self.frame_count = 0
        self.metadata_written = False
        self._dropped_frames = 0
        self._write_queue = queue.Queue(
            maxsize=self._write_queue_size)

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name='rgbd-writer',
            daemon=True)
        self._writer_thread.start()

        self.recording = True
        msg = f'Recording started -> {self.session_dir}'
        self.get_logger().info(msg)
        self.bridge_qt.log_message.emit(msg)

    def _stop_recording(self):
        # Stop accepting new frames. The caller holds self._lock, so
        # rgbd_callback cannot enqueue more frames while we drain the queue.
        self.recording = False

        if self._write_queue is not None:
            # FIFO sentinel: queued frames are processed before exit.
            self._write_queue.put(None)

        if self._writer_thread is not None:
            self._writer_thread.join()

        with self._metadata_file_lock:
            for f in (self.poses_file, self.assoc_file):
                if f:
                    f.close()
            self.poses_file = self.assoc_file = None
        self._write_queue = None
        self._writer_thread = None

        msg = (f'Recording stopped. {self.frame_count} frames '
               f'saved to {self.session_dir}')
        if self._dropped_frames:
            msg += (
                f' ({self._dropped_frames} RGB-D pairs dropped: '
                f'queue full)')
        self.get_logger().info(msg)
        self.bridge_qt.log_message.emit(msg)

    def stop_if_recording(self):
        with self._lock:
            if self.recording:
                self._stop_recording()

    def _writer_loop(self):
        """Write queued RGB-D pairs without blocking ROS image callbacks."""
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return

                color_msg, depth_msg = item
                self._save_frame(color_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(
                    f'Failed to save RGB-D frame: {e}')
            finally:
                self._write_queue.task_done()

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
        self.n_sync += 1

        # ---- One-time format validation ----
        if not self.formats_validated:
            if not self._validate_formats(color_msg, depth_msg):
                return          # fatal_error signal already emitted

        # ---- Export: enqueue raw messages; never do disk I/O here ----
        with self._lock:
            rec = self.recording
            capture_single = self._single_capture_requested
            if capture_single:
                self._single_capture_requested = False
            if rec and self._write_queue is not None:
                try:
                    self._write_queue.put_nowait(
                        (color_msg, depth_msg))
                except queue.Full:
                    self._dropped_frames += 1
                    if (self._dropped_frames == 1 or
                            self._dropped_frames % 100 == 0):
                        self.get_logger().warning(
                            f'Recording queue full; dropped '
                            f'{self._dropped_frames} RGB-D pairs.')
            count = self.frame_count

        if capture_single:
            threading.Thread(
                target=self._save_single_frame,
                args=(color_msg, depth_msg),
                name='rgbd-single-capture',
                daemon=True).start()

        # ---- Preview: decode only for display ----
        color = cv2.imdecode(
            np.frombuffer(color_msg.data, np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(
            np.frombuffer(depth_msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if color is None:
            self.get_logger().warning(
                'Failed to decode compressed color image',
                throttle_duration_sec=2.0)
            return
        if depth is None:
            self.get_logger().warning(
                'Failed to decode compressed depth image',
                throttle_duration_sec=2.0)
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

        cam_frame = self.camera_frame or color_msg.header.frame_id
        pose = self.lookup_pose(cam_frame, color_msg.header.stamp)
        with self._metadata_file_lock:
            self.assoc_file.write(
                f'{t_color:.6f} {rgb_name} {t_depth:.6f} {depth_name}\n')
            self.assoc_file.flush()
            if pose:
                (tx, ty, tz), (qx, qy, qz, qw) = pose
                self.poses_file.write(
                    f'{t_color:.6f} {tx:.6f} {ty:.6f} {tz:.6f} '
                    f'{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n')
                self.poses_file.flush()

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
        self._fatal_error_shown = False

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

        self.rec_text = QLabel('IDLE — SPACE: record sequence | C: capture pair')
        f = QFont()
        f.setPointSize(12)
        f.setBold(True)
        self.rec_text.setFont(f)

        header = QHBoxLayout()
        header.addWidget(self.rec_dot)
        header.addWidget(self.rec_text)
        header.addStretch()

        # --- data timeout warning banner ---
        self.warn_banner = QLabel()
        self.warn_banner.setAlignment(Qt.AlignCenter)
        self.warn_banner.setStyleSheet(
            'background-color: #aa6600; color: white; '
            'font-size: 14px; font-weight: bold; padding: 6px;')
        self.warn_banner.hide()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(header)
        layout.addWidget(self.warn_banner)
        layout.addWidget(self.image_label, stretch=1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            'SPACE: start/stop sequence   |   C: save one RGB-D pair   |   Q / ESC: quit')

        # --- signals from ROS thread (queued connections) ---
        bridge.frame_ready.connect(self.on_frame)
        bridge.log_message.connect(self.status.showMessage)
        bridge.fatal_error.connect(self.on_fatal_error)
        bridge.data_timeout.connect(self.on_data_timeout)

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
    #  Data timeout slot — runs in the GUI thread                        #
    # ------------------------------------------------------------------ #
    def on_data_timeout(self, timed_out: bool, elapsed: float):
        if timed_out:
            self.warn_banner.setText(
                f'⚠  No images received for {elapsed:.0f} s — '
                f'check camera connection / topics')
            self.warn_banner.show()
            if not self.recording:
                self.image_label.setStyleSheet(
                    'background-color: #202020; color: #aaaaaa; '
                    'border: 3px solid #aa6600; font-size: 16px;')
        else:
            self.warn_banner.hide()
            self.status.showMessage('Image reception resumed.', 5000)
            if not self.recording:
                self.image_label.setStyleSheet(
                    'background-color: #202020; color: #aaaaaa; '
                    'font-size: 16px;')

    # ------------------------------------------------------------------ #
    #  Fatal error slot — runs in the GUI thread                         #
    # ------------------------------------------------------------------ #
    def on_fatal_error(self, msg):
        if self._fatal_error_shown:
            return

        self._fatal_error_shown = True

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
            depth_mm, alpha=255.0 / self.node.depth_vis_max_mm)
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
            self.rec_text.setText('IDLE — SPACE: record sequence | C: capture pair')
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
        elif key == Qt.Key_C:
            self.node.request_single_capture()
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
