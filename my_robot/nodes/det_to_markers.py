#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_ros import TransformException

class DetToMarkers(Node):
    def __init__(self):
        super().__init__('det_to_markers')

        # ---- Params (giữ tối thiểu, phù hợp RViz hiện tại) ----
        self.declare_parameter('detections_topic', '/camera/yolo/detections')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')  # dùng aligned depth cho chuẩn
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_frame', 'map')

        # Topic + namespace theo RViz bạn đang dùng
        self.declare_parameter('marker_topic', '/labels')  # <-- xuất ra /labels
        self.declare_parameter('text_ns', 'labels')        # <-- chữ ở NS "labels"
        self.declare_parameter('marker_ns', 'ids')         # <-- chấm ở NS "ids"

        # Style
        self.declare_parameter('text_scale', 0.20)     # chiều cao chữ (m)
        self.declare_parameter('sphere_scale', 0.08)   # đường kính cầu (m)
        self.declare_parameter('z_text_offset', 0.15)  # chữ nhô lên so với điểm (m)

        self.det_topic      = self.get_parameter('detections_topic').value
        self.depth_topic    = self.get_parameter('depth_topic').value
        self.info_topic     = self.get_parameter('camera_info_topic').value
        self.target_frame   = self.get_parameter('target_frame').value
        self.marker_topic   = self.get_parameter('marker_topic').value
        self.text_ns        = self.get_parameter('text_ns').value
        self.marker_ns      = self.get_parameter('marker_ns').value
        self.text_scale     = float(self.get_parameter('text_scale').value)
        self.sphere_scale   = float(self.get_parameter('sphere_scale').value)
        self.z_text_offset  = float(self.get_parameter('z_text_offset').value)

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        self.have_cam_model = False
        self.last_depth: Optional[Image] = None
        self.depth_frame_id: Optional[str] = None

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs/Pubs
        self.create_subscription(Detection2DArray, self.det_topic, self.on_detections, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.info_topic, self.on_info, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.get_logger().info(
            f"[det_to_markers] det={self.det_topic}, depth={self.depth_topic}, "
            f"info={self.info_topic}, target={self.target_frame}, out={self.marker_topic}"
        )

    # --------- Callbacks ----------
    def on_info(self, msg: CameraInfo):
        if not self.have_cam_model:
            self.cam_model.fromCameraInfo(msg)
            self.have_cam_model = True
            self.get_logger().info("Camera model ready")

    def on_depth(self, msg: Image):
        self.last_depth = msg
        self.depth_frame_id = msg.header.frame_id

    def on_detections(self, msg: Detection2DArray):
        if not (self.have_cam_model and self.last_depth is not None):
            return

        depth_img = self.bridge.imgmsg_to_cv2(self.last_depth)
        depth_m = self._depth_to_meters(depth_img, self.last_depth.encoding)

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # TF: camera -> target_frame (theo timestamp của ảnh depth)
        try:
            stamp = rclpy.time.Time.from_msg(self.last_depth.header.stamp)
            tf = self.tf_buffer.lookup_transform(self.target_frame, self.depth_frame_id, stamp)
        except TransformException as e:
            self.get_logger().warn(f"Missing TF {self.depth_frame_id}->{self.target_frame}: {e}")
            return

        T = self._tf_to_mat(tf)

        for i, det in enumerate(msg.detections):
            if not det.results:
                continue

            # class_id từ YOLO đã là tên lớp (nếu node YOLO điền names[])
            label = det.results[0].hypothesis.class_id or "obj"

            cx = int(det.bbox.center.x)
            cy = int(det.bbox.center.y)

            z = self._median_depth(depth_m, cx, cy, k=11)  # median cửa sổ lớn hơn cho ổn định
            if not np.isfinite(z) or z <= 0.0:
                continue

            X, Y, Z = self._project_to_3d(cx, cy, z)
            p_cam = np.array([X, Y, Z, 1.0], dtype=float)
            p_map = (T @ p_cam)[:3]

            # Sphere marker (ids)
            mk = Marker()
            mk.header.stamp = now
            mk.header.frame_id = self.target_frame
            mk.ns = self.marker_ns
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position = Point(x=float(p_map[0]), y=float(p_map[1]), z=float(p_map[2]))
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = self.sphere_scale
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.9, 0.1, 1.0
            mk.lifetime.sec = 0
            mk.lifetime.nanosec = int(0.2e9)  # 0.2s
            markers.markers.append(mk)

            # Text marker (labels)
            tx = Marker()
            tx.header.stamp = now
            tx.header.frame_id = self.target_frame
            tx.ns = self.text_ns
            tx.id = 1000 + i
            tx.type = Marker.TEXT_VIEW_FACING
            tx.action = Marker.ADD
            tx.text = label
            tx.pose.position = Point(
                x=float(p_map[0]), y=float(p_map[1]), z=float(p_map[2] + self.z_text_offset)
            )
            tx.pose.orientation.w = 1.0
            tx.scale.z = self.text_scale
            tx.color.r, tx.color.g, tx.color.b, tx.color.a = 1.0, 0.9, 0.1, 1.0
            tx.lifetime.sec = 0
            tx.lifetime.nanosec = int(0.2e9)
            markers.markers.append(tx)

        if markers.markers:
            self.pub_markers.publish(markers)

    # --------- Helpers ----------
    def _depth_to_meters(self, depth, enc: str):
        if enc in ('16UC1', 'mono16'):
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def _median_depth(self, depth_m: np.ndarray, x: int, y: int, k: int = 5) -> float:
        h, w = depth_m.shape[:2]
        x0, y0 = max(0, x - k//2), max(0, y - k//2)
        x1, y1 = min(w, x0 + k), min(h, y0 + k)
        roi = depth_m[y0:y1, x0:x1]
        roi = roi[np.isfinite(roi) & (roi > 0.0)]
        return float(np.median(roi)) if roi.size else float('nan')

    def _project_to_3d(self, u: int, v: int, z: float) -> Tuple[float, float, float]:
        fx, fy, cx, cy = self.cam_model.fx(), self.cam_model.fy(), self.cam_model.cx(), self.cam_model.cy()
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        return float(X), float(Y), float(z)

    def _tf_to_mat(self, tf):
        t = tf.transform.translation
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
        ], dtype=float)
        T = np.eye(4, dtype=float); T[:3,:3] = R; T[:3,3] = [t.x, t.y, t.z]
        return T

def main():
    rclpy.init()
    node = DetToMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
