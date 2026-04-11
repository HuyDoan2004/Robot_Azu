#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 node: imu_to_tf
- Subscribes to sensor_msgs/Imu (default: /camera/imu)
- Broadcasts TF (default: map -> base_link) using IMU orientation
- Optional optical-frame offset for camera IMUs (Rz(-90°) * Rx(-90°))
- Optional static translation offsets (meters) and Euler offsets (deg)

Params:
  imu_topic (string, default: /camera/imu)
  parent_frame (string, default: map)
  child_frame (string, default: base_link)
  use_optical_offset (bool, default: true)
  roll_offset_deg (double, default: 0.0)
  pitch_offset_deg (double, default: 0.0)
  yaw_offset_deg (double, default: 0.0)
  tx (double, default: 0.0)
  ty (double, default: 0.0)
  tz (double, default: 0.0)

Usage example:
  ros2 run my_robot imu_to_tf --ros-args -p parent_frame:=map -p child_frame:=base_link -p imu_topic:=/camera/imu

Notes:
  - This node only uses IMU orientation. For full pose fusion, use robot_localization.
"""

import math
from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

def euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert RPY (rad) -> quaternion (x,y,z,w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return (x, y, z, w)

def quat_multiply(q1, q2):
    """Hamilton product (x,y,z,w)."""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return (x,y,z,w)

def normalize_quat(q):
    x,y,z,w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/n, y/n, z/n, w/n)

class ImuToTF(Node):
    def __init__(self):
        super().__init__('imu_to_tf')
        # Parameters
        self.declare_parameter('imu_topic', '/camera/imu')
        self.declare_parameter('parent_frame', 'map')
        self.declare_parameter('child_frame', 'base_link')
        self.declare_parameter('use_optical_offset', True)
        self.declare_parameter('roll_offset_deg', 0.0)
        self.declare_parameter('pitch_offset_deg', 0.0)
        self.declare_parameter('yaw_offset_deg', 0.0)
        self.declare_parameter('tx', 0.0)
        self.declare_parameter('ty', 0.0)
        self.declare_parameter('tz', 0.0)

        self.parent = self.get_parameter('parent_frame').get_parameter_value().string_value
        self.child  = self.get_parameter('child_frame').get_parameter_value().string_value
        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.use_optical = self.get_parameter('use_optical_offset').get_parameter_value().bool_value
        self.roll_off = math.radians(self.get_parameter('roll_offset_deg').get_parameter_value().double_value)
        self.pitch_off = math.radians(self.get_parameter('pitch_offset_deg').get_parameter_value().double_value)
        self.yaw_off = math.radians(self.get_parameter('yaw_offset_deg').get_parameter_value().double_value)
        self.tx = self.get_parameter('tx').get_parameter_value().double_value
        self.ty = self.get_parameter('ty').get_parameter_value().double_value
        self.tz = self.get_parameter('tz').get_parameter_value().double_value

        # Precompute offsets
        # Optical offset: Rz(-90°) * Rx(-90°) converts ROS ENU to optical (X right, Y down, Z forward)
        # We want the inverse to go from optical -> base ENU, so use Rx(+90) * Rz(+90)
        rz_p90 = euler_to_quat(0.0, 0.0, math.radians(90.0))
        rx_p90 = euler_to_quat(math.radians(90.0), 0.0, 0.0)
        self.q_optical_inv = quat_multiply(rx_p90, rz_p90)  # inverse of Rz(-90)*Rx(-90)

        self.q_user_off = euler_to_quat(self.roll_off, self.pitch_off, self.yaw_off)

        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Imu, self.imu_topic, self.cb, 50)
        self.get_logger().info(f"imu_to_tf: listening {self.imu_topic}, broadcasting {self.parent} -> {self.child}")

    def cb(self, msg: Imu):
        q = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        q = normalize_quat(q)

        # Apply offsets (order matters: base rotation = IMU * optical_inv * user_off)
        if self.use_optical:
            q = quat_multiply(q, self.q_optical_inv)
        q = quat_multiply(q, self.q_user_off)
        q = normalize_quat(q)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp if (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0) else self.get_clock().now().to_msg()
        t.header.frame_id = self.parent
        t.child_frame_id = self.child
        t.transform.translation.x = float(self.tx)
        t.transform.translation.y = float(self.ty)
        t.transform.translation.z = float(self.tz)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = ImuToTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
