#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class HeadTFNode(Node):
    def __init__(self):
        super().__init__('head_tf_node')
        # Frames
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        # Pose camera so với base (m, rad) – đổi live bằng ros2 param
        self.declare_parameter('x', 0.10)
        self.declare_parameter('y', 0.00)
        self.declare_parameter('z', 0.30)   # cao ~30 cm như bạn nói
        self.declare_parameter('roll',  0.0)
        self.declare_parameter('pitch', 0.0)  # cúi xuống: âm (vd -0.35 ~ -20°)
        self.declare_parameter('yaw',   0.0)

        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(1/50.0, self.publish_tf)  # 50 Hz
        self.add_on_set_parameters_callback(lambda _: __import__('rcl_interfaces.msg',fromlist=['SetParametersResult']).SetParametersResult(successful=True))

    def publish_tf(self):
        p = {pp.name: pp.value for pp in self.get_parameters(
            ['base_frame','camera_frame','x','y','z','roll','pitch','yaw'])}
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = p['base_frame']
        t.child_frame_id  = p['camera_frame']
        t.transform.translation.x = float(p['x'])
        t.transform.translation.y = float(p['y'])
        t.transform.translation.z = float(p['z'])
        cr, sr = math.cos(p['roll']/2),  math.sin(p['roll']/2)
        cp, sp = math.cos(p['pitch']/2), math.sin(p['pitch']/2)
        cy, sy = math.cos(p['yaw']/2),   math.sin(p['yaw']/2)
        t.transform.rotation.w = cr*cp*cy + sr*sp*sy
        t.transform.rotation.x = sr*cp*cy - cr*sp*sy
        t.transform.rotation.y = cr*sp*cy + sr*cp*sy
        t.transform.rotation.z = cr*cp*sy - sr*sp*cy
        self.br.sendTransform(t)

def main():
    rclpy.init(); rclpy.spin(HeadTFNode()); rclpy.shutdown()
if __name__ == '__main__':
    main()
