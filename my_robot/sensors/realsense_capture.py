import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Imu

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


class RealSenseCapture:
    """
    Đọc RGB, Depth và IMU (gyro + accel) từ Intel RealSense (ví dụ D435i).
    API:
      start() -> None
      read()  -> (ok, rgb[bgr8], depth[16UC1], imu_msg: Imu|None)
      stop()  -> None
    """
    def __init__(self, node: Node, fps=30):
        self.node = node
        self.fps = fps
        self.pipe = None
        self.align = None
        self._last_gyro = None   # (x,y,z) rad/s
        self._last_accel = None  # (x,y,z) m/s^2

    def start(self):
        if rs is None:
            self.node.get_logger().warn('pyrealsense2 không sẵn có; dùng frame giả.')
            return
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)
        # Bật IMU (nếu hỗ trợ)
        try:
            cfg.enable_stream(rs.stream.gyro)
            cfg.enable_stream(rs.stream.accel)
        except Exception:
            self.node.get_logger().warn('Không bật được IMU stream trên RealSense.')

        self.pipe = rs.pipeline()
        self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)

    def _pack_imu(self, stamp_sec):
        if self._last_gyro is None and self._last_accel is None:
            return None
        msg = Imu()
        # stamp lấy từ clock ROS để đồng bộ các publisher
        from builtin_interfaces.msg import Time
        sec = int(stamp_sec)
        nanosec = int((stamp_sec - sec) * 1e9)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = 'camera_imu_optical_frame'

        if self._last_gyro is not None:
            gx, gy, gz = self._last_gyro
            msg.angular_velocity.x = float(gx)
            msg.angular_velocity.y = float(gy)
            msg.angular_velocity.z = float(gz)
        if self._last_accel is not None:
            ax, ay, az = self._last_accel
            msg.linear_acceleration.x = float(ax)
            msg.linear_acceleration.y = float(ay)
            msg.linear_acceleration.z = float(az)
        # orientation sẽ do bộ lọc ước lượng -> giữ mặc định (0) ở đây
        return msg

    def read(self):
        """
        Trả về: ok, rgb(bgr8), depth(16UC1), imu_msg (Imu hoặc None)
        Ghi chú: frameset có thể chứa motion frames; ta gom gyro/accel gần nhất.
        """
        if self.pipe is None:
            rgb = np.zeros((480, 640, 3), np.uint8)
            depth = np.zeros((480, 640), np.uint16)
            return True, rgb, depth, None

        frames = self.pipe.wait_for_frames()
        # Lấy các motion frame nếu có
        for f in frames:
            try:
                if f.is_motion_frame():
                    md = f.as_motion_frame().get_motion_data()  # (x,y,z)
                    st = f.get_profile().stream_type()
                    if st == rs.stream.gyro:
                        # rad/s
                        self._last_gyro = (md.x, md.y, md.z)
                    elif st == rs.stream.accel:
                        # m/s^2
                        self._last_accel = (md.x, md.y, md.z)
            except Exception:
                pass

        frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return False, None, None, None

        rgb = np.asanyarray(color.get_data())
        depth_arr = np.asanyarray(depth.get_data()).astype(np.uint16)

        # Đóng gói IMU theo timestamp ROS hiện tại
        now_sec = self.node.get_clock().now().nanoseconds / 1e9
        imu_msg = self._pack_imu(now_sec)
        return True, rgb, depth_arr, imu_msg

    def stop(self):
        try:
            if self.pipe:
                self.pipe.stop()
        except Exception:
            pass
