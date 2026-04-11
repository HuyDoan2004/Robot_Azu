import math, numpy as np
from sensor_msgs.msg import Imu

def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    w = cr*cp*cy + sr*sp*sy
    q = np.array([x,y,z,w], dtype=float)
    return q/np.linalg.norm(q)

class ImuComplementary:
    """
    Complementary filter:
      - Gyro integrate -> r,p,y
      - Blend r,p về accel (gravity) theo alpha=exp(-dt/tau)
      - Yaw = chỉ gyro (không dùng từ trường)
    """
    def __init__(self, tau=0.5, accel_beta=0.2, accel_gate_g=0.35):
        self.tau = float(tau)            # hằng thời gian cho blend (s)
        self.accel_beta = float(accel_beta)  # EMA cho accel (0..1)
        self.accel_gate_g = float(accel_gate_g)  # gate khi |a|-1g quá lớn (g)
        self.roll = self.pitch = self.yaw = 0.0
        self._last_t = None
        self._ax = self._ay = self._az = None  # EMA accel

    def update(self, imu_msg: Imu):
        t = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec*1e-9
        if self._last_t is None:
            self._last_t = t
            return rpy_to_quat(self.roll, self.pitch, self.yaw)
        dt = max(1e-4, t - self._last_t)
        self._last_t = t

        # Gyro (rad/s)
        gx, gy, gz = float(imu_msg.angular_velocity.x), float(imu_msg.angular_velocity.y), float(imu_msg.angular_velocity.z)
        self.roll  += gx * dt
        self.pitch += gy * dt
        self.yaw   += gz * dt

        # Accel EMA
        ax, ay, az = float(imu_msg.linear_acceleration.x), float(imu_msg.linear_acceleration.y), float(imu_msg.linear_acceleration.z)
        if self._ax is None:
            self._ax, self._ay, self._az = ax, ay, az
        else:
            b = self.accel_beta
            self._ax = (1-b)*self._ax + b*ax
            self._ay = (1-b)*self._ay + b*ay
            self._az = (1-b)*self._az + b*az

        # Gate theo độ lệch khỏi 1g
        g = math.sqrt(self._ax**2 + self._ay**2 + self._az**2)
        if abs(g - 9.80665) < self.accel_gate_g*9.80665:
            axn, ayn, azn = self._ax/g, self._ay/g, self._az/g
            roll_acc  = math.atan2(ayn, azn)
            pitch_acc = math.atan2(-axn, math.sqrt(ayn*ayn + azn*azn))

            # Complementary blend với alpha theo dt
            alpha = math.exp(-dt/self.tau)  # 0..1
            self.roll  = alpha*self.roll  + (1.0-alpha)*roll_acc
            self.pitch = alpha*self.pitch + (1.0-alpha)*pitch_acc

        return rpy_to_quat(self.roll, self.pitch, self.yaw)
