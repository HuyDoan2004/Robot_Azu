from sensor_msgs.msg import CameraInfo

def make_camera_info(width, height, fx, fy, cx, cy):
    ci = CameraInfo()
    ci.width = width
    ci.height = height
    ci.k = [fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0]
    ci.p = [fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0]
    ci.d = [0.0]*5
    ci.r = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
    return ci
from sensor_msgs.msg import CameraInfo

def make_camera_info(width, height, fx, fy, cx, cy):
    ci = CameraInfo()
    ci.width = width
    ci.height = height
    ci.k = [fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0]
    ci.p = [fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0]
    ci.d = [0.0]*5
    ci.r = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
    return ci
