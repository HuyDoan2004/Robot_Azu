import numpy as np

def quat_from_rpy(roll, pitch, yaw):
    # xyzw
    cr = np.cos(roll/2); sr = np.sin(roll/2)
    cp = np.cos(pitch/2); sp = np.sin(pitch/2)
    cy = np.cos(yaw/2); sy = np.sin(yaw/2)
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    w = cr*cp*cy + sr*sp*sy
    return np.array([x,y,z,w])

def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_mul(a,b):
    ax,ay,az,aw = a; bx,by,bz,bw = b
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return np.array([x,y,z,w])
